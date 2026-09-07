"use strict";

/* ============ KaTeX 数式レンダリング ============ */
if (window.renderMathInElement) {
  window.renderMathInElement(document.body, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  });
}

function texToHTML(tex) {
  if (window.katex) {
    return window.katex.renderToString(tex, { throwOnError: false });
  }
  return tex;
}

/* ============ Python(NetworkX)側 ============ */
const PYTHON_CODE = `
import json
import random
import networkx as nx


def _load_graph(pajek_text):
    # Pajek 形式を読み込み、自己ループを除いた無向単純グラフ(ノード 0..n-1)に変換する
    Gm = nx.parse_pajek(pajek_text.splitlines())
    order = list(Gm.nodes())
    index = {v: i for i, v in enumerate(order)}
    G = nx.Graph()
    G.add_nodes_from(range(len(order)))
    G.add_edges_from((index[u], index[v]) for u, v in Gm.edges() if u != v)

    # Pajek ファイルに座標があれば利用する
    coords = []
    for v in order:
        d = Gm.nodes[v]
        try:
            coords.append((float(d["x"]), float(d["y"])))
        except (KeyError, TypeError, ValueError):
            coords = None
            break
    return G, coords


def _layout(G, coords, seed):
    n = G.number_of_nodes()
    if coords is not None and len(coords) == n:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        if max(xs) - min(xs) > 0 or max(ys) - min(ys) > 0:
            return xs, ys
    pos = nx.spring_layout(G, seed=seed)
    return (
        [float(pos[v][0]) for v in range(n)],
        [float(pos[v][1]) for v in range(n)],
    )


def _removal_order(G, strategy, rng):
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    if strategy == "random":
        return nodes
    if strategy == "hd":
        # HD 攻撃: 初期次数の高い順に除去(再計算なし、同次数はランダム順)
        deg = dict(G.degree())
        nodes.sort(key=lambda v: deg[v], reverse=True)
        return nodes
    # rb: RB 攻撃。除去のたびに媒介中心性を再計算し、最大のノードを除去する。
    # 除去で値が変わるのは除去ノードが属していた成分だけなので、その成分のみ再計算する
    # (未正規化の媒介中心性は成分をまたぐ最短路を持たないため全体計算と一致する)
    H = G.copy()
    tiebreak = {v: rng.random() for v in G.nodes()}
    order = []
    bc = nx.betweenness_centrality(H, normalized=False)
    while H.number_of_nodes():
        if H.number_of_edges() == 0:
            # エッジが尽きると媒介中心性は全て 0 なので残りはランダム順で除去
            rest = list(H.nodes())
            rest.sort(key=lambda v: tiebreak[v])
            order.extend(rest)
            break
        v = max(H.nodes(), key=lambda x: (bc[x], tiebreak[x]))
        comp = nx.node_connected_component(H, v)
        H.remove_node(v)
        bc.pop(v, None)
        order.append(v)
        affected = comp - {v}
        if affected:
            bc.update(nx.betweenness_centrality(H.subgraph(affected), normalized=False))
    return order


def _curve(G, order):
    # 除去順を逆再生し union-find でノードを追加していくと、
    # 全ての除去数 k について成分サイズが O(N+E) 級で求まる
    n = G.number_of_nodes()
    adj = [[] for _ in range(n)]
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    parent = [-1] * n
    size = [0] * n
    present = [False] * n
    cnt = {}  # 成分サイズ -> その成分数

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def cnt_add(s, d):
        c = cnt.get(s, 0) + d
        if c:
            cnt[s] = c
        else:
            cnt.pop(s, None)

    S1 = [0.0] * (n + 1)
    S2 = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        v = order[i]
        parent[v] = v
        size[v] = 1
        present[v] = True
        cnt_add(1, 1)
        for u in adj[v]:
            if present[u]:
                ru = find(u)
                rv = find(v)
                if ru != rv:
                    if size[ru] < size[rv]:
                        ru, rv = rv, ru
                    cnt_add(size[ru], -1)
                    cnt_add(size[rv], -1)
                    parent[rv] = ru
                    size[ru] += size[rv]
                    cnt_add(size[ru], 1)
        s1 = 0
        s2 = 0
        for s in cnt:
            if s > s1:
                s2 = s1
                s1 = s
            elif s > s2:
                s2 = s
        if cnt.get(s1, 0) > 1:
            s2 = s1
        S1[i] = s1
        S2[i] = s2
    return S1, S2


def run_simulation(params_json):
    p = json.loads(params_json)
    strategy = p["strategy"]
    seed = int(p["seed"])

    G, coords = _load_graph(p["pajek"])
    n = G.number_of_nodes()
    if n < 2:
        raise ValueError("ネットワークのノード数が少なすぎます(N >= 2 が必要)")

    xs, ys = _layout(G, coords, seed)
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0
    norm_pos = [
        [(xs[v] - cx) / span + 0.5, (ys[v] - cy) / span + 0.5]
        for v in range(n)
    ]

    rng = random.Random(seed)
    order = _removal_order(G, strategy, rng)
    S1, S2 = _curve(G, order)
    S = [S1[k] / n for k in range(n + 1)]
    s2 = [float(S2[k]) for k in range(n + 1)]

    # 頑健性指標 R = (1/N) * sum_{q=1/N..1} S(q)/N
    R = sum(S[1:]) / n
    # 臨界値 q_c: 第二最大連結成分サイズが最大となる q
    kc = max(range(n + 1), key=lambda k: s2[k])

    result = {
        "n": n,
        "edges": [[int(u), int(v)] for u, v in G.edges()],
        "pos": norm_pos,
        "degree": [int(G.degree(v)) for v in range(n)],
        "order": [int(v) for v in order],
        "q": [k / n for k in range(n + 1)],
        "S": S,
        "s2": s2,
        "R": R,
        "qc": kc / n,
        "meta": {
            "edges": G.number_of_edges(),
            "avg_deg": 2.0 * G.number_of_edges() / n,
        },
    }
    return json.dumps(result)
`;

/* ============ 状態 ============ */
let pyodide = null;
let pyRun = null;
let sim = null;      // run_simulation の結果
let curK = 0;        // 現在の除去ノード数
let playing = false;
let playPrev = 0;
let playAcc = 0;

const netCache = new Map();  // ファイル名 -> Pajek テキスト

const $ = (id) => document.getElementById(id);

const els = {
  loader: $("loader"),
  loaderMsg: $("loader-msg"),
  network: $("in-network"),
  strategy: $("in-strategy"),
  run: $("btn-run"),
  play: $("btn-play"),
  reset: $("btn-reset"),
  q: $("in-q"),
  qVal: $("q-val"),
  qRemoved: $("q-removed"),
  qTotal: $("q-total"),
  tileQ: $("tile-q"),
  tileS1: $("tile-s1"),
  tileS2: $("tile-s2"),
  tileNc: $("tile-nc"),
  canvas: $("net-canvas"),
  canvasWrap: $("net-canvas-wrap"),
  netMeta: $("net-meta"),
  viz: $("viz"),
  tiles: $("tiles"),
  chart1Sub: $("chart1-sub"),
  chart2Sub: $("chart2-sub"),
  netPlaceholder: $("net-placeholder"),
};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ============ 現在の q に対する成分分解(JS 側 union-find、スライダー応答用) ============ */
function componentInfo(k) {
  const n = sim.n;
  const removed = new Uint8Array(n);
  for (let i = 0; i < k; i++) removed[sim.order[i]] = 1;
  const parent = new Int32Array(n);
  const size = new Int32Array(n).fill(1);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (const [u, v] of sim.edges) {
    if (removed[u] || removed[v]) continue;
    let ru = find(u), rv = find(v);
    if (ru !== rv) {
      if (size[ru] < size[rv]) [ru, rv] = [rv, ru];
      parent[rv] = ru;
      size[ru] += size[rv];
    }
  }
  let r1 = -1, r2 = -1, ncomp = 0;
  for (let v = 0; v < n; v++) {
    if (removed[v] || find(v) !== v) continue;
    ncomp++;
    if (r1 === -1 || size[v] > size[r1]) { r2 = r1; r1 = v; }
    else if (r2 === -1 || size[v] > size[r2]) { r2 = v; }
  }
  // cat: 0=その他, 1=最大成分, 2=第二成分, 3=除去済み
  const cat = new Int8Array(n);
  for (let v = 0; v < n; v++) {
    if (removed[v]) { cat[v] = 3; continue; }
    const r = find(v);
    cat[v] = r === r1 ? 1 : r === r2 ? 2 : 0;
  }
  return {
    cat,
    s1: r1 >= 0 ? size[r1] : 0,
    s2: r2 >= 0 ? size[r2] : 0,
    ncomp,
  };
}

/* ============ ネットワーク描画 ============ */
function drawNetwork() {
  if (!sim) return;
  const canvas = els.canvas;
  // キャンバスは #net-canvas-wrap いっぱいに描く(カードが右列と同じ高さに伸びるため、
  // 描画領域の縦幅はレイアウトから受け取る)
  const wrap = els.canvasWrap;
  const cssW = wrap.clientWidth;
  const cssH = wrap.clientHeight;
  const info = componentInfo(curK);
  if (cssW <= 0 || cssH <= 0) return info;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = 14;
  const scale = Math.min(cssW, cssH) - pad * 2;
  const ox = (cssW - scale) / 2;
  const oy = (cssH - scale) / 2;
  const X = (v) => ox + sim.pos[v][0] * scale;
  const Y = (v) => oy + sim.pos[v][1] * scale;

  const colGrid = cssVar("--grid");
  const colAxis = cssVar("--axis");
  const colMuted = cssVar("--ink-muted");
  const colS1 = cssVar("--s1");
  const colS2 = cssVar("--s2");

  // 残存エッジ
  ctx.strokeStyle = colGrid;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  for (const [u, v] of sim.edges) {
    if (info.cat[u] === 3 || info.cat[v] === 3) continue;
    ctx.moveTo(X(u), Y(u));
    ctx.lineTo(X(v), Y(v));
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  const base = Math.max(2.2, Math.min(6.5, 130 / Math.sqrt(sim.n)));
  const avgDeg = Math.max(sim.meta.avg_deg, 0.5);
  const radius = (v) => base * (0.7 + 0.45 * Math.sqrt(sim.degree[v] / avgDeg));

  // 除去済み → その他 → 第二 → 最大 の順に上へ重ねる
  for (const pass of [3, 0, 2, 1]) {
    for (let v = 0; v < sim.n; v++) {
      if (info.cat[v] !== pass) continue;
      const r = pass === 3 ? Math.max(1.4, base * 0.4) : radius(v);
      ctx.beginPath();
      ctx.arc(X(v), Y(v), r, 0, Math.PI * 2);
      if (pass === 3) {
        ctx.strokeStyle = colAxis;
        ctx.lineWidth = 1;
        ctx.stroke();
      } else {
        ctx.fillStyle = pass === 1 ? colS1 : pass === 2 ? colS2 : colMuted;
        ctx.fill();
      }
    }
  }
  return info;
}

/* ============ SVG ラインチャート ============ */
const SVGNS = "http://www.w3.org/2000/svg";
const chartRegistry = [];

function svgEl(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  return e;
}

function niceCeil(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (m * p >= v) return m * p;
  }
  return 10 * p;
}

function createLineChart(root, cfg) {
  // cfg: { colorVar, seriesName, seriesTex, fixedYMax|null, yFmt, area }
  const M = { top: 10, right: 14, bottom: 40, left: 46 };
  const plotH = 190;
  const chart = {
    root, cfg,
    data: null,       // { q: [], y: [] }
    yMax: cfg.fixedYMax || 1,
    W: 0,
    svg: null,
    tooltip: null,
    parts: {},
  };

  const xPos = (q) => M.left + q * (chart.W - M.left - M.right);
  const yPos = (y) => M.top + (1 - y / chart.yMax) * plotH;

  function render() {
    chart.W = Math.max(root.clientWidth, 220);
    root.textContent = "";
    const H = M.top + plotH + M.bottom;
    const svg = svgEl("svg", {
      viewBox: "0 0 " + chart.W + " " + H,
      width: chart.W,
      height: H,
      role: "img",
      "aria-label": cfg.seriesName,
    });
    chart.svg = svg;

    if (chart.data && !cfg.fixedYMax) {
      chart.yMax = niceCeil(Math.max(...chart.data.y, 1) * 1.05);
    }

    // 横グリッド + y 目盛
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const yv = (chart.yMax / yTicks) * i;
      const py = yPos(yv);
      if (i > 0) {
        svg.appendChild(svgEl("line", {
          class: "gridline", x1: M.left, x2: chart.W - M.right, y1: py, y2: py,
        }));
      }
      const t = svgEl("text", {
        class: "ticktext", x: M.left - 7, y: py + 4, "text-anchor": "end",
      });
      t.textContent = (cfg.yTickFmt || cfg.yFmt)(yv);
      svg.appendChild(t);
    }
    // x 軸
    const baseY = M.top + plotH;
    svg.appendChild(svgEl("line", {
      class: "axisline", x1: M.left, x2: chart.W - M.right, y1: baseY, y2: baseY,
    }));
    for (let i = 0; i <= 5; i++) {
      const qv = i / 5;
      const t = svgEl("text", {
        class: "ticktext", x: xPos(qv), y: baseY + 16, "text-anchor": "middle",
      });
      t.textContent = qv.toFixed(1);
      svg.appendChild(t);
    }
    const xt = svgEl("text", {
      class: "axistitle",
      x: (M.left + chart.W - M.right) / 2,
      y: baseY + 32,
      "text-anchor": "middle",
    });
    xt.textContent = "攻撃率 ";
    const xtQ = svgEl("tspan", { "font-style": "italic" });
    xtQ.textContent = "q";
    xt.appendChild(xtQ);
    svg.appendChild(xt);

    if (chart.data) {
      const pts = chart.data.q.map((q, i) =>
        xPos(q).toFixed(1) + "," + yPos(chart.data.y[i]).toFixed(1));
      if (cfg.area) {
        const d = "M" + xPos(chart.data.q[0]) + "," + baseY +
          " L" + pts.join(" L") +
          " L" + xPos(chart.data.q[chart.data.q.length - 1]) + "," + baseY + " Z";
        svg.appendChild(svgEl("path", {
          d, fill: "var(" + cfg.colorVar + ")", "fill-opacity": "0.1", stroke: "none",
        }));
      }
      svg.appendChild(svgEl("path", {
        class: "series-line",
        d: "M" + pts.join(" L"),
        stroke: "var(" + cfg.colorVar + ")",
      }));
    }

    // 臨界値 q_c の破線マーカー
    if (sim && typeof sim.qc === "number") {
      const px = xPos(sim.qc);
      svg.appendChild(svgEl("line", {
        class: "qcline", x1: px, x2: px, y1: M.top, y2: baseY,
      }));
      const lbl = svgEl("text", {
        class: "qclabel",
        x: px + (sim.qc < 0.8 ? 5 : -5),
        y: M.top + 12,
        "text-anchor": sim.qc < 0.8 ? "start" : "end",
      });
      const lq = svgEl("tspan", { "font-style": "italic" });
      lq.textContent = "q";
      const lc = svgEl("tspan", { dy: "2", "font-size": "8" });
      lc.textContent = "c";
      lbl.append(lq, lc);
      svg.appendChild(lbl);
    }

    // 現在の q マーカーとクロスヘア
    chart.parts.qline = svgEl("line", {
      class: "qline", y1: M.top, y2: baseY, x1: 0, x2: 0, visibility: "hidden",
    });
    chart.parts.qdot = svgEl("circle", {
      class: "qdot", r: 4.5, fill: "var(" + cfg.colorVar + ")", visibility: "hidden",
    });
    chart.parts.cross = svgEl("line", {
      class: "crosshair", y1: M.top, y2: baseY, x1: 0, x2: 0, visibility: "hidden",
    });
    svg.appendChild(chart.parts.qline);
    svg.appendChild(chart.parts.cross);
    svg.appendChild(chart.parts.qdot);

    // ポインタ用オーバーレイ(ヒット領域はプロット全体)
    const ov = svgEl("rect", {
      class: "overlay-rect",
      x: M.left, y: M.top,
      width: chart.W - M.left - M.right, height: plotH,
      fill: "transparent",
    });
    svg.appendChild(ov);
    root.appendChild(svg);

    const tt = document.createElement("div");
    tt.className = "tooltip";
    const ttQ = document.createElement("div");
    ttQ.className = "tt-q";
    const ttRow = document.createElement("div");
    ttRow.className = "tt-row";
    const ttKey = document.createElement("span");
    ttKey.className = "tt-key";
    ttKey.style.background = "var(" + cfg.colorVar + ")";
    const ttVal = document.createElement("span");
    ttVal.className = "tt-val";
    const ttName = document.createElement("span");
    ttName.className = "tt-name";
    if (cfg.seriesTex) ttName.innerHTML = texToHTML(cfg.seriesTex);
    else ttName.textContent = cfg.seriesName;
    ttRow.append(ttKey, ttVal, ttName);
    tt.append(ttQ, ttRow);
    root.appendChild(tt);
    chart.tooltip = { box: tt, q: ttQ, val: ttVal };

    let scrubbing = false;
    const idxFromEvent = (ev) => {
      const rect = svg.getBoundingClientRect();
      const px = (ev.clientX - rect.left) * (chart.W / rect.width);
      const q = (px - M.left) / (chart.W - M.left - M.right);
      return Math.round(Math.min(1, Math.max(0, q)) * sim.n);
    };
    ov.addEventListener("pointermove", (ev) => {
      if (!sim) return;
      const i = idxFromEvent(ev);
      hoverAt(i, chart, ev);
      if (scrubbing) setK(i);
    });
    ov.addEventListener("pointerdown", (ev) => {
      if (!sim) return;
      scrubbing = true;
      ov.setPointerCapture(ev.pointerId);
      setK(idxFromEvent(ev));
    });
    ov.addEventListener("pointerup", () => { scrubbing = false; });
    ov.addEventListener("pointerleave", () => { if (!scrubbing) hoverAt(null); });

    positionMarker();
  }

  function positionMarker() {
    if (!sim || !chart.data || !chart.parts.qline) return;
    const q = curK / sim.n;
    const px = xPos(q);
    const py = yPos(chart.data.y[curK]);
    chart.parts.qline.setAttribute("x1", px);
    chart.parts.qline.setAttribute("x2", px);
    chart.parts.qline.setAttribute("visibility", "visible");
    chart.parts.qdot.setAttribute("cx", px);
    chart.parts.qdot.setAttribute("cy", py);
    chart.parts.qdot.setAttribute("visibility", "visible");
  }

  function showCross(i) {
    if (!chart.parts.cross || !sim) return;
    if (i === null) {
      chart.parts.cross.setAttribute("visibility", "hidden");
      return;
    }
    const px = xPos(i / sim.n);
    chart.parts.cross.setAttribute("x1", px);
    chart.parts.cross.setAttribute("x2", px);
    chart.parts.cross.setAttribute("visibility", "visible");
  }

  function showTooltip(i, ev) {
    const tt = chart.tooltip;
    if (!tt) return;
    if (i === null || !chart.data) {
      tt.box.style.display = "none";
      return;
    }
    tt.q.textContent = "q = " + (i / sim.n).toFixed(3);
    tt.val.textContent = cfg.yFmt(chart.data.y[i]);
    tt.box.style.display = "block";
    const rect = root.getBoundingClientRect();
    let lx = ev.clientX - rect.left + 14;
    const ly = ev.clientY - rect.top - 14;
    if (lx + tt.box.offsetWidth > rect.width - 4) {
      lx = ev.clientX - rect.left - tt.box.offsetWidth - 14;
    }
    tt.box.style.left = lx + "px";
    tt.box.style.top = Math.max(0, ly) + "px";
  }

  chart.render = render;
  chart.setData = (q, y) => {
    chart.data = q && y ? { q, y } : null;
    if (!chart.data) chart.yMax = cfg.fixedYMax || 1;
    render();
  };
  chart.positionMarker = positionMarker;
  chart.showCross = showCross;
  chart.showTooltip = showTooltip;
  chartRegistry.push(chart);
  return chart;
}

function hoverAt(i, sourceChart, ev) {
  for (const c of chartRegistry) {
    c.showCross(i);
    if (c === sourceChart && ev) c.showTooltip(i, ev);
    else c.showTooltip(null);
  }
}

const chartS = createLineChart(document.getElementById("chart-s"), {
  colorVar: "--s1",
  seriesName: "S(q)/N",
  seriesTex: "S(q)/N",
  fixedYMax: 1,
  yFmt: (v) => v.toFixed(3),
  yTickFmt: (v) => v.toFixed(1),
  area: true,
});
const chartS2 = createLineChart(document.getElementById("chart-s2"), {
  colorVar: "--s2",
  seriesName: "⟨s⟩",
  seriesTex: "\\langle s \\rangle",
  fixedYMax: null,
  yFmt: (v) => v.toFixed(2),
  yTickFmt: (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(1)),
  area: false,
});

/* ============ 待機状態(ネットワーク・攻撃方法が未選択、または未実行) ============ */
function setIdle() {
  stopPlay();
  sim = null;
  curK = 0;
  document.body.classList.add("is-idle");
  els.q.value = 0;
  els.q.disabled = true;
  els.play.disabled = true;
  els.reset.disabled = true;
  els.qVal.textContent = "–";
  els.qRemoved.textContent = "–";
  els.qTotal.textContent = "–";
  els.tileQ.textContent = "–";
  els.tileS1.textContent = "–";
  els.tileS2.textContent = "–";
  els.tileNc.textContent = "–";
  els.netMeta.textContent = "";
  els.chart1Sub.innerHTML = "頑健性指標 " + texToHTML("R");
  els.chart2Sub.innerHTML = "臨界値 " + texToHTML("q_c");
  for (const c of chartRegistry) c.setData(null, null);
}

function syncRunButton() {
  els.run.disabled = !pyRun || !els.network.value || !els.strategy.value;
}
els.network.addEventListener("change", syncRunButton);
els.strategy.addEventListener("change", syncRunButton);

/* ============ q の更新 ============ */
function setK(k) {
  if (!sim) return;
  curK = Math.min(sim.n, Math.max(0, Math.round(k)));
  els.q.value = curK;
  const q = curK / sim.n;
  els.qVal.textContent = q.toFixed(2);
  els.qRemoved.textContent = String(curK);

  const info = drawNetwork();
  els.tileQ.textContent = q.toFixed(2);
  els.tileS1.textContent = (info.s1 / sim.n).toFixed(3);
  els.tileS2.textContent = String(info.s2);
  els.tileNc.textContent = String(info.ncomp);

  for (const c of chartRegistry) c.positionMarker();
}

/* ============ アニメーション ============ */
function stopPlay() {
  playing = false;
  els.play.textContent = "▶ 再生";
}
function tickPlay(ts) {
  if (!playing || !sim) return;
  if (playPrev) {
    playAcc += ((ts - playPrev) / 1000) * (sim.n / 10);  // 全崩壊まで約 10 秒
    const step = Math.floor(playAcc);
    if (step > 0) {
      playAcc -= step;
      setK(curK + step);
    }
  }
  playPrev = ts;
  if (curK >= sim.n) { stopPlay(); return; }
  requestAnimationFrame(tickPlay);
}
els.play.addEventListener("click", () => {
  if (!sim) return;
  if (playing) { stopPlay(); return; }
  if (curK >= sim.n) setK(0);
  playing = true;
  playPrev = 0;
  playAcc = 0;
  els.play.textContent = "⏸ 停止";
  requestAnimationFrame(tickPlay);
});
els.reset.addEventListener("click", () => { stopPlay(); setK(0); });
els.q.addEventListener("input", () => { stopPlay(); setK(Number(els.q.value)); });

/* ============ データ表 ============ */
/* ============ ネットワークデータの取得 ============ */
async function fetchNetworkText(file) {
  if (netCache.has(file)) return netCache.get(file);
  const res = await fetch(file, { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(file + " を取得できませんでした(HTTP " + res.status + ")");
  }
  const text = await res.text();
  netCache.set(file, text);
  return text;
}

/* ============ シミュレーション実行 ============ */
function readParams() {
  // 試行回数 1 回・乱数シード 1 に固定
  return {
    strategy: els.strategy.value,
    seed: 1,
  };
}

async function runSimulation() {
  if (!pyRun || !els.network.value || !els.strategy.value) return;
  stopPlay();
  els.run.disabled = true;
  els.run.textContent = "計算中…";
  els.viz.classList.add("recomputing");
  els.tiles.classList.add("recomputing");
  // オーバーレイ描画を先に反映させてから重い計算に入る
  await new Promise((r) => setTimeout(r, 30));
  try {
    const params = readParams();
    const file = els.network.value;
    params.pajek = await fetchNetworkText(file);
    const json = pyRun(JSON.stringify(params));
    sim = JSON.parse(json);
    document.body.classList.remove("is-idle");
    els.q.disabled = false;
    els.play.disabled = false;
    els.reset.disabled = false;

    els.q.max = sim.n;
    els.qTotal.textContent = String(sim.n);
    els.netMeta.innerHTML =
      "総ノード数 " + texToHTML("N = " + sim.n) +
      " / 総リンク数 " + texToHTML("M = " + sim.meta.edges);
    els.chart1Sub.innerHTML =
      "頑健性指標 " + texToHTML("R = " + sim.R.toFixed(4));
    els.chart2Sub.innerHTML =
      "臨界値 " + texToHTML("q_c = " + sim.qc.toFixed(4));

    chartS.setData(sim.q, sim.S);
    chartS2.setData(sim.q, sim.s2);
    setK(0);
  } catch (err) {
    console.error(err);
    setIdle();
    alert("シミュレーションに失敗しました: " + err.message);
  } finally {
    els.viz.classList.remove("recomputing");
    els.tiles.classList.remove("recomputing");
    els.run.textContent = "実行";
    syncRunButton();
  }
}
els.run.addEventListener("click", runSimulation);

/* ============ リサイズ・テーマ変更 ============ */
let resizeTimer = null;
const resizeObserver = new ResizeObserver(() => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    for (const c of chartRegistry) c.render();
    if (sim) setK(curK);
  }, 100);
});
resizeObserver.observe(document.getElementById("viz"));
// 右列の高さが変わるとネットワークカードの描画領域も変わるので、こちらも監視する
resizeObserver.observe(els.canvasWrap);

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (sim) setK(curK);
});

/* ============ 起動 ============ */
async function loadManifest() {
  const res = await fetch("networks.json", { cache: "no-cache" });
  if (!res.ok) {
    throw new Error("networks.json を取得できませんでした(HTTP " + res.status + ")");
  }
  const manifest = await res.json();
  const list = manifest.networks || [];
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "―― 選択してください ――";
  els.network.appendChild(placeholder);
  for (const net of list) {
    const opt = document.createElement("option");
    opt.value = net.file;
    opt.textContent = net.label || net.file;
    els.network.appendChild(opt);
  }
  if (els.network.options.length < 2) {
    throw new Error("networks.json にネットワークが定義されていません");
  }
  els.network.value = "";
}

async function boot() {
  setIdle();
  try {
    if (location.protocol === "file:") {
      throw new Error(
        "file: で直接開くとネットワークデータを取得できません。" +
        "ローカル HTTP サーバー経由で開いてください(例: python3 -m http.server)"
      );
    }
    els.loaderMsg.textContent = "ネットワーク一覧を読み込み中…";
    await loadManifest();
    els.loaderMsg.textContent = "Pyodide を読み込み中…(初回は数十秒かかります)";
    pyodide = await loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/",
    });
    els.loaderMsg.textContent = "NetworkX をインストール中…";
    await pyodide.loadPackage(["networkx", "numpy"]);
    pyodide.runPython(PYTHON_CODE);
    pyRun = pyodide.globals.get("run_simulation");
    els.loader.style.display = "none";
    syncRunButton();
  } catch (err) {
    console.error(err);
    els.loaderMsg.textContent = "読み込みに失敗しました: " + err.message;
  }
}
boot();
