// axis_lanes.js — THE AXIS. Five lanes, one pixel-aligned lambda scale.

import { setupCanvas, lamColor, C } from './charts.js';
import { S, setS, gridLams, DENS, WTS, AMP, CLOCKS, tOfLam_rf, shiftLam } from './axis_main.js';

let cv, ctx, W, H;
const ML = 122, MR = 158;
const LANES = {
  strip: { y: 30, h: 22 },
  clocks: { y: 66, h: 96 },
  train: { y: 172, h: 106 },
  infer: { y: 288, h: 42 },
  distill: { y: 340, h: 70 },
  cond: { y: 420, h: 66 },
};
H = 512;

export function initLanes(canvas) {
  cv = canvas;
  const resize = () => {
    W = Math.min(1480, cv.parentElement.clientWidth);
    ctx = setupCanvas(cv, W, H);
    setS({});   // canvas was reset — repaint everything (no-op before boot completes)
  };
  resize();
  window.addEventListener('resize', resize);
}

const X = l => ML + (l - S.lamMin) / (S.lamMax - S.lamMin) * (W - ML - MR);
const LamOfX = x => S.lamMin + (x - ML) / (W - ML - MR) * (S.lamMax - S.lamMin);
const MONO = 'Consolas, monospace';
const dim = (on) => on ? 1 : 0.4;

function laneLabel(lane, text, on) {
  ctx.globalAlpha = dim(on);
  ctx.font = `10.5px ${MONO}`; ctx.fillStyle = on ? '#F2B03D' : '#5C6675';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText(text, 10, LANES[lane].y - 1);
  ctx.globalAlpha = 1;
}

export function drawLanes() {
  if (!ctx) return;
  ctx.clearRect(0, 0, W, H);
  const { mode } = S;

  // ---- readout chips ----
  const a2 = 1 / (1 + Math.exp(-S.lam)), a = Math.sqrt(a2), s = Math.sqrt(1 - a2);
  const t = tOfLam_rf(S.lam), sig = Math.exp(-S.lam / 2), phi = Math.atan(sig);
  ctx.font = `11.5px ${MONO}`; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
  const chips = [
    ['λ = ' + S.lam.toFixed(2), lamColor(S.lam)],
    [`SNR ${(4.3429 * S.lam).toFixed(1)} dB`, '#9AA3B2'],
    [`α ${a.toFixed(3)}`, '#9AA3B2'], [`σ ${s.toFixed(3)}`, '#9AA3B2'],
    [`t_RF ${t.toFixed(3)}`, '#D65077'], [`σ_EDM ${sig < 100 ? sig.toPrecision(3) : sig.toFixed(0)}`, '#C28324'],
    [`φ ${phi.toFixed(3)}`, '#1FA9A3'],
  ];
  let cx = ML;
  for (const [txt, col] of chips) {
    ctx.fillStyle = col; ctx.fillText(txt, cx, 6);
    cx += ctx.measureText(txt).width + 22;
  }
  ctx.fillStyle = '#5C6675';
  ctx.textAlign = 'right';
  ctx.fillText(mode === 'video' ? '拖动无效 · 观看帧游标' : '拖动光谱条移动 λ', W - 8, 6);

  // ---- spectrum strip ----
  const st = LANES.strip;
  for (let px = ML; px < W - MR; px++) {
    ctx.fillStyle = lamColor(LamOfX(px));
    ctx.fillRect(px, st.y, 1, st.h);
  }
  ctx.strokeStyle = 'rgba(232,228,218,0.25)';
  ctx.strokeRect(ML, st.y, W - ML - MR, st.h);
  ctx.font = `10px ${MONO}`; ctx.fillStyle = '#5C6675'; ctx.textAlign = 'left';
  ctx.fillText('← 纯噪声', ML, st.y + st.h + 4);
  ctx.textAlign = 'right'; ctx.fillText('数据 →', W - MR, st.y + st.h + 4);
  // lambda ticks along bottom of strip
  ctx.textAlign = 'center';
  for (let l = -8; l <= 10; l += 2) {
    ctx.fillStyle = '#5C6675'; ctx.fillText(String(l), X(l), st.y + st.h + 4);
    ctx.fillRect(X(l), st.y + st.h, 1, 3);
  }

  // video frames ride the strip
  if (mode === 'video') drawVideoFrames(st);

  // ---- cursor(s): full-height guide ----
  const cxp = X(S.lam);
  ctx.strokeStyle = lamColor(S.lam); ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.moveTo(cxp, st.y - 6); ctx.lineTo(cxp, H - 10); ctx.stroke();
  ctx.fillStyle = '#E8E4DA';
  ctx.beginPath(); ctx.moveTo(cxp, st.y - 7); ctx.lineTo(cxp - 6, st.y - 15); ctx.lineTo(cxp + 6, st.y - 15); ctx.closePath(); ctx.fill();
  if (mode === 'distill') {
    const rxp = X(S.r);
    ctx.strokeStyle = 'rgba(31,169,163,0.9)'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(rxp, st.y - 6); ctx.lineTo(rxp, LANES.distill.y + LANES.distill.h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#1FA9A3';
    ctx.beginPath(); ctx.moveTo(rxp, st.y - 7); ctx.lineTo(rxp - 5, st.y - 14); ctx.lineTo(rxp + 5, st.y - 14); ctx.closePath(); ctx.fill();
    ctx.font = `10px ${MONO}`; ctx.textAlign = 'center';
    ctx.fillText('r(可拖)', rxp, st.y - 26);
  }

  drawClocks();
  drawTrain(mode === 'train');
  if (mode === 'video') drawFilmstrip(); else drawInfer(mode === 'infer');
  drawDistill(mode === 'distill');
  drawCond();
}

// ---- lane: clocks ----
function drawClocks() {
  const L = LANES.clocks; laneLabel('clocks', '各家钟面', true);
  const rows = CLOCKS, rh = L.h / rows.length;
  ctx.font = `10px ${MONO}`;
  rows.forEach((c, i) => {
    const y = L.y + i * rh + rh / 2;
    ctx.strokeStyle = 'rgba(154,163,178,0.14)';
    ctx.beginPath(); ctx.moveTo(ML, y); ctx.lineTo(W - MR, y); ctx.stroke();
    ctx.fillStyle = '#9AA3B2'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(c.label, ML - 8, y);
    ctx.strokeStyle = 'rgba(232,228,218,0.55)';
    for (const l of c.ticks()) {
      if (l < S.lamMin || l > S.lamMax) continue;
      const x = X(l);
      ctx.beginPath(); ctx.moveTo(x, y - rh * 0.32); ctx.lineTo(x, y + rh * 0.32); ctx.stroke();
    }
    ctx.fillStyle = '#E8E4DA'; ctx.textAlign = 'left';
    ctx.fillText(c.fmt(S.lam), W - MR + 8, y);
  });
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#5C6675'; ctx.font = `10px ${MONO}`; ctx.textAlign = 'left';
  ctx.fillText('每行:该约定自己的均匀刻度,映到同一根 λ 轴 —— 均匀的是钟面,不是 λ。RF+shift 整体左移 2ln3=2.20', ML, L.y + L.h + 2);
}

// ---- lane: training (density areas + weight curves) ----
function drawTrain(on) {
  const L = LANES.train; laneLabel('train', '训练:撒点×汇率', on);
  ctx.globalAlpha = dim(on);
  const dH = 56, wY = L.y + dH + 12, wH = L.h - dH - 12;
  // densities, each normalized to its own max within range
  for (const d of DENS) {
    let mx = 0; const vals = [];
    for (let x = ML; x <= W - MR; x += 2) { const v = d.f(LamOfX(x)); vals.push([x, v]); mx = Math.max(mx, v); }
    ctx.beginPath(); ctx.moveTo(ML, L.y + dH);
    for (const [x, v] of vals) ctx.lineTo(x, L.y + dH - v / mx * (dH - 6));
    ctx.lineTo(W - MR, L.y + dH); ctx.closePath();
    ctx.fillStyle = d.color + '2A'; ctx.fill();
    ctx.strokeStyle = d.color; ctx.lineWidth = 1.4; ctx.beginPath();
    vals.forEach(([x, v], i) => i ? ctx.lineTo(x, L.y + dH - v / mx * (dH - 6)) : ctx.moveTo(x, L.y + dH - v / mx * (dH - 6)));
    ctx.stroke();
  }
  ctx.font = `10px ${MONO}`; ctx.fillStyle = '#9AA3B2'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText('撒点 p(λ)', ML - 8, L.y + dH / 2);
  // weights (log scale shapes)
  const lo = -1.5, hi = 4.5; // log10 range
  for (const w of WTS) {
    ctx.strokeStyle = w.color; ctx.lineWidth = 1.2; ctx.setLineDash([5, 3]); ctx.beginPath();
    let started = false;
    for (let x = ML; x <= W - MR; x += 2) {
      const v = Math.log10(w.f(LamOfX(x)));
      const yy = wY + wH - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * wH;
      started ? ctx.lineTo(x, yy) : (ctx.moveTo(x, yy), started = true);
    }
    ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.fillStyle = '#9AA3B2'; ctx.textAlign = 'right';
  ctx.fillText('汇率 w(λ) log', ML - 8, wY + wH / 2);
  // legends
  ctx.textAlign = 'left';
  DENS.forEach((d, i) => { ctx.fillStyle = d.color; ctx.fillText(d.label, W - MR + 8, L.y + 8 + i * 13); });
  WTS.forEach((w, i) => { ctx.fillStyle = w.color; ctx.fillText(w.label, W - MR + 8, wY + 4 + i * 13); });
  ctx.textBaseline = 'alphabetic';
  ctx.globalAlpha = 1;
}

// ---- lane: inference (step grid + walker + eta hops) ----
function drawInfer(on) {
  const L = LANES.infer; laneLabel('infer', '推理:行走', on);
  ctx.globalAlpha = dim(on);
  const y = L.y + L.h / 2;
  ctx.strokeStyle = 'rgba(154,163,178,0.3)';
  ctx.beginPath(); ctx.moveTo(ML, y); ctx.lineTo(W - MR, y); ctx.stroke();
  const grid = gridLams(S.sched, S.N);
  grid.forEach((l, i) => {
    const x = X(l);
    ctx.fillStyle = i <= S.walkK && (S.playing || S.walkK > 0) ? '#F2B03D' : '#8A93A6';
    ctx.beginPath(); ctx.arc(x, y, i === 0 || i === grid.length - 1 ? 4 : 2.6, 0, 7); ctx.fill();
    if (i < grid.length - 1 && S.eta > 0) { // stochastic: hop left, then walk right
      const x2 = X(grid[i + 1]);
      ctx.strokeStyle = 'rgba(226,179,60,0.55)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc((x + x2) / 2 - 4, y - 7, 5, Math.PI * 0.1, Math.PI * 0.9, true); ctx.stroke();
    }
  });
  // walker
  const wl = grid[Math.min(S.walkK, grid.length - 1)];
  ctx.fillStyle = '#E8E4DA';
  ctx.beginPath(); ctx.arc(X(wl), y, 5.5, 0, 7); ctx.fill();
  ctx.fillStyle = '#101318';
  ctx.beginPath(); ctx.arc(X(wl), y, 2.2, 0, 7); ctx.fill();
  ctx.font = `10px ${MONO}`; ctx.fillStyle = '#9AA3B2'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  ctx.fillText(`N=${S.N} · NFE=${S.N}${S.eta > 0 ? ' · η=' + S.eta.toFixed(2) + '(左跳=再注噪)' : ' · η=0 确定性'}`, W - MR + 8, y);
  ctx.textBaseline = 'alphabetic';
  ctx.globalAlpha = 1;
}

// ---- lane: distillation (jumps on the axis) ----
function jump(x1, x2, y, h, head = true) {  // flat bezier hop from x1 to x2, bulge h
  ctx.beginPath(); ctx.moveTo(x1, y);
  ctx.quadraticCurveTo((x1 + x2) / 2, y - h, x2, y);
  ctx.stroke();
  if (head) arrowHead(x2, y, 0, ctx.strokeStyle);
}
function drawDistill(on) {
  const L = LANES.distill; laneLabel('distill', '蒸馏:跳跃', on);
  ctx.globalAlpha = dim(on);
  const yT = L.y + 16, yJ = L.y + 48;
  // teacher: many small hops
  const grid = gridLams(S.sched, Math.max(12, S.N));
  ctx.strokeStyle = '#D65077'; ctx.lineWidth = 1.2;
  for (let i = 0; i < grid.length - 1; i++) {
    jump(X(grid[i]), X(grid[i + 1]), yT, 9, false);
  }
  ctx.font = `10px ${MONO}`; ctx.fillStyle = '#D65077'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText('教师:碎步', ML - 8, yT);
  // student jumps: CM to endpoint; flow-map interval r->lam; PD merge glyph
  const xc = X(S.lam), xr = X(S.r), xe = X(S.lamMax - 0.05);
  // CM / 1-step: cursor -> data end
  ctx.strokeStyle = '#1FA9A3'; ctx.lineWidth = 2;
  jump(xc, xe, yJ, 24);
  // interval flow map r -> cursor (CTM / MeanFlow average velocity)
  ctx.strokeStyle = '#B08FEA'; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]);
  jump(xr, xc, yJ, 17, false);
  ctx.setLineDash([]); arrowHead(Math.max(xr, xc), yJ, 0, '#B08FEA');
  ctx.fillStyle = '#9AA3B2'; ctx.textAlign = 'right';
  ctx.fillText('学生:跳跃', ML - 8, yJ);
  // labels right
  ctx.textAlign = 'left';
  ctx.fillStyle = '#1FA9A3'; ctx.fillText('CM/一步: f(x_t)→x₀', W - MR + 8, yJ - 14);
  ctx.fillStyle = '#B08FEA'; ctx.fillText('CTM/MeanFlow: Φ_{r→t} = ū·Δ', W - MR + 8, yJ);
  ctx.fillStyle = '#5C6675'; ctx.fillText('DMD: 只对端点分布负责,无轨迹', W - MR + 8, yJ + 14);
  // PD glyph: pair-merge
  ctx.fillStyle = '#9AA3B2'; ctx.textAlign = 'left';
  ctx.fillText('PD: 两步并一步 ⌒⌒→⌒  · ReFlow: 不跳,换路(见探头A) · 全部都在学同一个 flow map Φ', ML, L.y + L.h + 2);
  ctx.textBaseline = 'alphabetic';
  ctx.globalAlpha = 1;
}
function arrowHead(x, y, ang, col = '#1FA9A3') {
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 7, y - 4); ctx.lineTo(x - 7, y + 4); ctx.closePath(); ctx.fill();
}

// ---- lane: conditioning / error amplification ----
function drawCond() {
  const L = LANES.cond; laneLabel('cond', '条件数:1 份头误差', true);
  const lo = -2, hi = 2.2; // log10
  for (const aDef of AMP) {
    const isSel = aDef.key === S.head || (S.head === 'x0' && aDef.key === 'x0');
    ctx.strokeStyle = aDef.color; ctx.lineWidth = isSel ? 2.4 : 1.1;
    ctx.globalAlpha = isSel ? 1 : 0.45;
    ctx.beginPath(); let started = false;
    for (let x = ML; x <= W - MR; x += 2) {
      const v = Math.log10(Math.max(1e-3, aDef.f(LamOfX(x))));
      const yy = L.y + L.h - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * L.h;
      started ? ctx.lineTo(x, yy) : (ctx.moveTo(x, yy), started = true);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // unit line
  const yy1 = L.y + L.h - (0 - (-2)) / (2.2 - (-2)) * L.h;
  ctx.strokeStyle = 'rgba(154,163,178,0.35)'; ctx.setLineDash([3, 4]);
  ctx.beginPath(); ctx.moveTo(ML, yy1); ctx.lineTo(W - MR, yy1); ctx.stroke(); ctx.setLineDash([]);
  ctx.font = `10px ${MONO}`; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  AMP.forEach((aDef, i) => {
    ctx.fillStyle = aDef.color; ctx.globalAlpha = aDef.key === S.head ? 1 : 0.6;
    ctx.fillText(aDef.label + (aDef.key === S.head ? ' ←' : ''), W - MR + 8, L.y + 6 + i * 11);
  });
  ctx.globalAlpha = 1;
  const ampNow = AMP.find(x => x.key === S.head).f(S.lam);
  ctx.fillStyle = '#E8E4DA'; ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(`换成 x̂₀ 误差 ×${ampNow.toPrecision(3)}`, ML - 8, L.y + L.h - 4);
  ctx.fillStyle = '#5C6675'; ctx.textAlign = 'left';
  ctx.fillText('虚线=×1 · ε 死于左端,x₀/score 病于两侧,v/û/EDM-D 全程有界 —— 选衣服=选误差住在哪段', ML, L.y + L.h + 4);
}

// ---- video mode: frames on the strip + filmstrip lane ----
const KF = 8;
export function frameLams() {
  const { vpreset: p, vphase: tau, lamMin, lamMax } = S;
  const lo = lamMin + 0.4, hi = lamMax - 0.3, out = [];
  if (p === 'offline') {
    const l = lo + (hi - lo) * tau;
    for (let k = 0; k < KF; k++) out.push(l);
  } else if (p === 'df') {
    for (let k = 0; k < KF; k++) {
      const l = lo + (hi - lo) * (tau * 2.1) - k * 2.1;
      out.push(Math.max(lo, Math.min(hi, l)));
    }
  } else { // stream
    const slot = tau * KF, a = Math.min(KF - 1, Math.floor(slot)), f = slot - a;
    for (let k = 0; k < KF; k++) {
      if (k < a) out.push(hi);
      else if (k === a) out.push(lo + (hi - lo) * Math.min(1, f * 1.15));
      else out.push(lo);
    }
  }
  return out;
}
function drawVideoFrames(st) {
  const lams = frameLams();
  // fan stacked frames: identical lambdas offset upward like a deck of cards
  const seen = [];
  lams.forEach((l, k) => {
    const x = X(l);
    const pile = seen.filter(v => Math.abs(v - x) < 14).length;
    seen.push(x);
    const r = 8, y = st.y + st.h / 2 - pile * 5;
    ctx.fillStyle = lamColor(l);
    ctx.strokeStyle = '#E8E4DA'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.rect(x - r, y - r, r * 2, r * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#101318'; ctx.font = `9px ${MONO}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(k), x, y);
  });
  ctx.textBaseline = 'alphabetic';
}
function drawFilmstrip() {
  const L = LANES.infer; laneLabel('infer', '帧带(时间→)', true);
  const lams = frameLams();
  const cw = 30, y = L.y + 4;
  lams.forEach((l, k) => {
    const x = ML + k * (cw + 6);
    ctx.fillStyle = lamColor(l);
    ctx.fillRect(x, y, cw, 22);
    ctx.strokeStyle = 'rgba(232,228,218,0.3)'; ctx.strokeRect(x, y, cw, 22);
  });
  ctx.font = `10px ${MONO}`; ctx.fillStyle = '#9AA3B2'; ctx.textAlign = 'left';
  const cap = {
    offline: '离线整段:全帧同 λ 齐降 — 质量最好,定长,不可交互(Sora 形状)',
    df: 'Diffusion Forcing:每帧独立 λ,过去干净未来更噪 — 可流式、把上下文当带噪观测(抗漂移疫苗)',
    stream: '流式 AR:历史=完成帧(=λ+∞,进 KV cache),当前帧 1–4 步俯冲(蒸馏),未来排队 — Genie/实时数字人形状',
  }[S.vpreset];
  ctx.fillText(cap, ML + KF * (cw + 6) + 14, y + 15);
  ctx.fillStyle = '#5C6675';
  ctx.fillText('记忆 / 动作 / 世界状态不在这根轴上 —— 这里只画"每帧此刻站在哪"。', ML, L.y + L.h + 2);
}

// ---- hit testing ----
export function lanesHit(set, resetWalk) {
  let dragging = null;
  const pos = e => {
    const r = cv.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  cv.addEventListener('mousedown', e => {
    if (S.mode === 'video') return;
    const { x, y } = pos(e);
    if (x < ML - 10 || x > W - MR + 10) return;
    if (S.mode === 'distill' && Math.abs(x - X(S.r)) < 8 && Math.abs(x - X(S.lam)) > 8) dragging = 'r';
    else dragging = 'lam';
    if (S.playing) { S.playing = false; document.getElementById('ax-play').textContent = '▶ 行走'; }
    move(x);
  });
  window.addEventListener('mousemove', e => { if (dragging) move(pos(e).x); });
  window.addEventListener('mouseup', () => dragging = null);
  function move(x) {
    const l = Math.max(S.lamMin, Math.min(S.lamMax, LamOfX(x)));
    if (dragging === 'r') set({ r: Math.min(l, S.lam - 0.3) });
    else set({ lam: l });
  }
}
