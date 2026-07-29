// axis_cards.js — v2: one card per concept; canvases draw data only, all words live in HTML.
// Every canvas shares the same lambda range and shows the same global cursor.

import { setupCanvas, lamColor } from './charts.js';
import { S, setS, gridLams, DENS, WTS, AMP, TIMEROWS, tOfLam_rf, shiftLam } from './axis_main.js';
import { refChip, refList } from './axis_refs.js';

const MONO = 'Consolas, monospace';
const PAD = 14;
const cvs = {};   // name -> {cv, ctx, w, h}

function reg(id, h) {
  const cv = document.getElementById(id);
  const measure = () => {
    const pw = cv.parentElement.clientWidth;
    return pw >= 320 ? pw : 1128;   // hidden panes report bogus tiny widths — use a sane provisional
  };
  const resize = () => {
    const w = measure();
    if (cvs[id] && Math.abs(cvs[id].w - w) < 4) return;
    cvs[id] = { cv, ctx: setupCanvas(cv, w, h), w, h };
    setS({});
  };
  resize();
  new ResizeObserver(resize).observe(cv.parentElement);
  return cv;
}
const X = (id, l) => PAD + (l - S.lamMin) / (S.lamMax - S.lamMin) * (cvs[id].w - 2 * PAD);
const LamOf = (id, x) => S.lamMin + (x - PAD) / (cvs[id].w - 2 * PAD) * (S.lamMax - S.lamMin);

function cursor(id, { r = false } = {}) {
  const { ctx, h } = cvs[id];
  const x = X(id, S.lam);
  ctx.strokeStyle = 'rgba(233,237,244,0.85)'; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, h - 2); ctx.stroke();
  if (r) {
    const xr = X(id, S.r);
    ctx.strokeStyle = 'rgba(31,169,163,0.9)'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(xr, 2); ctx.lineTo(xr, h - 2); ctx.stroke();
    ctx.setLineDash([]);
  }
}
function curve(id, f, { color, width = 1.6, dash = [], yMap }) {
  const { ctx, w } = cvs[id];
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
  ctx.beginPath();
  let started = false;
  for (let x = PAD; x <= w - PAD; x += 2) {
    const y = yMap(f(LamOf(id, x)));
    started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), started = true);
  }
  ctx.stroke(); ctx.setLineDash([]);
}

// ---------------- build static DOM (refs, legends, time rows) ----------------
export function initCards() {
  document.getElementById('hd-ref').innerHTML = refChip('vdm', 'tweedie');
  document.getElementById('time-refs').innerHTML = refChip('ddpm', 'iddpm', 'edm', 'rf', 'fm', 'sd3', 'scm');
  document.getElementById('train-refs').innerHTML = refChip('ddpm', 'edm', 'sd3', 'kingma23');
  document.getElementById('target-refs').innerHTML = refChip('tweedie', 'ddpm', 'vpred', 'fm', 'sde', 'edm');
  document.getElementById('sample-refs').innerHTML = refChip('ddim', 'ddpm', 'edm', 'sd3');
  document.getElementById('distill-refs').innerHTML = refChip('vpred', 'cm', 'ctm', 'scm', 'meanflow', 'shortcut', 'dmd', 'rf');
  document.getElementById('video-refs').innerHTML = refChip('df', 'causvid', 'selfforce');
  document.getElementById('all-refs').innerHTML = refList();

  // master
  reg('cv-master', 46);
  // time rows
  const rows = document.getElementById('time-rows');
  TIMEROWS.forEach((t, i) => {
    const div = document.createElement('div'); div.className = 'trow';
    div.innerHTML = `<div class="tname">${t.name}<span class="tsub">${t.cite} · ${t.desc}</span></div>
      <canvas id="trow-${i}"></canvas><div class="tread" id="tread-${i}"></div>`;
    rows.append(div);
  });
  TIMEROWS.forEach((_, i) => reg(`trow-${i}`, 26));
  // plots
  reg('cv-dens', 130); reg('cv-wts', 112); reg('cv-amp', 150);
  reg('cv-grid', 72); reg('cv-jump', 158); reg('cv-video', 128);
  // legends
  document.getElementById('leg-dens').innerHTML = DENS.map(d =>
    `<span><i style="background:${d.color}"></i>${d.label}</span>`).join('');
  document.getElementById('leg-wts').innerHTML = WTS.map(w =>
    `<span><i class="dash" style="color:${w.color}"></i>${w.label}</span>`).join('');
  document.getElementById('leg-amp').innerHTML = AMP.map(a =>
    `<span id="leg-amp-${a.key}"><i style="background:${a.color}"></i>${a.label}</span>`).join('');
  // hit: drag lambda on any registered canvas; r on jump card
  for (const id of ['cv-master', 'cv-dens', 'cv-wts', 'cv-amp', 'cv-grid', 'cv-jump', ...TIMEROWS.map((_, i) => `trow-${i}`)]) {
    hitDrag(id, id === 'cv-jump');
  }
  buildReadouts();
}

let dragging = null;
function hitDrag(id, allowR) {
  const cv = cvs[id].cv;
  cv.style.cursor = 'ew-resize';
  cv.addEventListener('mousedown', e => {
    const x = e.clientX - cv.getBoundingClientRect().left;
    dragging = { id, r: allowR && Math.abs(x - X(id, S.r)) < 9 && Math.abs(x - X(id, S.lam)) > 9 };
    if (S.playing) setS({ playing: false });
    move(e);
  });
  window.addEventListener('mousemove', move);
  window.addEventListener('mouseup', () => dragging = null);
  function move(e) {
    if (!dragging || dragging.id !== id) return;
    const rect = cv.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (cvs[id].w / rect.width);   // CSS box may be scaled
    const l = Math.max(S.lamMin, Math.min(S.lamMax, LamOf(id, x)));
    if (dragging.r) setS({ r: Math.min(l, S.lam - 0.3) });
    else setS({ lam: l });
  }
}

function buildReadouts() {
  document.getElementById('ax-readouts').innerHTML = `
    <span class="ro lamro">λ<b id="ro-lam"></b></span>
    <span class="ro">SNR<b id="ro-snr"></b></span>
    <span class="ro">α<b id="ro-a"></b></span>
    <span class="ro">σ<b id="ro-s"></b></span>
    <span class="ro">t<sub>RF</sub><b id="ro-t"></b></span>
    <span class="ro">σ<sub>EDM</sub><b id="ro-sig"></b></span>
    <span class="ro">φ<sub>Trig</sub><b id="ro-phi"></b></span>`;
}

// ---------------- draw everything ----------------
export function drawCards() {
  if (!cvs['cv-master']) return;
  drawMaster(); drawTimeRows(); drawDens(); drawWts(); drawAmp(); drawGrid(); drawJump();
  drawVideoCard();
}

function drawMaster() {
  const { ctx, w, h } = cvs['cv-master'];
  ctx.clearRect(0, 0, w, h);
  for (let x = PAD; x < w - PAD; x++) {
    ctx.fillStyle = lamColor(LamOf('cv-master', x));
    ctx.fillRect(x, 12, 1, 22);
  }
  ctx.strokeStyle = 'rgba(233,237,244,0.25)'; ctx.strokeRect(PAD, 12, w - 2 * PAD, 22);
  ctx.font = `10px ${MONO}`; ctx.fillStyle = '#5C6675'; ctx.textAlign = 'center';
  for (let l = -8; l <= 10; l += 2) {
    ctx.fillText(String(l), X('cv-master', l), 44);
    ctx.fillRect(X('cv-master', l) - 0.5, 34, 1, 3);
  }
  ctx.textAlign = 'left'; ctx.fillText('噪声端', PAD, 8);
  ctx.textAlign = 'right'; ctx.fillText('数据端', w - PAD, 8);
  const x = X('cv-master', S.lam);
  ctx.fillStyle = '#E9EDF4';
  ctx.beginPath(); ctx.moveTo(x, 12); ctx.lineTo(x - 6, 2); ctx.lineTo(x + 6, 2); ctx.closePath(); ctx.fill();
  ctx.fillRect(x - 1, 12, 2, 22);
  // readouts
  const a2 = 1 / (1 + Math.exp(-S.lam)), a = Math.sqrt(a2), s = Math.sqrt(1 - a2);
  const sig = Math.exp(-S.lam / 2);
  document.getElementById('ro-lam').textContent = S.lam.toFixed(2);
  document.getElementById('ro-snr').textContent = (4.3429 * S.lam).toFixed(1) + ' dB';
  document.getElementById('ro-a').textContent = a.toFixed(3);
  document.getElementById('ro-s').textContent = s.toFixed(3);
  document.getElementById('ro-t').textContent = tOfLam_rf(S.lam).toFixed(3);
  document.getElementById('ro-sig').textContent = sig < 100 ? sig.toPrecision(3) : sig.toFixed(0);
  document.getElementById('ro-phi').textContent = Math.atan(sig).toFixed(3);
}

function drawTimeRows() {
  TIMEROWS.forEach((t, i) => {
    const id = `trow-${i}`, { ctx, w, h } = cvs[id];
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(154,163,178,0.16)';
    ctx.beginPath(); ctx.moveTo(PAD, h / 2); ctx.lineTo(w - PAD, h / 2); ctx.stroke();
    ctx.strokeStyle = 'rgba(233,237,244,0.6)'; ctx.lineWidth = 1;
    for (const l of t.ticks()) {
      if (l < S.lamMin || l > S.lamMax) continue;
      const x = X(id, l);
      ctx.beginPath(); ctx.moveTo(x, h / 2 - 7); ctx.lineTo(x, h / 2 + 7); ctx.stroke();
    }
    cursor(id);
    document.getElementById(`tread-${i}`).textContent = t.fmt(S.lam);
  });
}

function drawDens() {
  const id = 'cv-dens', { ctx, w, h } = cvs[id];
  ctx.clearRect(0, 0, w, h);
  baseline(id);
  for (const d of DENS) {
    let mx = 0;
    for (let x = PAD; x <= w - PAD; x += 2) mx = Math.max(mx, d.f(LamOf(id, x)));
    ctx.beginPath(); ctx.moveTo(PAD, h - 6);
    for (let x = PAD; x <= w - PAD; x += 2) ctx.lineTo(x, h - 6 - d.f(LamOf(id, x)) / mx * (h - 18));
    ctx.lineTo(w - PAD, h - 6); ctx.closePath();
    ctx.fillStyle = d.color + '22'; ctx.fill();
    curve(id, l => d.f(l), { color: d.color, yMap: v => h - 6 - v / mx * (h - 18) });
  }
  cursor(id);
}
function drawWts() {
  const id = 'cv-wts', { ctx, w, h } = cvs[id];
  ctx.clearRect(0, 0, w, h);
  baseline(id);
  const lo = -1.5, hi = 4.5;
  for (const wt of WTS) {
    curve(id, l => Math.log10(wt.f(l)), {
      color: wt.color, dash: [6, 4], width: 1.5,
      yMap: v => h - 6 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * (h - 14),
    });
  }
  cursor(id);
}
function drawAmp() {
  const id = 'cv-amp', { ctx, w, h } = cvs[id];
  ctx.clearRect(0, 0, w, h);
  baseline(id);
  const lo = -2, hi = 2.2;
  const yOf = v => h - 8 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * (h - 18);
  ctx.strokeStyle = 'rgba(154,163,178,0.4)'; ctx.setLineDash([3, 5]);
  ctx.beginPath(); ctx.moveTo(PAD, yOf(0)); ctx.lineTo(w - PAD, yOf(0)); ctx.stroke(); ctx.setLineDash([]);
  for (const a of AMP) {
    const sel = a.key === S.head;
    ctx.globalAlpha = sel ? 1 : 0.4;
    curve(id, l => Math.log10(Math.max(1e-3, a.f(l))), { color: a.color, width: sel ? 2.6 : 1.2, yMap: yOf });
    ctx.globalAlpha = 1;
    const leg = document.getElementById(`leg-amp-${a.key}`);
    if (leg) leg.style.opacity = sel ? 1 : 0.5;
  }
  cursor(id);
}
function baseline(id) {
  const { ctx, w, h } = cvs[id];
  ctx.strokeStyle = 'rgba(154,163,178,0.14)';
  for (let l = -8; l <= 10; l += 2) {
    const x = X(id, l);
    ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, h - 2); ctx.stroke();
  }
}

function drawGrid() {
  const id = 'cv-grid', { ctx, w, h } = cvs[id];
  ctx.clearRect(0, 0, w, h);
  const y = h / 2 + 6;
  ctx.strokeStyle = 'rgba(154,163,178,0.3)';
  ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(w - PAD, y); ctx.stroke();
  const grid = gridLams(S.sched, S.N);
  grid.forEach((l, i) => {
    const x = X(id, l);
    ctx.fillStyle = i <= S.walkK && S.walkK > 0 ? '#F2B03D' : '#93A0B4';
    ctx.beginPath(); ctx.arc(x, y, i === 0 || i === grid.length - 1 ? 4.5 : 3, 0, 7); ctx.fill();
    if (S.eta > 0 && i < grid.length - 1) {   // eta: re-noise = a small hop back toward noise
      ctx.strokeStyle = 'rgba(242,176,61,0.6)'; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(x + 3, y - 9); ctx.lineTo(x - 5, y - 9); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x - 2, y - 12); ctx.lineTo(x - 5, y - 9); ctx.lineTo(x - 2, y - 6); ctx.stroke();
    }
  });
  const wl = grid[Math.min(S.walkK, grid.length - 1)];
  ctx.fillStyle = '#E9EDF4';
  ctx.beginPath(); ctx.arc(X(id, wl), y, 6, 0, 7); ctx.fill();
  ctx.fillStyle = '#0C0F15';
  ctx.beginPath(); ctx.arc(X(id, wl), y, 2.4, 0, 7); ctx.fill();
  cursor(id);
  document.getElementById('ax-nfe').textContent = `NFE=${S.N} · ${S.eta > 0 ? 'η=' + S.eta.toFixed(2) + '(向左小钩=重掺噪声)' : 'η=0 确定性(DDIM)'}`;
}

function jumpArc(id, l1, l2, y, bulge, { color, width = 2, dash = [], head = true }) {
  const { ctx } = cvs[id];
  const x1 = X(id, l1), x2 = X(id, l2);
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
  ctx.beginPath(); ctx.moveTo(x1, y);
  ctx.quadraticCurveTo((x1 + x2) / 2, y - bulge, x2, y);
  ctx.stroke(); ctx.setLineDash([]);
  if (head) {
    ctx.fillStyle = color;
    const dir = x2 >= x1 ? 1 : -1;
    ctx.beginPath(); ctx.moveTo(x2, y); ctx.lineTo(x2 - 7 * dir, y - 4); ctx.lineTo(x2 - 7 * dir, y + 4); ctx.closePath(); ctx.fill();
  }
}

function drawJump() {
  const id = 'cv-jump', { ctx, w, h } = cvs[id];
  ctx.clearRect(0, 0, w, h);
  baseline(id);
  const yT = 30, yS = 74, yP = 122;
  label(id, 'teacher · 多步积分', yT - 16);
  label(id, 'Consistency → 数据端 · CTM / MeanFlow → 区间 [r, λ]', yS - 16);
  label(id, 'Progressive Distillation · 两步并一步(反复)', yP - 16);
  // teacher small hops on current grid
  const grid = gridLams(S.sched, Math.max(12, S.N));
  for (let i = 0; i < grid.length - 1; i++) {
    jumpArc(id, grid[i], grid[i + 1], yT, 9, { color: '#D65077', width: 1.2, head: false });
  }
  // consistency: cursor -> data end
  jumpArc(id, S.lam, S.lamMax - 0.05, yS, 22, { color: '#1FA9A3', width: 2.2 });
  // interval flow map: r -> cursor
  jumpArc(id, S.r, S.lam, yS, 15, { color: '#B08FEA', width: 1.8, dash: [6, 4] });
  // PD: pair-merge glyph on an 8-step grid
  const g8 = gridLams(S.sched, 8);
  for (let i = 0; i < 8; i++) jumpArc(id, g8[i], g8[i + 1], yP, 7, { color: 'rgba(147,160,180,0.8)', width: 1.1, head: false });
  for (let i = 0; i < 8; i += 2) jumpArc(id, g8[i], g8[i + 2], yP, 14, { color: '#F2B03D', width: 1.4, head: false, dash: [4, 3] });
  cursor(id, { r: true });
  const span = (S.lam - S.r);
  document.getElementById('distill-note').innerHTML =
    `当前区间 [r=<b>${S.r.toFixed(2)}</b> → λ=<b>${S.lam.toFixed(2)}</b>](跨度 ${span.toFixed(2)} nat):紫色虚线弧即 CTM / MeanFlow 要学的映射 Φ,等于"平均速度 ū × 区间长"。` +
    `DMD 不在图上——它不学任何轨迹,只要求一步样本的<b>分布</b>与教师端点一致;ReFlow 也不在图上——它不改跳法,改路(探头 A 的"蒸馏轨迹"页:同种子下教师弯、ReFlow 直、一步学生是一根弦)。`;
}
function label(id, text, y) {
  const { ctx } = cvs[id];
  ctx.font = `10.5px ${MONO}`; ctx.fillStyle = '#7B8698'; ctx.textAlign = 'left';
  ctx.fillText(text, PAD, y);
}

// ---------------- video card ----------------
const KF = 8;
function frameLams() {
  const { vpreset: p, vphase: tau } = S;
  const lo = S.lamMin + 0.4, hi = S.lamMax - 0.3, out = [];
  if (p === 'offline') {
    const l = lo + (hi - lo) * tau;
    for (let k = 0; k < KF; k++) out.push(l);
  } else if (p === 'df') {
    for (let k = 0; k < KF; k++) out.push(Math.max(lo, Math.min(hi, lo + (hi - lo) * tau * 2.1 - k * 2.1)));
  } else {
    const slot = tau * KF, a = Math.min(KF - 1, Math.floor(slot)), f = slot - a;
    for (let k = 0; k < KF; k++) out.push(k < a ? hi : k === a ? lo + (hi - lo) * Math.min(1, f * 1.15) : lo);
  }
  return out;
}
export function drawVideoCard() {
  const id = 'cv-video';
  if (!cvs[id]) return;
  const { ctx, w, h } = cvs[id];
  ctx.clearRect(0, 0, w, h);
  // thin axis strip
  for (let x = PAD; x < w - PAD; x++) {
    ctx.fillStyle = lamColor(LamOf(id, x));
    ctx.globalAlpha = 0.5;
    ctx.fillRect(x, 60, 1, 8);
  }
  ctx.globalAlpha = 1;
  const lams = frameLams();
  const seen = [];
  lams.forEach((l, k) => {
    const x = X(id, l);
    const pile = seen.filter(v => Math.abs(v - x) < 15).length;
    seen.push(x);
    const r = 9, y = 52 - pile * 6;
    ctx.fillStyle = lamColor(l);
    ctx.strokeStyle = '#E9EDF4'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.rect(x - r, y - r, 2 * r, 2 * r); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#0C0F15'; ctx.font = `9.5px ${MONO}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(k), x, y);
  });
  ctx.textBaseline = 'alphabetic';
  // filmstrip in time order
  const cw = 26;
  ctx.font = `10px ${MONO}`; ctx.fillStyle = '#7B8698'; ctx.textAlign = 'left';
  ctx.fillText('时间顺序 →', PAD, 88);
  lams.forEach((l, k) => {
    const x = PAD + 78 + k * (cw + 5);
    ctx.fillStyle = lamColor(l);
    ctx.fillRect(x, 78, cw, 18);
    ctx.strokeStyle = 'rgba(233,237,244,0.3)'; ctx.strokeRect(x, 78, cw, 18);
  });
  document.getElementById('video-note').innerHTML = {
    offline: '<b>整段扩散</b>:8 帧绑成一叠,同一 λ 齐降(时空 patch 一起去噪)。质量最好;定长、高延迟、无交互接口。',
    df: '<b>Diffusion Forcing</b>:每帧独立 λ,越靠后越噪。模型同时学会"生成未来"与"把带噪历史当条件"——流式与抗误差累积的来源。',
    stream: '<b>流式因果</b>:历史帧已完成(λ→数据端,进 KV cache);当前帧用少量大步俯冲(帧内蒸馏);未来帧在噪声端排队。实时交互生成的形状。',
  }[S.vpreset];
}
