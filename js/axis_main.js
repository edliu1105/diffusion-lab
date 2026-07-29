// axis_main.js — v2. One state; every card and both probes sync to S.lam.

import { load, R } from './models.js';
import { initCards, drawCards, drawVideoCard } from './axis_cards.js';
import { initToy, drawToy, toyReset, toyStep } from './axis_toy.js';
import { initMnist, drawMnist } from './axis_mnist.js';

export const S = {
  lamMin: -9, lamMax: 10,
  lam: -1.0,
  head: 'v',
  model: 'u_fm',
  sched: 'rf', N: 16, eta: 0,
  r: -5.5,
  toyView: 'bridge',        // bridge | walk | distill
  playing: false, walkK: 0,
  vpreset: 'offline', vphase: 0,
};
let ready = false;
export function setS(patch) {
  Object.assign(S, patch);
  if (ready) drawAll();
}
export function drawAll() {
  drawCards(); drawToy(); drawMnist();
}

// ---------- time parameterizations: every "time" is a reparameterization of lambda ----------
export const lamOfT_rf = t => 2 * Math.log((1 - t) / t);
export const tOfLam_rf = l => 1 / (1 + Math.exp(l / 2));
export const lamOfSig = s => -2 * Math.log(s);
export const lamOfT_cos = t => {
  const s = 0.008, f0 = Math.cos(s / (1 + s) * Math.PI / 2);
  const a = Math.max(1e-6, Math.min(1 - 1e-9, Math.cos((t + s) / (1 + s) * Math.PI / 2) / f0));
  return 2 * Math.log(a / Math.sqrt(1 - a * a));
};
export const lamOfT_vplin = t => {
  const T = 1000, b0 = 1e-4, b1 = 0.02;
  const ab = Math.exp(-T * (b0 * t + (b1 - b0) * t * t / 2));
  return Math.log(ab / (1 - ab));
};
export const SHIFT_S = 3;
export const shiftLam = -2 * Math.log(SHIFT_S);   // SD3 resolution shift = rigid translation on lambda

function uniformTicks(lamOf, t0, t1, n = 15) {
  const out = [];
  for (let i = 0; i <= n; i++) out.push(lamOf(t0 + (t1 - t0) * i / n));
  return out;
}
function invMono(lamOf, target) {
  let lo = 1e-4, hi = 1 - 1e-4;
  const dec = lamOf(lo) > lamOf(hi);
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if ((lamOf(mid) > target) === dec) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}
export const TIMEROWS = [
  { name: 'VP linear-β 的 t', cite: 'Ho et al. 2020', desc: '离散 1000 步,β 线性',
    ticks: () => uniformTicks(lamOfT_vplin, 0.02, 0.999), fmt: l => 't = ' + invMono(lamOfT_vplin, l).toFixed(3) },
  { name: 'VP cosine 的 t', cite: 'Nichol & Dhariwal 2021', desc: 'ᾱ 余弦形',
    ticks: () => uniformTicks(lamOfT_cos, 0.001, 0.999), fmt: l => 't = ' + invMono(lamOfT_cos, l).toFixed(3) },
  { name: 'EDM 的 σ', cite: 'Karras et al. 2022', desc: 'σ = e^{−λ/2},ρ=7 网格两端加密',
    ticks: () => [80, 40, 20, 10, 5, 2, 1, .5, .2, .1, .05, .02, .01, .005, .002].map(lamOfSig),
    fmt: l => 'σ = ' + Math.exp(-l / 2).toPrecision(3) },
  { name: 'Rectified Flow 的 t', cite: 'Liu et al. 2022 / Lipman et al. 2022', desc: 'x=(1−t)x₀+t·ε',
    ticks: () => uniformTicks(lamOfT_rf, 0.002, 0.998), fmt: l => 't = ' + tOfLam_rf(l).toFixed(3) },
  { name: `RF + SD3 shift(s=${SHIFT_S})`, cite: 'Esser et al. 2024', desc: `λ 轴上整体平移 −2ln s = ${shiftLam.toFixed(2)}`,
    ticks: () => uniformTicks(t => lamOfT_rf(t) + shiftLam, 0.002, 0.998), fmt: l => "t′ = " + tOfLam_rf(l - shiftLam).toFixed(3) },
  { name: 'TrigFlow 的 φ', cite: 'Lu & Song 2024(sCM)', desc: 'x = cosφ·x₀ + sinφ·ε,φ = arctan e^{−λ/2}',
    ticks: () => { const a = []; for (let i = 1; i < 16; i++) a.push(-2 * Math.log(Math.tan(i / 16 * Math.PI / 2))); return a; },
    fmt: l => 'φ = ' + Math.atan(Math.exp(-l / 2)).toFixed(3) },
];

export function gridLams(sched, N) {
  const out = [];
  if (sched === 'rf' || sched === 'rf_shift') {
    for (let i = 0; i <= N; i++) {
      const t = 0.985 + (0.002 - 0.985) * i / N;
      out.push(lamOfT_rf(t) + (sched === 'rf_shift' ? shiftLam : 0));
    }
  } else if (sched === 'edm') {
    const rho = 7, smax = 80, smin = 0.002;
    for (let i = 0; i <= N; i++) {
      const f = i / N;
      out.push(lamOfSig((smax ** (1 / rho) + f * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho));
    }
  } else if (sched === 'cosine') {
    for (let i = 0; i <= N; i++) out.push(lamOfT_cos(0.999 - (0.999 - 0.001) * i / N));
  } else {
    for (let i = 0; i <= N; i++) out.push(-8.5 + (9 - (-8.5)) * i / N);
  }
  return out.map(l => Math.max(S.lamMin, Math.min(S.lamMax, l)));
}

// ---------- training densities & unified loss weights (analytic, canonical params) ----------
const nrm = (x, m, sd) => Math.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
export const DENS = [
  { key: 'cos', label: '均匀 t · cosine(Nichol 2021)', color: '#5083DC',
    f: l => { const d = 0.01, tOf = L => { const a2 = 1 / (1 + Math.exp(-L)); const s = 0.008, f0 = Math.cos(s / (1 + s) * Math.PI / 2); return Math.max(0, Math.min(1, (2 / Math.PI) * Math.acos(Math.min(1, Math.sqrt(a2) * f0)) * (1 + s) - s)); }; return Math.abs((tOf(l + d) - tOf(l - d)) / (2 * d)); } },
  { key: 'edm', label: 'lognormal σ(Karras 2022)→ λ~N(2.4, 2.4²)', color: '#C28324', f: l => nrm(l, 2.4, 2.4) },
  { key: 'sd3', label: 'logit-normal t(Esser 2024)→ λ~N(0, 2²)', color: '#D65077', f: l => nrm(l, 0, 2) },
  { key: 'sd3s', label: `+ shift s=${SHIFT_S}(同文)→ 平移 ${shiftLam.toFixed(2)}`, color: '#1FA9A3', f: l => nrm(l, shiftLam, 2) },
];
const SD_DATA = 1.083;
export const WTS = [
  { key: 'eps', label: 'ε 损失 ⇒ w=e^λ(Ho 2020)', color: '#5083DC', f: l => Math.exp(l) },
  { key: 'v', label: 'v 损失 ⇒ w=1+e^λ(Salimans & Ho 2022)', color: '#B08FEA', f: l => 1 + Math.exp(l) },
  { key: 'u', label: 'u 损失 ⇒ w=(1+e^{λ/2})²(Lipman 2022)', color: '#D65077', f: l => (1 + Math.exp(l / 2)) ** 2 },
  { key: 'D', label: 'EDM λ(σ) 权重(Karras 2022)', color: '#C28324', f: l => (Math.exp(-l) + SD_DATA ** 2) / (Math.exp(-l) * SD_DATA ** 2) },
];
// error amplification: one unit of target error -> x0_hat error (VP frame)
export const AMP = [
  { key: 'eps', label: 'ε̂:×e^{−λ/2}', color: '#5083DC', f: l => Math.exp(-l / 2) },
  { key: 'x0', label: 'x̂₀:×1', color: '#9AA3B2', f: () => 1 },
  { key: 'v', label: 'v̂:×σ(有界)', color: '#B08FEA', f: l => Math.sqrt(1 / (1 + Math.exp(l))) },
  { key: 'u', label: 'û:×t(有界)', color: '#D65077', f: l => 1 / (1 + Math.exp(l / 2)) },
  { key: 'score', label: 'score:×σ²/α', color: '#E2B33C', f: l => (1 / (1 + Math.exp(l))) / Math.sqrt(1 / (1 + Math.exp(-l))) },
  { key: 'D', label: 'EDM-D:×c_out(有界)', color: '#C28324', f: l => { const s = Math.exp(-l / 2); return s * SD_DATA / Math.sqrt(s * s + SD_DATA * SD_DATA); } },
];

// ---------- boot ----------
const el = id => document.getElementById(id);
async function boot() {
  await load();
  initCards();
  initToy(el('ax-toy'));
  await initMnist(el('ax-mnist'), el('mn-foot'));
  wire();
  toyReset();
  const q = new URLSearchParams(location.search);
  if (q.get('head')) { const b = document.querySelector(`#ax-heads [data-h="${q.get('head')}"]`); if (b) b.click(); }
  if (q.get('view')) { const b = document.querySelector(`#toy-tabs [data-v="${q.get('view')}"]`); if (b) b.click(); }
  if (q.get('sched')) { el('ax-sched').value = q.get('sched'); S.sched = q.get('sched'); }
  if (q.get('N')) { el('ax-n').value = q.get('N'); S.N = +q.get('N'); el('ax-nv').textContent = q.get('N'); }
  if (q.get('lam')) S.lam = Math.max(S.lamMin, Math.min(S.lamMax, +q.get('lam')));
  if (q.get('r')) S.r = +q.get('r');
  ready = true;
  drawAll();
  // video phase animation (visible-tab only; card redraw is cheap)
  let last = performance.now();
  (function frame(now) {
    const dt = now - last; last = now;
    S.vphase = (S.vphase + dt / 9000) % 1;
    if (ready) drawVideoCard();
    requestAnimationFrame(frame);
  })(performance.now());
  // sampler playback (interval: hidden-tab safe)
  setInterval(() => {
    if (!S.playing) return;
    const grid = gridLams(S.sched, S.N);
    if (S.walkK < grid.length - 1) {
      toyStep(grid[S.walkK], grid[S.walkK + 1], S.eta);
      S.walkK++; setS({ lam: grid[S.walkK] });
    } else { S.playing = false; el('ax-play').textContent = '▶ 在探头 A 里走一遍'; setS({}); }
  }, 130);
}

function resetWalk() {
  S.playing = false; S.walkK = 0;
  el('ax-play').textContent = '▶ 在探头 A 里走一遍';
  toyReset();
}

function wire() {
  el('ax-heads').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    el('ax-heads').querySelectorAll('button').forEach(o => o.classList.remove('on'));
    b.classList.add('on'); setS({ head: b.dataset.h });
  }));
  el('ax-model').addEventListener('change', e => { setS({ model: e.target.value }); resetWalk(); setS({}); });
  el('ax-sched').addEventListener('change', e => { S.sched = e.target.value; resetWalk(); setS({}); });
  el('ax-n').addEventListener('input', e => { S.N = +e.target.value; el('ax-nv').textContent = e.target.value; resetWalk(); setS({}); });
  el('ax-eta').addEventListener('input', e => { S.eta = +e.target.value; el('ax-etav').textContent = (+e.target.value).toFixed(2); setS({}); });
  el('ax-play').addEventListener('click', togglePlay);
  el('toy-tabs').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    el('toy-tabs').querySelectorAll('button').forEach(o => o.classList.remove('on'));
    b.classList.add('on');
    if (b.dataset.v !== 'walk') { S.playing = false; }
    setS({ toyView: b.dataset.v });
  }));
  el('ax-vpreset').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    el('ax-vpreset').querySelectorAll('button').forEach(o => o.classList.remove('on'));
    b.classList.add('on'); S.vphase = 0; setS({ vpreset: b.dataset.v });
  }));
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') { setS({ lam: Math.max(S.lamMin, S.lam - (e.shiftKey ? 1 : 0.15)) }); e.preventDefault(); }
    if (e.key === 'ArrowRight') { setS({ lam: Math.min(S.lamMax, S.lam + (e.shiftKey ? 1 : 0.15)) }); e.preventDefault(); }
    if (e.key === ' ') { togglePlay(); e.preventDefault(); }
  });
}
function togglePlay() {
  if (S.playing) { S.playing = false; el('ax-play').textContent = '▶ 在探头 A 里走一遍'; setS({}); return; }
  const grid = gridLams(S.sched, S.N);
  if (S.walkK >= grid.length - 1) { S.walkK = 0; toyReset(); }
  S.lam = grid[S.walkK];
  S.playing = true;
  el('ax-play').textContent = '⏸ 暂停';
  document.querySelectorAll('#toy-tabs button').forEach(o => o.classList.toggle('on', o.dataset.v === 'walk'));
  setS({ toyView: 'walk' });
}

boot();
