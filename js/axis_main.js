// axis_main.js — one instrument, one state. Everything syncs to S.lam.

import { load, R } from './models.js';
import { initLanes, drawLanes, lanesHit } from './axis_lanes.js';
import { initToy, drawToy, toyReset, toyStep } from './axis_toy.js';
import { initMnist, drawMnist } from './axis_mnist.js';

export const S = {
  lamMin: -9, lamMax: 10,
  lam: -1.0,
  mode: 'train',            // train | infer | distill | video
  head: 'v',                // x0 | eps | v | u | score | D
  model: 'u_fm',
  sched: 'rf', N: 16, eta: 0,
  r: -5.5,                  // second cursor (flow-map interval start), distill mode
  playing: false, walkK: 0,
  vpreset: 'offline', vphase: 0,
  dirty: true,
};
export const bus = { onchange: [] };
let ready = false;
export function setS(patch) {
  Object.assign(S, patch); S.dirty = true;
  for (const f of bus.onchange) f(patch);
  if (ready) drawAll();   // synchronous: works even in hidden tabs (no rAF dependency)
}
export function drawAll() {
  drawLanes(S); drawToy(S); drawMnist(S);
}

// ---------- schedules: every "time" is a monotone reparam of lambda ----------
export const lamOfT_rf = t => 2 * Math.log((1 - t) / t);
export const tOfLam_rf = l => 1 / (1 + Math.exp(l / 2));
export const lamOfSig = s => -2 * Math.log(s);
export const lamOfT_cos = t => {
  const s = 0.008, f0 = Math.cos(s / (1 + s) * Math.PI / 2);
  const a = Math.max(1e-6, Math.min(1 - 1e-9, Math.cos((t + s) / (1 + s) * Math.PI / 2) / f0));
  return 2 * Math.log(a / Math.sqrt(1 - a * a));
};
export const lamOfT_vplin = t => { // continuous linear-beta, T=1000, 1e-4..0.02
  const T = 1000, b0 = 1e-4, b1 = 0.02;
  const integ = T * (b0 * t + (b1 - b0) * t * t / 2);
  const ab = Math.exp(-integ);
  return Math.log(ab / (1 - ab));
};
export const SHIFT_S = 3;
export const shiftLam = -2 * Math.log(SHIFT_S);   // SD3 shift on the lambda axis = rigid translation

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
      const s = (smax ** (1 / rho) + f * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho;
      out.push(lamOfSig(s));
    }
  } else if (sched === 'cosine') {
    for (let i = 0; i <= N; i++) out.push(lamOfT_cos(0.999 - (0.999 - 0.001) * i / N));
  } else { // logsnr uniform
    for (let i = 0; i <= N; i++) out.push(-8.5 + (9 - (-8.5)) * i / N);
  }
  return out.map(l => Math.max(S.lamMin, Math.min(S.lamMax, l)));
}

// ---------- densities & weights on the axis (analytic; canonical params) ----------
const nrm = (x, m, sd) => Math.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
export const DENS = [
  { key: 'cos', label: 'uniform-t·cosine', color: '#5083DC',
    f: l => { const d = 0.01, tOf = L => { const a2 = 1 / (1 + Math.exp(-L)); const s = 0.008, f0 = Math.cos(s / (1 + s) * Math.PI / 2); return Math.max(0, Math.min(1, (2 / Math.PI) * Math.acos(Math.min(1, Math.sqrt(a2) * f0)) * (1 + s) - s)); }; return Math.abs((tOf(l + d) - tOf(l - d)) / (2 * d)); } },
  { key: 'edm', label: 'EDM logN(−1.2,1.2)', color: '#C28324', f: l => nrm(l, 2.4, 2.4) },
  { key: 'sd3', label: 'SD3 logitN(0,1)', color: '#D65077', f: l => nrm(l, 0, 2) },
  { key: 'sd3s', label: `SD3 + shift ${SHIFT_S} (平移 ${shiftLam.toFixed(2)})`, color: '#1FA9A3', f: l => nrm(l, shiftLam, 2) },
];
const SD_DATA = 1.083;
export const WTS = [
  { key: 'eps', label: 'w_ε=e^λ', color: '#5083DC', f: l => Math.exp(l) },
  { key: 'v', label: 'w_v=1+e^λ', color: '#B08FEA', f: l => 1 + Math.exp(l) },
  { key: 'u', label: 'w_u=(1+e^{λ/2})²', color: '#D65077', f: l => (1 + Math.exp(l / 2)) ** 2 },
  { key: 'D', label: 'w_EDM', color: '#C28324', f: l => (Math.exp(-l) + SD_DATA ** 2) / (Math.exp(-l) * SD_DATA ** 2) },
];
// error amplification: head error -> x0_hat error (VP frame), per lambda
export const AMP = [
  { key: 'eps', label: 'ε̂', color: '#5083DC', f: l => Math.exp(-l / 2) },
  { key: 'x0', label: 'x̂₀', color: '#9AA3B2', f: () => 1 },
  { key: 'v', label: 'v̂', color: '#B08FEA', f: l => Math.sqrt(1 / (1 + Math.exp(l))) },
  { key: 'u', label: 'û', color: '#D65077', f: l => 1 / (1 + Math.exp(l / 2)) },
  { key: 'score', label: 'score', color: '#E2B33C', f: l => (1 / (1 + Math.exp(l))) / Math.sqrt(1 / (1 + Math.exp(-l))) },
  { key: 'D', label: 'EDM-D', color: '#C28324', f: l => { const s = Math.exp(-l / 2); return s * SD_DATA / Math.sqrt(s * s + SD_DATA * SD_DATA); } },
];

// clock rows: each convention's own uniform ticks, mapped onto lambda
export const CLOCKS = [
  { label: 'VP-linear t', ticks: () => rangeTicks(t => lamOfT_vplin(t), 0.02, 0.999), fmt: l => 't=' + tickInv(t => lamOfT_vplin(t), l).toFixed(3) },
  { label: 'VP-cosine t', ticks: () => rangeTicks(t => lamOfT_cos(t), 0.001, 0.999), fmt: l => 't=' + tickInv(t => lamOfT_cos(t), l).toFixed(3) },
  { label: 'EDM σ', ticks: () => [80, 40, 20, 10, 5, 2, 1, .5, .2, .1, .05, .02, .01, .005, .002].map(lamOfSig), fmt: l => 'σ=' + Math.exp(-l / 2).toPrecision(3) },
  { label: 'RF t', ticks: () => rangeTicks(t => lamOfT_rf(t), 0.002, 0.998), fmt: l => 't=' + tOfLam_rf(l).toFixed(3) },
  { label: `RF+shift${SHIFT_S}`, ticks: () => rangeTicks(t => lamOfT_rf(t) + shiftLam, 0.002, 0.998), fmt: l => "t'=" + tOfLam_rf(l - shiftLam).toFixed(3) },
  { label: 'TrigFlow φ', ticks: () => { const a = []; for (let i = 1; i < 16; i++) a.push(-2 * Math.log(Math.tan(i / 16 * Math.PI / 2))); return a; }, fmt: l => 'φ=' + Math.atan(Math.exp(-l / 2)).toFixed(3) },
];
function rangeTicks(lamOf, t0, t1, n = 15) {
  const out = [];
  for (let i = 0; i <= n; i++) out.push(lamOf(t0 + (t1 - t0) * i / n));
  return out;
}
function tickInv(lamOf, target) { // numeric inverse on t in (0,1)
  let lo = 1e-4, hi = 1 - 1e-4;
  const dec = lamOf(lo) > lamOf(hi);
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if ((lamOf(mid) > target) === dec) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

// ---------- boot ----------
const el = id => document.getElementById(id);
async function boot() {
  await load();
  const axisCv = el('ax-axis'), toyCv = el('ax-toy');
  initLanes(axisCv);
  initToy(toyCv);
  await initMnist(el('ax-mnist'), el('mn-foot'));
  wire();
  toyReset();
  // shareable state via URL (?mode=distill&lam=2&head=u&sched=edm&N=8&r=-5)
  const q = new URLSearchParams(location.search);
  if (q.get('mode')) {
    const b = document.querySelector(`#ax-modes [data-m="${q.get('mode')}"]`);
    if (b) b.click();
  }
  if (q.get('head')) {
    const b = document.querySelector(`#ax-heads [data-h="${q.get('head')}"]`);
    if (b) b.click();
  }
  if (q.get('sched')) { el('ax-sched').value = q.get('sched'); S.sched = q.get('sched'); }
  if (q.get('N')) { el('ax-n').value = q.get('N'); S.N = +q.get('N'); el('ax-nv').textContent = q.get('N'); }
  if (q.get('lam')) S.lam = Math.max(S.lamMin, Math.min(S.lamMax, +q.get('lam')));
  if (q.get('r')) S.r = +q.get('r');
  ready = true;
  drawAll();
  let last = performance.now();
  function frame(now) {
    const dt = now - last; last = now;
    if (S.mode === 'video') { S.vphase = (S.vphase + dt / 9000) % 1; drawAll(); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  setInterval(() => {   // play stepping: interval, not rAF (hidden-tab safe)
    if (!S.playing) return;
    const grid = gridLams(S.sched, S.N);
    if (S.walkK < grid.length - 1) {
      toyStep(grid[S.walkK], grid[S.walkK + 1], S.eta);
      S.walkK++; setS({ lam: grid[S.walkK] });
    } else { S.playing = false; el('ax-play').textContent = '▶ 行走'; setS({}); }
  }, 130);
}

function resetWalk() {
  S.playing = false; S.walkK = 0; el('ax-play').textContent = '▶ 行走';
  toyReset(); S.dirty = true;
}

function wire() {
  // mode tabs
  el('ax-modes').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    el('ax-modes').querySelectorAll('button').forEach(o => o.classList.remove('on'));
    b.classList.add('on');
    setS({ mode: b.dataset.m });
    el('g-video').style.display = S.mode === 'video' ? '' : 'none';
    ['g-sched', 'g-n', 'g-eta'].forEach(id => el(id).style.opacity = S.mode === 'video' ? 0.35 : 1);
    el('ax-play').style.display = S.mode === 'video' ? 'none' : '';
    resetWalk();
  }));
  // heads
  el('ax-heads').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    el('ax-heads').querySelectorAll('button').forEach(o => o.classList.remove('on'));
    b.classList.add('on'); setS({ head: b.dataset.h });
  }));
  el('ax-model').addEventListener('change', e => { setS({ model: e.target.value }); resetWalk(); });
  el('ax-sched').addEventListener('change', e => { setS({ sched: e.target.value }); resetWalk(); });
  el('ax-n').addEventListener('input', e => { setS({ N: +e.target.value }); el('ax-nv').textContent = e.target.value; resetWalk(); });
  el('ax-eta').addEventListener('input', e => { setS({ eta: +e.target.value }); el('ax-etav').textContent = (+e.target.value).toFixed(2); });
  el('ax-play').addEventListener('click', togglePlay);
  el('ax-vpreset').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    el('ax-vpreset').querySelectorAll('button').forEach(o => o.classList.remove('on'));
    b.classList.add('on'); setS({ vpreset: b.dataset.v, vphase: 0 });
  }));
  // keyboard
  window.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') { setS({ lam: Math.max(S.lamMin, S.lam - (e.shiftKey ? 1 : 0.15)) }); e.preventDefault(); }
    if (e.key === 'ArrowRight') { setS({ lam: Math.min(S.lamMax, S.lam + (e.shiftKey ? 1 : 0.15)) }); e.preventDefault(); }
    if (e.key === ' ') { togglePlay(); e.preventDefault(); }
  });
  lanesHit((patch) => setS(patch), resetWalk);
}
function togglePlay() {
  if (S.mode === 'video') return;
  if (S.playing) { S.playing = false; el('ax-play').textContent = '▶ 行走'; setS({}); return; }
  const grid = gridLams(S.sched, S.N);
  if (S.walkK >= grid.length - 1) { S.walkK = 0; toyReset(); }
  S.lam = grid[S.walkK];
  S.playing = true; el('ax-play').textContent = '⏸ 暂停'; setS({});
}

boot();
