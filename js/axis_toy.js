// axis_toy.js — probe A: frozen (x0, eps) pairs surf the bridge as lambda moves;
// samplers walk the same axis; distillation shows curved vs straightened vs one-chord.

import { R } from './models.js';
import { Plane, MODE_COLORS } from './viz2d.js';
import { lamColor } from './charts.js';
import { mulberry32, gauss } from './gmm.js';
import { S } from './axis_main.js';
import { vpFromLam, fmTFromLam } from './nn.js';

let plane, foot;
let NP = 620;
let x0s, epss;                 // frozen pairs
let xsWalk = null;             // sampler state (VP frame)
let trails = [];               // per tracked particle: [x,y,...]
let rngWalk;
let distillCache = null;       // {key, teacher:[], reflow:[], chords:[], ends:[]}
let distillResults = null;

const TRACK = 34;

export function initToy(canvas) {
  foot = document.getElementById('toy-foot');
  const sizeOf = () => {
    const pw = canvas.parentElement.clientWidth;
    return pw >= 360 ? Math.min(480, pw - 52) : 440;   // hidden-pane fallback
  };
  plane = new Plane(canvas, { w: sizeOf(), h: sizeOf(), range: 2.6 });
  new ResizeObserver(() => {
    const s = sizeOf();
    if (Math.abs(plane.w - s) < 6) return;
    plane = new Plane(canvas, { w: s, h: s, range: 2.6 });
    import('./axis_main.js').then(m => m.setS({}));
  }).observe(canvas.parentElement);
  const rng = mulberry32(20260728);
  x0s = new Float32Array(NP * 2); epss = new Float32Array(NP * 2);
  const { x } = R.gmm.sample(NP, rng);
  x0s.set(x);
  for (let i = 0; i < NP * 2; i++) epss[i] = gauss(rng);
  fetch('data/distill2d_results.json').then(r => r.json()).then(j => { distillResults = j; S.dirty = true; }).catch(() => { });
}

export function toyReset() {
  xsWalk = null; trails = []; distillCache = null;
}

function denoise(xs, lam, n) {
  const lams = new Float32Array(n).fill(lam);
  if (S.model === 'analytic') return R.gmm.denoiseVP(xs, lams, n);
  return R.models[S.model].denoiseVP(xs, lams, n);
}

// one generalized-DDIM step on the chosen lambda grid (VP frame), eta-stochastic
export function toyStep(lamA, lamB, eta) {
  if (!xsWalk) initWalk(lamA);
  const n = NP;
  const { a, s } = vpFromLam(lamA), { a: a2, s: s2 } = vpFromLam(lamB);
  const x0h = denoise(xsWalk, lamA, n);
  const tauF = eta * (s2 / s) * Math.sqrt(Math.max(0, 1 - (a / a2) ** 2));
  const tau = Math.min(tauF, s2);
  const sig = Math.sqrt(Math.max(0, s2 * s2 - tau * tau));
  for (let i = 0; i < n * 2; i++) {
    const epsh = (xsWalk[i] - a * x0h[i]) / s;
    xsWalk[i] = a2 * x0h[i] + sig * epsh + (tau > 0 ? tau * gauss(rngWalk) : 0);
  }
  const scB = scOf(lamB);
  for (let k = 0; k < TRACK; k++) trails[k].push(xsWalk[k * 2] / scB, xsWalk[k * 2 + 1] / scB);
}
function initWalk(lam0) {
  const { a, s } = vpFromLam(lam0);
  xsWalk = new Float32Array(NP * 2);
  for (let i = 0; i < NP * 2; i++) xsWalk[i] = a * x0s[i] + s * epss[i];
  rngWalk = mulberry32(777);
  trails = [];
  const sc0 = scOf(lam0);
  for (let k = 0; k < TRACK; k++) trails.push([xsWalk[k * 2] / sc0, xsWalk[k * 2 + 1] / sc0]);
}

// heads at a grid point, from x0_hat (VP frame)
function headVec(hx, hy, x, y, lam) {
  const { a, s } = vpFromLam(lam);
  const ex = (x - a * hx) / s, ey = (y - a * hy) / s;
  switch (S.head) {
    case 'x0': case 'D': return [hx - x, hy - y, 1];
    case 'eps': return [ex, ey, 0.32];
    case 'v': return [a * ex - s * hx, a * ey - s * hy, 0.32];
    case 'u': return [ex - hx, ey - hy, 0.3];
    case 'score': return [-ex / s, -ey / s, 0.1];
  }
}

// display frame = FM (linear-path native coords: straight stays straight)
const scOf = lam => { const { a } = vpFromLam(lam); return a / Math.max(1 - fmTFromLam(lam), 1e-9); };

export function drawToy() {
  if (!plane) return;
  plane.clear();
  const lam = S.lam, { a, s } = vpFromLam(lam), t = fmTFromLam(lam);
  // data anchors
  for (let m = 0; m < R.gmm.K; m++) plane.cross(R.gmm.mu[m][0], R.gmm.mu[m][1], { color: MODE_COLORS[m], size: 5 });

  if (S.toyView === 'distill') { drawDistill(); return; }

  // bridge cloud (frozen pairs at current lambda), FM coords
  const ghost = S.toyView === 'walk' && xsWalk;
  const bri = new Float32Array(NP * 2);
  for (let i = 0; i < NP * 2; i++) bri[i] = (1 - t) * x0s[i] + t * epss[i];
  plane.scatter(bri, { color: lamColor(lam), r: ghost ? 1.3 : 2.0, alpha: ghost ? 0.22 : 0.8 });

  if (S.toyView === 'bridge') {
    drawField(lam);
    drawTrainingPairs(lam);
    foot.innerHTML = `训练所见的混合分布:x_τ=(1−τ)x₀+τ·ε,当前 λ=<b>${lam.toFixed(2)}</b>(τ=<b>${t.toFixed(3)}</b>,α=${a.toFixed(3)} σ=${s.toFixed(3)})。` +
      `金色箭头 = 5 个训练样本对此刻的回归目标(目标=<b>${S.head}</b>,卡③选择);灰箭头 = 网络 <b>${S.model}</b> 的实际输出场。`;
  } else {
    if (xsWalk) {
      const sc = scOf(lam);
      const disp = new Float32Array(NP * 2);
      for (let i = 0; i < NP * 2; i++) disp[i] = xsWalk[i] / sc;
      for (const tr of trails) plane.path(new Float32Array(tr), { color: 'rgba(232,228,218,0.5)', width: 0.8, alpha: 0.5 });
      plane.scatter(disp, { color: lamColor(lam), r: 2.2, alpha: 0.95 });
      foot.innerHTML = `采样器 = 广义 DDIM(η=${S.eta.toFixed(2)}),调度 <b>${S.sched}</b>,步 <b>${S.walkK}</b>/${S.N}。` +
        `亮点 = 真实采样轨迹;暗点 = 训练混合(桥)。两团云的<b>形状</b>在每个 λ 都一致,但单个粒子早已离开它的桥搭档——分布相同,路径不同。`;
    } else {
      foot.innerHTML = `按卡④的「▶」:620 个粒子从噪声端出发,沿所选步点走真实采样器。`;
    }
  }
}

function drawField(lam) {
  const G = 12, span = 2.3, n = G * G;
  const fm = new Float32Array(n * 2), vp = new Float32Array(n * 2), lams = new Float32Array(n).fill(lam);
  const sc0 = scOf(lam);
  let k = 0;
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    fm[k * 2] = -span + 2 * span * i / (G - 1);
    fm[k * 2 + 1] = -span + 2 * span * j / (G - 1);
    vp[k * 2] = fm[k * 2] * sc0; vp[k * 2 + 1] = fm[k * 2 + 1] * sc0;
    k++;
  }
  const x0h = S.model === 'analytic' ? R.gmm.denoiseVP(vp, lams, n) : R.models[S.model].denoiseVP(vp, lams, n);
  for (let i = 0; i < n; i++) {
    let vx, vy;
    if (S.head === 'x0' || S.head === 'D') {           // mapping arrow, in display coords
      vx = x0h[i * 2] - fm[i * 2]; vy = x0h[i * 2 + 1] - fm[i * 2 + 1];
    } else {
      const [dx, dy, hsc] = headVec(x0h[i * 2], x0h[i * 2 + 1], vp[i * 2], vp[i * 2 + 1], lam);
      vx = dx * hsc; vy = dy * hsc;
    }
    const L = Math.hypot(vx, vy), mx = 0.5;
    if (L > mx) { vx *= mx / L; vy *= mx / L; }
    plane.arrow(fm[i * 2], fm[i * 2 + 1], vx, vy, { color: 'rgba(154,163,178,0.55)', width: 1, head: 3 });
  }
}

function drawTrainingPairs(lam) {
  const { a, s } = vpFromLam(lam), t = fmTFromLam(lam);
  for (let k = 0; k < 5; k++) {
    const i = 40 + k * 97;
    const xt = [(1 - t) * x0s[i * 2] + t * epss[i * 2], (1 - t) * x0s[i * 2 + 1] + t * epss[i * 2 + 1]];
    // exact target for the selected head at this pair
    let tx, ty, sc;
    const ex = epss[i * 2], ey = epss[i * 2 + 1], ox = x0s[i * 2], oy = x0s[i * 2 + 1];
    switch (S.head) {
      case 'x0': case 'D': tx = ox - xt[0]; ty = oy - xt[1]; sc = 1; break;
      case 'eps': tx = ex; ty = ey; sc = 0.32; break;
      case 'v': tx = a * ex - s * ox; ty = a * ey - s * oy; sc = 0.32; break;
      case 'u': tx = ex - ox; ty = ey - oy; sc = 0.3; break;
      case 'score': tx = -ex / s; ty = -ey / s; sc = 0.1; break;
    }
    plane.ring(xt[0], xt[1], 0.07, { color: '#F2B03D', dash: [] });
    plane.arrow(xt[0], xt[1], tx * sc, ty * sc, { color: '#F2B03D', width: 1.8, head: 5 });
  }
}

// ---------- distillation view: three real bundles ----------
function buildDistill() {
  const key = 'd';
  if (distillCache && distillCache.key === key) return distillCache;
  const n = 14, steps = 40, tS = 0.985, tE = 0.002;
  const rng = mulberry32(4242);
  const x1 = new Float32Array(n * 2);
  let got = 0;                                  // curated seeds: moderate radius, readable plot
  while (got < n) {
    const gx = gauss(rng), gy = gauss(rng);
    const r = Math.hypot(gx, gy);
    if (r > 1.9 || r < 0.35) continue;
    x1[got * 2] = gx; x1[got * 2 + 1] = gy; got++;
  }
  const run = (m) => {
    const xs = Float32Array.from(x1);
    const poly = Array.from({ length: n }, () => []);
    const lamAt = t => 2 * Math.log((1 - t) / t);
    for (let k = 0; k <= steps; k++) {
      const t = tS + (tE - tS) * k / steps;
      const lam = lamAt(Math.max(t, 1e-6));
      for (let i = 0; i < n; i++) poly[i].push(xs[i * 2], xs[i * 2 + 1]);   // FM coords: straight stays straight
      if (k === steps) break;
      const t2 = tS + (tE - tS) * (k + 1) / steps;
      const lams = new Float32Array(n).fill(lam);
      const out = m.forward(xs, lams, n);       // native u head
      for (let i = 0; i < n * 2; i++) xs[i] += (t2 - t) * out[i];
    }
    return poly;
  };
  const teacher = run(R.models.u_fm);
  const reflow = run(R.models.u_reflow);
  // one-step student: mean-velocity chord (FM coords, literally a chord)
  const lam1 = 2 * Math.log((1 - tS) / tS);
  const lams1 = new Float32Array(n).fill(lam1);
  const uu = R.models.u_1step.forward(x1, lams1, n);
  const chords = [];
  for (let i = 0; i < n; i++) {
    chords.push([x1[i * 2], x1[i * 2 + 1],
      x1[i * 2] - (tS - tE) * uu[i * 2], x1[i * 2 + 1] - (tS - tE) * uu[i * 2 + 1]]);
  }
  distillCache = { key, teacher, reflow, chords, tS, tE, steps };
  return distillCache;
}

function drawDistill() {
  const D = buildDistill();
  for (const c of D.chords) plane.path(new Float32Array(c), { color: 'rgba(232,228,218,0.55)', width: 0.8, alpha: 0.4, dash: [5, 5] });
  for (const p of D.teacher) plane.path(new Float32Array(p), { color: '#D65077', width: 1.3, alpha: 0.5 });
  for (const p of D.reflow) plane.path(new Float32Array(p), { color: '#1FA9A3', width: 1.5, alpha: 0.8 });
  for (const p of D.teacher) {
    const m = p.length;
    plane.cross(p[m - 2], p[m - 1], { color: '#D65077', size: 3, width: 1 });
  }
  // interval [r, lam] highlighted on teacher paths: the piece a flow map swallows
  const tOf = l => 1 / (1 + Math.exp(l / 2));
  const kOf = l => Math.round((tOf(l) - D.tS) / (D.tE - D.tS) * D.steps);
  const k1 = Math.max(0, Math.min(D.steps, kOf(S.r))), k2 = Math.max(0, Math.min(D.steps, kOf(S.lam)));
  const [ka, kb] = [Math.min(k1, k2), Math.max(k1, k2)];
  if (kb > ka) {
    for (const p of D.teacher.slice(0, 8)) {
      plane.path(new Float32Array(p.slice(ka * 2, kb * 2 + 2)), { color: '#B08FEA', width: 2.6, alpha: 0.95 });
    }
  }
  const sg = distillResults ? distillResults.straightness : null;
  foot.innerHTML =
    `同 14 份噪声种子,三个真实模型:<b style="color:#D65077">教师 u_fm(弯,40 步)</b> · ` +
    `<b style="color:#1FA9A3">ReFlow 学生(直,同 40 步;2 步端点漂移 ${sg ? sg.reflow.gap_2step.toFixed(4) : '…'} vs 教师 ${sg ? sg.teacher.gap_2step.toFixed(3) : '…'} ≈ 132 倍改善)</b> · ` +
    `<b>一步学生(虚线弦,NFE=1;= MeanFlow 取区间为全轴的特例)</b>。` +
    `<span style="color:#B08FEA">紫色粗段</span> = 卡⑤区间 [r, λ] 在教师真轨迹上的对应弯路——区间映射 Φ 要一口吞掉的就是它,在卡⑤拖 r 看它伸缩。`;
}
