// diff.js — samplers. Every sampler consumes the SAME callable
//   denoise(x_vp, lam, n) -> x0_hat
// and differs only in: which frame it walks in, which grid it walks on,
// and how much noise it re-injects. That code shape is the whole point.

import { vpFromLam, fmTFromLam } from './nn.js';
import { gauss } from './gmm.js';

// ---- frame helpers: at equal logSNR, frames differ by a per-lambda SCALAR ----
export function vpToFM(xv, lam) { const { a } = vpFromLam(lam); const t = fmTFromLam(lam); return xv * (1 - t) / Math.max(a, 1e-9); }
export function fmToVP(xf, lam) { const { a } = vpFromLam(lam); const t = fmTFromLam(lam); return xf * a / Math.max(1 - t, 1e-9); }
export function veToVP(xe, lam) { const { a } = vpFromLam(lam); return xe * a; }
export function lamFromT_fm(t) { return 2 * Math.log((1 - t) / Math.max(t, 1e-12)); }
export function lamFromSig_ve(s) { return -2 * Math.log(s); }

export function karrasSigmas(n, smin = 0.02, smax = 8.0, rho = 7.0) {
  const out = [];
  for (let i = 0; i <= n; i++) {
    const f = i / n;
    out.push((smax ** (1 / rho) + f * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho);
  }
  out[n] = 0;
  return out;
}

// Each sampler: create(denoise, opts) -> {frame, xs, k, done, step(), readout}
// xs is Float32Array (n*2) in the sampler's NATIVE frame. Panels convert for display.

export function fmEuler(denoise, x1, n, steps, { tStart = 0.985, tEnd = 0.002 } = {}) {
  // tStart<1: every family integrates from a FINITE noise ceiling (EDM's sigma_max,
  // DDPM's lambda_min, FM's t_start) — the model never trained at lam = -inf.
  const ts = Array.from({ length: steps + 1 }, (_, i) => tStart + (tEnd - tStart) * i / steps);
  const xs = Float32Array.from(x1);
  let k = 0;
  const lamArr = new Float32Array(n);
  return {
    frame: 'fm', xs, steps,
    get k() { return k; }, get done() { return k >= steps; },
    step() {
      if (k >= steps) return null;
      const t = ts[k], tn = ts[k + 1], lam = lamFromT_fm(Math.max(t, 1e-6));
      lamArr.fill(lam);
      // convert to VP for the canonical denoiser, x0_hat comes back frame-free
      const xvp = new Float32Array(n * 2);
      const { a } = vpFromLam(lam); const sc = a / Math.max(1 - t, 1e-9);
      for (let i = 0; i < n * 2; i++) xvp[i] = xs[i] * sc;
      const x0h = denoise(xvp, lamArr, n);
      let du = 0;
      for (let i = 0; i < n * 2; i++) {
        const u = (xs[i] - x0h[i]) / Math.max(t, 1e-6);
        const d = (tn - t) * u;
        xs[i] += d; du += d * d;
      }
      k++;
      return { t, lam, dxRms: Math.sqrt(du / n), x0h, label: `t=${t.toFixed(3)}  λ=${lam.toFixed(2)}` };
    }
  };
}

// DDIM over a logSNR grid; eta=0 deterministic, eta=1 ~ ancestral DDPM.
export function ddim(denoise, xT, n, steps, { eta = 0, lamMin = -8.5, lamMax = 9.0, rng = null } = {}) {
  const lams = Array.from({ length: steps + 1 }, (_, i) => lamMin + (lamMax - lamMin) * i / steps);
  const xs = Float32Array.from(xT);
  let k = 0;
  const lamArr = new Float32Array(n);
  return {
    frame: 'vp', xs, steps,
    get k() { return k; }, get done() { return k >= steps; },
    step() {
      if (k >= steps) return null;
      const lam = lams[k], lamN = lams[k + 1];
      const { a, s } = vpFromLam(lam), { a: an, s: sn } = vpFromLam(lamN);
      lamArr.fill(lam);
      const x0h = denoise(xs, lamArr, n);
      const tauF = eta * (sn / s) * Math.sqrt(Math.max(0, 1 - (a / an) ** 2));
      const tau = Math.min(tauF, sn);
      const sig = Math.sqrt(Math.max(0, sn * sn - tau * tau));
      let du = 0;
      for (let i = 0; i < n * 2; i++) {
        const epsh = (xs[i] - a * x0h[i]) / s;
        let nx = an * x0h[i] + sig * epsh;
        if (tau > 0 && rng) nx += tau * gauss(rng);
        du += (nx - xs[i]) ** 2; xs[i] = nx;
      }
      k++;
      return { lam, a, s, dxRms: Math.sqrt(du / n), x0h, label: `λ=${lam.toFixed(2)}  α=${a.toFixed(3)} σ=${s.toFixed(3)}${eta > 0 ? '  +noise' : ''}` };
    }
  };
}

// EDM Heun with optional churn. Walks in VE frame on a Karras sigma grid.
export function edmHeun(denoise, xVE, n, steps, { churn = 0, smin = 0.02, smax = 8.0, rng = null, order = 2 } = {}) {
  const sig = karrasSigmas(steps, smin, smax);
  const xs = Float32Array.from(xVE);
  let k = 0;
  const lamArr = new Float32Array(n);
  const callD = (x, s) => {
    const lam = lamFromSig_ve(Math.max(s, 1e-9));
    lamArr.fill(lam);
    const { a } = vpFromLam(lam);
    const xvp = new Float32Array(n * 2);
    for (let i = 0; i < n * 2; i++) xvp[i] = x[i] * a;
    return denoise(xvp, lamArr, n);
  };
  return {
    frame: 've', xs, steps, nfePerStep: order,
    get k() { return k; }, get done() { return k >= steps; },
    step() {
      if (k >= steps) return null;
      let s = sig[k]; const sn = sig[k + 1];
      if (churn > 0 && rng && s > 0) {
        const gamma = Math.min(churn / steps, Math.SQRT2 - 1);
        const sh = s * (1 + gamma);
        const add = Math.sqrt(Math.max(sh * sh - s * s, 0));
        for (let i = 0; i < n * 2; i++) xs[i] += add * gauss(rng);
        s = sh;
      }
      const x0h = callD(xs, s);
      const d = new Float32Array(n * 2);
      for (let i = 0; i < n * 2; i++) d[i] = (xs[i] - x0h[i]) / s;
      let du = 0;
      if (sn > 0 && order === 2) {
        const xe = new Float32Array(n * 2);
        for (let i = 0; i < n * 2; i++) xe[i] = xs[i] + (sn - s) * d[i];
        const x0h2 = callD(xe, sn);
        for (let i = 0; i < n * 2; i++) {
          const d2 = (xe[i] - x0h2[i]) / sn;
          const nx = xs[i] + (sn - s) * 0.5 * (d[i] + d2);
          du += (nx - xs[i]) ** 2; xs[i] = nx;
        }
      } else {
        for (let i = 0; i < n * 2; i++) {
          const nx = xs[i] + (sn - s) * d[i];
          du += (nx - xs[i]) ** 2; xs[i] = nx;
        }
      }
      k++;
      return { sigma: s, lam: lamFromSig_ve(Math.max(s, 1e-9)), dxRms: Math.sqrt(du / n), x0h, label: `σ=${s.toFixed(3)}${churn > 0 ? '  churn' : ''}` };
    }
  };
}

export function runToEnd(sampler) { while (!sampler.done) sampler.step(); return sampler.xs; }

// CFG as a denoiser wrapper — head-agnostic because every head is affine in x0_hat.
export function cfgDenoise(model, w, cls, nullCls) {
  return (x, lam, n) => {
    const clsArr = new Int32Array(n).fill(cls);
    const nullArr = new Int32Array(n).fill(nullCls);
    const dc = model.denoiseVP(x, lam, n, clsArr);
    if (w === 1) return dc;
    const du = model.denoiseVP(x, lam, n, nullArr);
    const out = new Float32Array(n * 2);
    for (let i = 0; i < n * 2; i++) out[i] = du[i] + w * (dc[i] - du[i]);
    return out;
  };
}

// endpoint error vs a reference run from identical seeds (mean L2)
export function endpointGap(xsA, xsB, n) {
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const dx = xsA[i * 2] - xsB[i * 2], dy = xsA[i * 2 + 1] - xsB[i * 2 + 1];
    acc += Math.sqrt(dx * dx + dy * dy);
  }
  return acc / n;
}
