// bundled from app/js/{gmm,nn,diff}.js — ESM stripped, classic script
(function(){
// ==== gmm.js ====
// gmm.js — the analytic ground truth, exact in the browser.
// For an isotropic Gaussian mixture, E[x0|x_t] and the posterior variance
// (= the irreducible loss floor) are closed-form. No weights, no approximation:
// this is the object every trained model is trying to become.

class GMM {
  constructor(mu, std, w) {       // mu: [[x,y]...], std: [k], w: [k]
    this.mu = mu; this.std = std;
    const Z = w.reduce((a, b) => a + b, 0);
    this.logw = w.map(v => Math.log(v / Z));
    this.K = mu.length;
  }

  // x0_hat at one point. a, s scalars (VP or any frame's alpha/sigma).
  // onlyComp: condition on a single mode (class-conditional denoiser, exact CFG analytics).
  denoisePoint(x0v, x1v, a, s, onlyComp = -1) {
    const K = this.K;
    const logp = new Array(K), gain = new Array(K), dx = new Array(K), dy = new Array(K);
    let mx = -Infinity;
    for (let k = 0; k < K; k++) {
      const vt = (a * this.std[k]) ** 2 + s * s;
      dx[k] = x0v - a * this.mu[k][0]; dy[k] = x1v - a * this.mu[k][1];
      logp[k] = -0.5 * (dx[k] * dx[k] + dy[k] * dy[k]) / vt - Math.log(vt) + this.logw[k];
      gain[k] = a * this.std[k] * this.std[k] / vt;
      if (logp[k] > mx) mx = logp[k];
    }
    let Z = 0; const r = new Array(K);
    for (let k = 0; k < K; k++) { r[k] = Math.exp(logp[k] - mx); Z += r[k]; }
    let ox = 0, oy = 0;
    for (let k = 0; k < K; k++) {
      const rr = (onlyComp >= 0) ? (k === onlyComp ? 1 : 0) : r[k] / Z;
      ox += rr * (this.mu[k][0] + gain[k] * dx[k]);
      oy += rr * (this.mu[k][1] + gain[k] * dy[k]);
    }
    return [ox, oy];
  }

  // batch: x (n*2) in VP frame, lam (n) -> x0_hat (n*2)
  denoiseVP(x, lam, n, onlyComp = -1) {
    const out = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      const a2 = 1 / (1 + Math.exp(-lam[i]));
      const a = Math.sqrt(a2), s = Math.sqrt(1 - a2);
      const [ox, oy] = this.denoisePoint(x[i * 2], x[i * 2 + 1], a, s, onlyComp);
      out[i * 2] = ox; out[i * 2 + 1] = oy;
    }
    return out;
  }

  // E[||x0 - E[x0|x_t]||^2 | x_t] at one point — the loss floor, pointwise.
  posteriorVarPoint(x0v, x1v, a, s) {
    const K = this.K;
    const logp = new Array(K); let mx = -Infinity;
    const pm = new Array(K), pv = new Array(K);
    for (let k = 0; k < K; k++) {
      const vt = (a * this.std[k]) ** 2 + s * s;
      const dx = x0v - a * this.mu[k][0], dy = x1v - a * this.mu[k][1];
      logp[k] = -0.5 * (dx * dx + dy * dy) / vt - Math.log(vt) + this.logw[k];
      const g = a * this.std[k] * this.std[k] / vt;
      pm[k] = [this.mu[k][0] + g * dx, this.mu[k][1] + g * dy];
      pv[k] = this.std[k] * this.std[k] * s * s / vt;   // per-dim
      if (logp[k] > mx) mx = logp[k];
    }
    let Z = 0; const r = new Array(K);
    for (let k = 0; k < K; k++) { r[k] = Math.exp(logp[k] - mx); Z += r[k]; }
    let mmx = 0, mmy = 0, second = 0;
    for (let k = 0; k < K; k++) {
      const rr = r[k] / Z;
      mmx += rr * pm[k][0]; mmy += rr * pm[k][1];
      second += rr * (2 * pv[k] + pm[k][0] * pm[k][0] + pm[k][1] * pm[k][1]);
    }
    return second - (mmx * mmx + mmy * mmy);
  }

  sample(n, rng) {
    const out = new Float32Array(n * 2), cls = new Int32Array(n);
    const w = this.logw.map(Math.exp);
    for (let i = 0; i < n; i++) {
      let u = rng(), k = 0;
      while (k < this.K - 1 && u > w[k]) { u -= w[k]; k++; }
      cls[i] = k;
      out[i * 2] = this.mu[k][0] + this.std[k] * gauss(rng);
      out[i * 2 + 1] = this.mu[k][1] + this.std[k] * gauss(rng);
    }
    return { x: out, cls };
  }
}

// deterministic RNG (mulberry32) + gaussian; seeds make every demo replayable
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
// ==== nn.js ====
// nn.js — run the actual trained PyTorch MLPs in the browser.
// Weights arrive as base64 Float32; forward pass is hand-rolled matmuls.
// Every model exposes ONE canonical method: denoiseVP(x, lam) -> x0_hat,
// converting from its native head (eps / EDM-D / FM-u) with the exact affine maps.

function b64f32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

function vpFromLam(lam) { // VP: a^2 = sigmoid(lam)
  const a2 = 1 / (1 + Math.exp(-lam));
  return { a: Math.sqrt(a2), s: Math.sqrt(1 - a2) };
}
function fmTFromLam(lam) { return 1 / (1 + Math.exp(lam / 2)); } // t = sigmoid(-lam/2)
function sigVEFromLam(lam) { return Math.exp(-lam / 2); }

class MLPModel {
  constructor(spec) {
    this.type = spec.type;                  // 'eps' | 'edm' | 'u'
    this.lamScale = spec.lam_scale;
    this.freqs = spec.freqs;
    this.sigmaData = spec.sigma_data;
    this.nClass = spec.n_class || 0;
    this.layers = spec.layers.map(l => ({ w: b64f32(l.w), b: b64f32(l.b), out: l.shape[0], in: l.shape[1] }));
    if (this.nClass) { this.emb = b64f32(spec.emb); this.embDim = spec.emb_dim; }
    this.inDim = this.layers[0].in;
    this._ws = null; this._wsN = 0;
  }

  _workspace(n) {
    if (this._wsN < n) {
      const maxH = Math.max(...this.layers.map(l => l.out), this.inDim);
      this._ws = [new Float32Array(n * maxH), new Float32Array(n * maxH)];
      this._wsN = n;
    }
    return this._ws;
  }

  // Native forward: x is Float32Array (n*2) in the model's OWN frame, lam per-sample (n), cls per-sample or null.
  forward(x, lam, n, cls = null, out = null) {
    const [bufA, bufB] = this._workspace(n);
    const inD = this.inDim;
    // build input features
    for (let i = 0; i < n; i++) {
      const o = i * inD;
      bufA[o] = x[i * 2]; bufA[o + 1] = x[i * 2 + 1];
      const l = lam[i] * this.lamScale;
      bufA[o + 2] = l;
      let k = 3;
      for (const f of this.freqs) { bufA[o + k++] = Math.sin(f * l); bufA[o + k++] = Math.cos(f * l); }
      if (this.nClass) {
        const c = cls === null ? this.nClass - 1 : cls[i]; // null token = last index (6 for gmm6)
        for (let e = 0; e < this.embDim; e++) bufA[o + k + e] = this.emb[c * this.embDim + e];
      }
    }
    // dense layers, SiLU between
    let src = bufA, dst = bufB, srcD = inD;
    for (let li = 0; li < this.layers.length; li++) {
      const { w, b, out: oD, in: iD } = this.layers[li];
      for (let i = 0; i < n; i++) {
        const so = i * srcD, dOff = i * oD;
        for (let j = 0; j < oD; j++) {
          let acc = b[j];
          const wOff = j * iD;
          for (let k2 = 0; k2 < iD; k2++) acc += w[wOff + k2] * src[so + k2];
          dst[dOff + j] = (li < this.layers.length - 1) ? acc / (1 + Math.exp(-acc)) : acc; // SiLU
        }
      }
      [src, dst] = [dst, src]; srcD = oD;
    }
    const res = out || new Float32Array(n * 2);
    for (let i = 0; i < n * 2; i++) res[i] = src[i];
    return res;
  }

  // Canonical API. x in VP frame (n*2), lam (n). Returns x0_hat (n*2).
  // Frame changes are pure per-lambda scalars — that scalar IS the lesson.
  denoiseVP(x, lam, n, cls = null) {
    const xn = new Float32Array(n * 2);
    const out = new Float32Array(n * 2);
    if (this.type === 'eps') {
      this.forward(x, lam, n, cls, out);
      for (let i = 0; i < n; i++) {
        const { a, s } = vpFromLam(lam[i]);
        const aa = Math.max(a, 1e-6);
        out[i * 2] = (x[i * 2] - s * out[i * 2]) / aa;
        out[i * 2 + 1] = (x[i * 2 + 1] - s * out[i * 2 + 1]) / aa;
      }
      return out;
    }
    if (this.type === 'edm') {
      // VE frame: x_ve = x_vp / a ; D = c_skip*x + c_out*F(c_in*x)
      const sd = this.sigmaData;
      const cin = new Float32Array(n), cskip = new Float32Array(n), cout = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const { a } = vpFromLam(lam[i]);
        const sig = sigVEFromLam(lam[i]);
        const xv0 = x[i * 2] / Math.max(a, 1e-9), xv1 = x[i * 2 + 1] / Math.max(a, 1e-9);
        const d2 = sig * sig + sd * sd;
        cskip[i] = sd * sd / d2; cout[i] = sig * sd / Math.sqrt(d2); cin[i] = 1 / Math.sqrt(d2);
        xn[i * 2] = cin[i] * xv0; xn[i * 2 + 1] = cin[i] * xv1;
      }
      this.forward(xn, lam, n, cls, out);
      for (let i = 0; i < n; i++) {
        const { a } = vpFromLam(lam[i]);
        const xv0 = x[i * 2] / Math.max(a, 1e-9), xv1 = x[i * 2 + 1] / Math.max(a, 1e-9);
        out[i * 2] = cskip[i] * xv0 + cout[i] * out[i * 2];
        out[i * 2 + 1] = cskip[i] * xv1 + cout[i] * out[i * 2 + 1];
      }
      return out;
    }
    // 'u': FM frame: x_fm = x_vp * (1-t)/a ; x0 = x_fm - t*u
    for (let i = 0; i < n; i++) {
      const { a } = vpFromLam(lam[i]);
      const t = fmTFromLam(lam[i]);
      const sc = (1 - t) / Math.max(a, 1e-9);
      xn[i * 2] = x[i * 2] * sc; xn[i * 2 + 1] = x[i * 2 + 1] * sc;
    }
    this.forward(xn, lam, n, cls, out);
    for (let i = 0; i < n; i++) {
      const t = fmTFromLam(lam[i]);
      const { a } = vpFromLam(lam[i]);
      const sc = (1 - t) / Math.max(a, 1e-9);
      out[i * 2] = xn[i * 2] - t * out[i * 2];
      out[i * 2 + 1] = xn[i * 2 + 1] - t * out[i * 2 + 1];
      // x0 is frame-free (alpha=1 at lam=+inf in every frame): no rescale needed
      void sc;
    }
    return out;
  }
}

async function loadModel(url) {
  const spec = await (await fetch(url)).json();
  return new MLPModel(spec);
}
// ==== diff.js ====
// diff.js — samplers. Every sampler consumes the SAME callable
//   denoise(x_vp, lam, n) -> x0_hat
// and differs only in: which frame it walks in, which grid it walks on,
// and how much noise it re-injects. That code shape is the whole point.



// ---- frame helpers: at equal logSNR, frames differ by a per-lambda SCALAR ----
function vpToFM(xv, lam) { const { a } = vpFromLam(lam); const t = fmTFromLam(lam); return xv * (1 - t) / Math.max(a, 1e-9); }
function fmToVP(xf, lam) { const { a } = vpFromLam(lam); const t = fmTFromLam(lam); return xf * a / Math.max(1 - t, 1e-9); }
function veToVP(xe, lam) { const { a } = vpFromLam(lam); return xe * a; }
function lamFromT_fm(t) { return 2 * Math.log((1 - t) / Math.max(t, 1e-12)); }
function lamFromSig_ve(s) { return -2 * Math.log(s); }

function karrasSigmas(n, smin = 0.02, smax = 8.0, rho = 7.0) {
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

function fmEuler(denoise, x1, n, steps, { tStart = 0.985, tEnd = 0.002 } = {}) {
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
function ddim(denoise, xT, n, steps, { eta = 0, lamMin = -8.5, lamMax = 9.0, rng = null } = {}) {
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
function edmHeun(denoise, xVE, n, steps, { churn = 0, smin = 0.02, smax = 8.0, rng = null, order = 2 } = {}) {
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

function runToEnd(sampler) { while (!sampler.done) sampler.step(); return sampler.xs; }

// CFG as a denoiser wrapper — head-agnostic because every head is affine in x0_hat.
function cfgDenoise(model, w, cls, nullCls) {
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
function endpointGap(xsA, xsB, n) {
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const dx = xsA[i * 2] - xsB[i * 2], dy = xsA[i * 2 + 1] - xsB[i * 2 + 1];
    acc += Math.sqrt(dx * dx + dy * dy);
  }
  return acc / n;
}

window.HF = { MLPModel, vpFromLam, fmTFromLam, sigVEFromLam, b64f32,
  GMM, mulberry32, gauss,
  fmEuler, ddim, edmHeun, runToEnd, cfgDenoise, endpointGap, karrasSigmas,
  vpToFM, fmToVP, veToVP, lamFromT_fm, lamFromSig_ve };
})();
