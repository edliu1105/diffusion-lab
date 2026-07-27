// gmm.js — the analytic ground truth, exact in the browser.
// For an isotropic Gaussian mixture, E[x0|x_t] and the posterior variance
// (= the irreducible loss floor) are closed-form. No weights, no approximation:
// this is the object every trained model is trying to become.

export class GMM {
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
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
