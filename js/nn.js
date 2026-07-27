// nn.js — run the actual trained PyTorch MLPs in the browser.
// Weights arrive as base64 Float32; forward pass is hand-rolled matmuls.
// Every model exposes ONE canonical method: denoiseVP(x, lam) -> x0_hat,
// converting from its native head (eps / EDM-D / FM-u) with the exact affine maps.

export function b64f32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

export function vpFromLam(lam) { // VP: a^2 = sigmoid(lam)
  const a2 = 1 / (1 + Math.exp(-lam));
  return { a: Math.sqrt(a2), s: Math.sqrt(1 - a2) };
}
export function fmTFromLam(lam) { return 1 / (1 + Math.exp(lam / 2)); } // t = sigmoid(-lam/2)
export function sigVEFromLam(lam) { return Math.exp(-lam / 2); }

export class MLPModel {
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

export async function loadModel(url) {
  const spec = await (await fetch(url)).json();
  return new MLPModel(spec);
}
