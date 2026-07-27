// p_hero.js — the opening image: a REAL flow-matching model turning noise into
// data, live, drawn as an assembly line (left = noise, right = data).

import { R } from './models.js';
import { setupCanvas, lamColor } from './charts.js';
import { mulberry32, gauss } from './gmm.js';
import { fmEuler, fmToVP, lamFromT_fm } from './diff.js';

export async function init() {
  const cv = document.getElementById('herostrip');
  const W = cv.parentElement.clientWidth - 2, H = 64;
  const ctx = setupCanvas(cv, W, H);
  const model = R.models.u_fm;
  if (!model) return;

  const N = 240, STEPS = 24;
  const rng = mulberry32((Math.random() * 1e9) | 0);
  const x1 = new Float32Array(N * 2);
  for (let i = 0; i < N * 2; i++) x1[i] = gauss(rng);
  const denoise = (x, lam, n) => model.denoiseVP(x, lam, n);
  const s = fmEuler(denoise, x1, N, STEPS);
  const snaps = [{ xs: Float32Array.from(s.xs), lam: lamFromT_fm(0.985) }];
  while (!s.done) {
    const r = s.step();
    snaps.push({ xs: Float32Array.from(s.xs), lam: r.lam });
  }
  // draw: each snapshot is a thin vertical scatter slice, colored by its lam
  ctx.clearRect(0, 0, W, H);
  const cols = snaps.length, colW = W / cols;
  for (let c = 0; c < cols; c++) {
    const { xs, lam } = snaps[c];
    const col = lamColor(Math.max(-9, Math.min(10, lam)), 0.85);
    ctx.fillStyle = col;
    for (let i = 0; i < N; i++) {
      // convert FM-frame position to VP for a bounded display
      const vx = fmToVP(xs[i * 2], lam), vy = fmToVP(xs[i * 2 + 1], lam);
      const px = c * colW + colW * 0.5 + vx * colW * 0.16;
      const py = H / 2 - vy * (H / 2 / 2.4);
      ctx.fillRect(px, Math.max(0, Math.min(H - 1, py)), 1.4, 1.4);
    }
  }
}
