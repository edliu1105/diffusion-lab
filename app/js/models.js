// models.js — one-time load of every trained network + the analytic GMM.
// Panels import from here; nothing else touches the data files.

import { loadModel } from './nn.js';
import { GMM } from './gmm.js';

export const R = {  // registry
  ready: null, meta: null, results: null,
  models: {},      // eps_ddpm, x0_edm, u_fm, u_fm_cond, u_fm_moons (+ distill later)
  gmm: null,
  mnist: null,     // filled by panels that need prerendered assets
};

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

export function load() {
  if (R.ready) return R.ready;
  R.ready = (async () => {
    R.results = await fetchJSON('data/toy2d_results.json');
    R.meta = R.results.meta;
    const g = R.meta.gmm;
    R.gmm = new GMM(g.mu, g.std, g.w);
    const names = ['eps_ddpm', 'x0_edm', 'u_fm', 'u_fm_cond'];
    for (const opt of ['u_fm_moons', 'u_reflow', 'u_1step']) names.push(opt);
    await Promise.all(names.map(async n => {
      try { R.models[n] = await loadModel(`data/model_${n}.json`); }
      catch (e) { console.warn(`model ${n} not available:`, e.message); }
    }));
    return R;
  })();
  return R.ready;
}

// canonical denoisers, all VP-frame x0_hat
export function denoisers() {
  const d = { analytic: (x, lam, n) => R.gmm.denoiseVP(x, lam, n) };
  for (const [name, m] of Object.entries(R.models)) {
    if (name === 'u_fm_cond' || name === 'u_1step') continue;
    d[name] = (x, lam, n) => m.denoiseVP(x, lam, n);
  }
  return d;
}
