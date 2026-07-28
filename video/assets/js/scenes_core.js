// scenes_core.js — canvas scene engine + scenes S0–S4.
// Every draw is a PURE function of absolute time t (seconds). No clocks, no rAF,
// no unseeded random. All stochastic content is precomputed at load with seeded PRNG.
(function () {
  const W = 1920, H = 1080;
  const C = {
    bg: '#0C0F14', panel: '#161B23', ink: '#E8E4DA', ink2: '#9AA3B2', ink3: '#5C6675',
    ddpm: '#5E8FE8', ddim: '#A26BE4', edm: '#D89A3A', fm: '#E86A8A', student: '#2FBFB9',
    ok: '#79C99E', bad: '#E06C5F', grid: 'rgba(154,163,178,0.10)',
  };
  const MODE_COLORS = ['#8FA8D8', '#C39ED0', '#D89AA6', '#CBB183', '#9DBE93', '#84BBB8'];
  const LAM_STOPS = [[-9, [110, 116, 238]], [-4, [122, 130, 210]], [0, [150, 148, 160]], [4, [210, 165, 92]], [10, [242, 176, 61]]];
  function lamColor(lam, a = 1) {
    let i = 0;
    while (i < LAM_STOPS.length - 2 && lam > LAM_STOPS[i + 1][0]) i++;
    const [l0, c0] = LAM_STOPS[i], [l1, c1] = LAM_STOPS[i + 1];
    const f = Math.min(1, Math.max(0, (lam - l0) / (l1 - l0)));
    const c = c0.map((v, k) => Math.round(v + f * (c1[k] - v)));
    return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  }
  const clamp01 = v => Math.min(1, Math.max(0, v));
  const ease = p => p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2; // easeInOutQuad
  const lerp = (a, b, p) => a + (b - a) * p;
  // window phase: 0→1 over [t0,t0+d]
  const ph = (t, t0, d) => clamp01((t - t0) / d);

  const HF = window.HF, D = window.HF_DATA;
  const gmm = new HF.GMM(D.gmm.mu, D.gmm.std, D.gmm.w);
  const models = {};
  for (const [k, v] of Object.entries(D.models)) models[k] = new HF.MLPModel(v);
  const denU = (x, lam, n) => models.u_fm.denoiseVP(x, lam, n);
  const denA = (x, lam, n) => gmm.denoiseVP(x, lam, n);

  // ---------- shared drawing helpers ----------
  function text(ctx, s, x, y, { size = 36, color = C.ink, align = 'left', font = '"Noto Sans SC", sans-serif', weight = '', alpha = 1 } = {}) {
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.font = `${weight} ${size}px ${font}`.trim();
    ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = 'middle';
    ctx.fillText(s, x, y); ctx.restore();
  }
  const mono = '"JetBrains Mono", "Noto Sans SC", monospace';
  function panel(ctx, x, y, w, h, { alpha = 1, fill = 'rgba(22,27,35,0.85)', stroke = 'rgba(154,163,178,0.25)' } = {}) {
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 14); ctx.fill(); ctx.stroke(); ctx.restore();
  }
  function spectrumStrip(ctx, x, y, w, h, { lamMin = -9, lamMax = 9, alpha = 1, cursor = null } = {}) {
    ctx.save(); ctx.globalAlpha = alpha;
    for (let px = 0; px < w; px += 2) {
      ctx.fillStyle = lamColor(lamMin + (lamMax - lamMin) * px / w);
      ctx.fillRect(x + px, y, 2, h);
    }
    if (cursor != null) {
      const cx = x + (cursor - lamMin) / (lamMax - lamMin) * w;
      ctx.fillStyle = C.ink; ctx.fillRect(cx - 3, y - 8, 6, h + 16);
    }
    ctx.restore();
  }
  function axisBox(ctx, x, y, w, h, alpha = 1) {
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.strokeStyle = 'rgba(154,163,178,0.35)'; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }
  function polyline(ctx, pts, { color = C.ink, width = 3, alpha = 1, dash = [], frac = 1 } = {}) {
    if (pts.length < 2) return;
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    const n = Math.max(2, Math.floor(pts.length * clamp01(frac)));
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < n; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke(); ctx.restore();
  }
  function arrow(ctx, x0, y0, x1, y1, { color = C.ink, width = 4, alpha = 1, head = 14 } = {}) {
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const ang = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - head * Math.cos(ang - 0.45), y1 - head * Math.sin(ang - 0.45));
    ctx.lineTo(x1 - head * Math.cos(ang + 0.45), y1 - head * Math.sin(ang + 0.45));
    ctx.closePath(); ctx.fill(); ctx.restore();
  }

  // ============================================================ S0 hook
  // real u_fm sampling, 300 particles, 26 recorded steps
  const S0 = (() => {
    const N = 300, STEPS = 26;
    const rng = HF.mulberry32(20260728);
    const x1 = new Float32Array(N * 2);
    for (let i = 0; i < N * 2; i++) x1[i] = HF.gauss(rng);
    const s = HF.fmEuler(denU, x1, N, STEPS);
    const snaps = [{ xs: Float32Array.from(s.xs), lam: -8.4 }];
    while (!s.done) { const r = s.step(); snaps.push({ xs: Float32Array.from(s.xs), lam: Math.min(9, r.lam) }); }
    return { snaps, N, STEPS };
  })();
  function drawS0(ctx, t, T) {
    const local = t - T.s0.start, dur = T.s0.dur;
    // particles denoise over first ~70% of scene, hold after
    const p = ease(ph(local, 0.4, dur * 0.62));
    const fi = p * (S0.snaps.length - 1);
    const i0 = Math.min(S0.snaps.length - 2, Math.floor(fi)), f = fi - i0;
    const a = S0.snaps[i0], b = S0.snaps[i0 + 1];
    const lam = lerp(a.lam, b.lam, f);
    const cx = W / 2, cy = H / 2 + 30, sc = 128;
    ctx.save();
    for (let i = 0; i < S0.N; i++) {
      const xf = lerp(a.xs[i * 2], b.xs[i * 2], f), yf = lerp(a.xs[i * 2 + 1], b.xs[i * 2 + 1], f);
      const vx = HF.fmToVP(xf, lam), vy = HF.fmToVP(yf, lam);
      ctx.fillStyle = lamColor(lam, 0.9);
      ctx.beginPath(); ctx.arc(cx + vx * sc, cy - vy * sc, 5, 0, 7); ctx.fill();
    }
    ctx.restore();
    spectrumStrip(ctx, cx - 420, cy + 316, 840, 12, { cursor: lam, alpha: 0.9 });
    text(ctx, '纯噪声', cx - 420, cy + 352, { size: 26, color: C.ink3, align: 'left', font: mono });
    text(ctx, '数据', cx + 420, cy + 352, { size: 26, color: C.ink3, align: 'right', font: mono });
    text(ctx, `λ = ${lam.toFixed(1)}`, cx, cy + 352, { size: 26, color: C.ink2, align: 'center', font: mono });
  }

  // ============================================================ S1 the axis
  const S1 = (() => {
    // decision band: responsibility entropy exp over sigma_bar grid, 3 data scales
    const scales = [0.5, 1, 2];
    const grid = [];
    for (let g = -1.3; g <= 1.31; g += 0.035) grid.push(Math.pow(10, g)); // sigma_bar 0.05..20
    const rng = HF.mulberry32(77);
    const curves = scales.map(sc => grid.map(sb => {
      // E over x_t of exp(entropy of responsibilities), alpha=1 (VE view), data scaled by sc
      const n = 220; let acc = 0;
      for (let s2 = 0; s2 < n; s2++) {
        const k = (rng() * 6) | 0;
        const x0 = [D.gmm.mu[k][0] * sc + D.gmm.std[k] * sc * HF.gauss(rng), D.gmm.mu[k][1] * sc + D.gmm.std[k] * sc * HF.gauss(rng)];
        const xt = [x0[0] + sb * HF.gauss(rng), x0[1] + sb * HF.gauss(rng)];
        // responsibilities over scaled mixture
        let mx = -1e30; const lp = [];
        for (let m = 0; m < 6; m++) {
          const vt = (D.gmm.std[m] * sc) ** 2 + sb * sb;
          const dx = xt[0] - D.gmm.mu[m][0] * sc, dy = xt[1] - D.gmm.mu[m][1] * sc;
          const l = -0.5 * (dx * dx + dy * dy) / vt - Math.log(vt);
          lp.push(l); if (l > mx) mx = l;
        }
        let Z = 0; for (const l of lp) Z += Math.exp(l - mx);
        let ent = 0;
        for (const l of lp) { const r = Math.exp(l - mx) / Z; if (r > 1e-12) ent -= r * Math.log(r); }
        acc += Math.exp(ent);
      }
      return acc / n;
    }));
    return { grid, curves, scales };
  })();
  function drawS1(ctx, t, T) {
    const local = t - T.s1.start;
    const seg = T.s1.segs;
    // phase A (s1a..s1c): mixing + clocks ; phase B (s1d..s1e): decision band
    const bandStart = seg[3].start - T.s1.start;
    if (local < bandStart) {
      // -- mixing diagram: x0 cloud + eps cloud -> x_t, then the lambda axis with 4 clock rows
      const pA = ease(ph(local, 0.5, 3.0));
      const mixFade = 1 - 0.92 * ease(ph(local, seg[2].start - T.s1.start + 0.6, 1.6));
      const cy = 400;
      // data cloud (left), noise cloud (right), mix (center)
      const rng = HF.mulberry32(11);
      ctx.save();
      for (let i = 0; i < 260; i++) {
        const k = (rng() * 6) | 0;
        const x0 = [D.gmm.mu[k][0] + D.gmm.std[k] * HF.gauss(rng), D.gmm.mu[k][1] + D.gmm.std[k] * HF.gauss(rng)];
        const e = [HF.gauss(rng), HF.gauss(rng)];
        const mixT = ease(ph(local, 1.2, 2.2));
        const alpha = Math.cos(mixT * Math.PI / 2), sig = Math.sin(mixT * Math.PI / 2);
        const lx = 430 + x0[0] * 62, ly = cy - x0[1] * 62;
        const rx = 1490 + e[0] * 62, ry = cy - e[1] * 62;
        const mx = W / 2 + (alpha * x0[0] + sig * e[0]) * 62, my = cy - (alpha * x0[1] + sig * e[1]) * 62;
        ctx.globalAlpha = 0.85 * pA * mixFade;
        ctx.fillStyle = MODE_COLORS[k]; ctx.beginPath(); ctx.arc(lx, ly, 4, 0, 7); ctx.fill();
        ctx.fillStyle = 'rgba(110,116,238,0.8)'; ctx.beginPath(); ctx.arc(rx, ry, 4, 0, 7); ctx.fill();
        const lam = 2 * Math.log((alpha + 1e-4) / (sig + 1e-4));
        ctx.fillStyle = lamColor(Math.max(-9, Math.min(9, lam))); ctx.beginPath(); ctx.arc(mx, my, 4.5, 0, 7); ctx.fill();
      }
      ctx.restore();
      text(ctx, '数据 x₀', 430, cy + 190, { size: 30, align: 'center', color: C.ink2, alpha: mixFade });
      text(ctx, '噪声 ε', 1490, cy + 190, { size: 30, align: 'center', color: C.ink2, alpha: mixFade });
      text(ctx, 'x_t = α·x₀ + σ·ε', W / 2, cy + 190, { size: 34, align: 'center', color: C.ink, font: mono, alpha: pA * mixFade });
      // clocks: 4 schedules ticks on the shared lambda axis
      const p2 = ease(ph(local, seg[1].start - T.s1.start + 0.5, 2.5));
      if (p2 > 0) {
        const ax = 300, aw = 1320, ay = 800;
        spectrumStrip(ctx, ax, ay, aw, 14, { alpha: p2 });
        text(ctx, 'λ = logSNR', ax + aw / 2, ay + 46, { size: 28, align: 'center', color: C.ink2, font: mono, alpha: p2 });
        const rows = [
          ['DDPM · 1000 步(cosine)', C.ddpm, k => { const tt = (k + 0.5) / 60; const a2 = Math.cos((tt + 0.008) / 1.008 * Math.PI / 2) ** 2; return Math.log(a2 / (1 - a2 + 1e-9)); }, 60],
          ['EDM · 18 个 σ(ρ=7)', C.edm, k => { const i2 = k / 17; const s2 = (80 ** (1 / 7) + i2 * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7; return -2 * Math.log(s2); }, 18],
          ['FM · 28 个均匀 t', C.fm, k => { const tt = 0.02 + 0.96 * k / 27; return 2 * Math.log((1 - tt) / tt); }, 28],
          ['FM + shift 3(SD3 高分辨率)', '#C05A78', k => { let tt = 0.02 + 0.96 * k / 27; tt = 3 * tt / (1 + 2 * tt); return 2 * Math.log((1 - tt) / tt); }, 28],
        ];
        const p3 = ease(ph(local, seg[2].start - T.s1.start, 2.0));
        rows.forEach(([label, col, fn, n], r) => {
          const alpha = r < 1 ? p2 : p3;
          if (alpha <= 0) return;
          const yy = ay - 60 - r * 56;
          text(ctx, label, ax - 18, yy, { size: 24, align: 'right', color: col, font: mono, alpha });
          for (let k = 0; k < n; k++) {
            const lam = Math.max(-9, Math.min(9, fn(k)));
            const px = ax + (lam + 9) / 18 * aw;
            ctx.save(); ctx.globalAlpha = alpha * 0.9; ctx.fillStyle = col; ctx.fillRect(px, yy - 14, 2.5, 28); ctx.restore();
          }
        });
      }
    } else {
      // -- decision band
      const p = ease(ph(local, bandStart + 0.3, 2.2));
      const bx = 330, by = 260, bw = 1260, bh = 520;
      panel(ctx, bx - 40, by - 60, bw + 80, bh + 160, { alpha: p });
      axisBox(ctx, bx, by, bw, bh, p);
      text(ctx, '还没分清的簇数(责任熵的指数)', bx, by - 26, { size: 26, color: C.ink2, alpha: p });
      text(ctx, 'σ̄(log 轴)→ 噪声变大', bx + bw / 2, by + bh + 44, { size: 26, align: 'center', color: C.ink2, alpha: p });
      const X = i => bx + i / (S1.grid.length - 1) * bw;
      const Y = v => by + bh - (v - 1) / 5.2 * bh;
      // scale sweep on s1e
      const p2 = ph(local, seg[4].start - T.s1.start + 0.6, seg[4].dur - 1.2);
      const scIdx = p2 <= 0 ? 1 : (p2 < 0.5 ? 1 + p2 : 2); // morph 1x -> 2x
      const [cols] = [[C.ink3, C.ink, C.edm]];
      S1.curves.forEach((cv, ci) => {
        const isMain = ci === 1, isBig = ci === 2;
        let alpha = isMain ? p : 0;
        if (isBig) alpha = p * clamp01(p2 * 2);
        if (ci === 0) alpha = 0;
        if (alpha <= 0) return;
        polyline(ctx, cv.map((v, i) => [X(i), Y(v)]), { color: isBig ? C.edm : C.ink, width: 5, alpha, frac: p });
      });
      // step labels
      const labels = [['簇内散布', 0.10], ['子簇间距', 0.42], ['左右超簇', 0.80]];
      labels.forEach(([s2, fx], i) => {
        ctx.save(); ctx.globalAlpha = p * 0.8; ctx.strokeStyle = 'rgba(154,163,178,0.4)'; ctx.setLineDash([6, 6]); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(bx + fx * bw, by); ctx.lineTo(bx + fx * bw, by + bh); ctx.stroke(); ctx.restore();
        text(ctx, s2, bx + fx * bw, by + 30 + i * 0, { size: 24, align: 'center', color: C.ink3, alpha: p });
      });
      if (p2 > 0) {
        text(ctx, '数据放大 ×2 → 整条曲线右移', bx + bw - 20, by + 70, { size: 30, align: 'right', color: C.edm, alpha: clamp01(p2 * 2) });
        text(ctx, '= 高分辨率为什么必须 shift 调度', bx + bw - 20, by + 112, { size: 26, align: 'right', color: C.ink2, alpha: clamp01(p2 * 2) });
      }
      spectrumStrip(ctx, bx, by + bh + 76, bw, 10, { lamMin: 9, lamMax: -9, alpha: p * 0.8 });
    }
  }

  // ============================================================ S2 five readouts
  const S2 = (() => {
    // one pinned x_t; sweep lambda; arrows for x0hat & eps-hat decomposition
    const pin = [0.9, -1.35];
    const lams = [];
    for (let l = -8; l <= 8.01; l += 0.25) lams.push(l);
    // amplification curves
    const ampEps = lams.map(l => Math.exp(-l / 2));  // eps err -> x0 err
    const ampX0 = lams.map(l => Math.exp(l / 2));    // x0 err -> eps err ~ e^{lam/2}? (x0->eps amplifier alpha/sigma)
    // measured model error curves (from python eval)
    const ev = D.vs_analytic;
    return { pin, lams, ampEps, ampX0, ev };
  })();
  function drawS2(ctx, t, T) {
    const local = t - T.s2.start, seg = T.s2.segs;
    const leftX = 520, leftY = 430, sc = 165;
    const p0 = ease(ph(local, 0.4, 1.6));
    // left: field view with pinned point, lambda sweeps with narration
    const sweepP = ph(local, seg[1].start - T.s2.start, seg[1].dur + seg[0].dur * 0);
    const lam = lerp(2.5, -2.5, ease(ph(local, 1.0, seg[0].dur + seg[1].dur - 2)));
    // data scatter
    ctx.save();
    for (let i = 0; i < 900; i += 2) {
      const pnt = D.data_points[i];
      ctx.globalAlpha = 0.25 * p0; ctx.fillStyle = '#39414E';
      ctx.beginPath(); ctx.arc(leftX + pnt[0] * sc / 1.6, leftY - pnt[1] * sc / 1.6, 3, 0, 7); ctx.fill();
    }
    ctx.restore();
    // pinned x_t + heads
    const xs = new Float32Array([S2.pin[0], S2.pin[1]]);
    const lamA = new Float32Array([lam]);
    const x0h = denA(xs, lamA, 1);
    const a = Math.sqrt(1 / (1 + Math.exp(-lam))), s = Math.sqrt(1 - a * a);
    const eps = [(S2.pin[0] - a * x0h[0]) / s, (S2.pin[1] - a * x0h[1]) / s];
    const px = v => leftX + v * sc / 1.6, py = v => leftY - v * sc / 1.6;
    // decomposition: alpha*x0h (amber) + sigma*eps (violet) = x_t
    arrow(ctx, px(0), py(0), px(a * x0h[0]), py(a * x0h[1]), { color: C.edm, width: 6, alpha: p0 });
    arrow(ctx, px(a * x0h[0]), py(a * x0h[1]), px(S2.pin[0]), py(S2.pin[1]), { color: '#6E74EE', width: 6, alpha: p0 });
    ctx.save(); ctx.globalAlpha = p0; ctx.fillStyle = C.ink;
    ctx.beginPath(); ctx.arc(px(S2.pin[0]), py(S2.pin[1]), 10, 0, 7); ctx.fill(); ctx.restore();
    text(ctx, 'x_t', px(S2.pin[0]) + 24, py(S2.pin[1]), { size: 30, color: C.ink, font: mono, alpha: p0 });
    text(ctx, 'α·x̂₀', px(a * x0h[0] * 0.55), py(a * x0h[1] * 0.55) - 26, { size: 28, color: C.edm, font: mono, alpha: p0 });
    text(ctx, 'σ·ε̂', (px(a * x0h[0]) + px(S2.pin[0])) / 2 + 30, (py(a * x0h[1]) + py(S2.pin[1])) / 2, { size: 28, color: '#8B90F0', font: mono, alpha: p0 });
    spectrumStrip(ctx, leftX - 300, leftY + 330, 600, 10, { cursor: lam, alpha: p0 });
    text(ctx, `λ = ${lam.toFixed(1)}`, leftX, leftY + 372, { size: 26, align: 'center', color: C.ink2, font: mono, alpha: p0 });
    void sweepP;
    // right: readout card (five heads live numbers)
    const u = [eps[0] - x0h[0], eps[1] - x0h[1]];
    const v = [a * eps[0] - s * x0h[0], a * eps[1] - s * x0h[1]];
    const scr = [-eps[0] / s, -eps[1] / s];
    const rx = 1120, ry = 200;
    panel(ctx, rx, ry, 640, 470, { alpha: p0 });
    const rows = [
      ['x̂₀', x0h, C.ink], ['ε̂', eps, '#8B90F0'], ['v̂', v, C.ink2], ['û', u, C.fm], ['score', scr, C.bad],
    ];
    text(ctx, '同一次前向 → 五种读数(实时换算)', rx + 30, ry + 46, { size: 28, color: C.ink2, alpha: p0 });
    rows.forEach(([nm, val, col], i) => {
      const yy = ry + 110 + i * 70;
      text(ctx, nm, rx + 40, yy, { size: 34, color: col, font: mono, alpha: p0 });
      text(ctx, `(${val[0].toFixed(2).padStart(6)}, ${val[1].toFixed(2).padStart(6)})`, rx + 600, yy, { size: 32, color: col, font: mono, align: 'right', alpha: p0 });
    });
    // amplifier chart phase (s2c..s2e)
    const pAmp = ease(ph(local, seg[2].start - T.s2.start + 0.3, 2.2));
    if (pAmp > 0) {
      const bx = 240, by = 760, bw = 1440, bh = 260;
      panel(ctx, bx - 30, by - 56, bw + 60, bh + 120, { alpha: pAmp });
      axisBox(ctx, bx, by, bw, bh, pAmp);
      const X = i => bx + i / (S2.lams.length - 1) * bw;
      const Ylog = v => by + bh - clamp01((Math.log10(v) + 1) / 3.4) * bh;
      polyline(ctx, S2.ampEps.map((v2, i) => [X(i), Ylog(v2)]), { color: C.ddpm, width: 5, alpha: pAmp, frac: pAmp });
      polyline(ctx, S2.ampX0.map((v2, i) => [X(i), Ylog(v2)]), { color: C.edm, width: 5, alpha: pAmp, frac: pAmp });
      polyline(ctx, S2.lams.map((l, i) => [X(i), Ylog(1 + 0 * l)]), { color: C.fm, width: 5, alpha: pAmp, dash: [10, 8] });
      text(ctx, 'ε̂ 误差 → x̂₀ 误差(×e^{−λ/2}):λ=−9 处 ×90', bx + 30, by + bh - 36, { size: 26, color: C.ddpm, font: mono, alpha: pAmp });
      text(ctx, 'x̂₀ 误差 → ε̂ 误差(×e^{+λ/2})', bx + bw - 30, by + 40, { size: 26, color: C.edm, font: mono, align: 'right', alpha: pAmp });
      text(ctx, 'v / u / EDM:全程有界', bx + bw / 2 + 150, by + bh - 36, { size: 26, color: C.fm, font: mono, align: 'center', alpha: pAmp });
      text(ctx, 'λ →', bx + bw / 2, by + bh + 40, { size: 24, color: C.ink3, align: 'center', font: mono, alpha: pAmp });
      // measured 71x marker on s2d
      const pM = ease(ph(local, seg[3].start - T.s2.start + 0.5, 1.5));
      if (pM > 0) {
        text(ctx, '实测:λ=−8 处,ε 网络的场误差 = FM 网络的 71×', bx + bw / 2, by - 26, { size: 30, color: C.bad, align: 'center', alpha: pM, weight: 'bold' });
      }
    }
  }

  // ============================================================ S3 training microscope
  const S3 = (() => {
    const lams = []; for (let l = -9; l <= 10.01; l += 0.2) lams.push(l);
    const nPDF = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    const tOfLam = lam => { const a2 = 1 / (1 + Math.exp(-lam)); const s = 0.008, f0 = Math.cos(s / (1 + s) * Math.PI / 2); return Math.max(0, Math.min(1, (2 / Math.PI) * Math.acos(Math.sqrt(a2) * f0) * (1 + s) - s)); };
    const pCos = lams.map(l => Math.abs((tOfLam(l + 0.01) - tOfLam(l - 0.01)) / 0.02));
    const pFM = lams.map(l => nPDF(l, 0, 2));
    const pEDM = lams.map(l => nPDF(l, 0.8, 2.4));
    const curve = D.curve_u_fm; // snapshots: {step, x0_per_bin[24]}
    return { lams, pCos, pFM, pEDM, curve };
  })();
  function drawS3(ctx, t, T) {
    const local = t - T.s3.start, seg = T.s3.segs;
    // phase 1: six-line loop is DOM. canvas: density chart (s3b) then binned loss (s3c/d)
    const pD = ease(ph(local, seg[1].start - T.s3.start + 0.3, 2));
    if (pD > 0 && local < seg[2].start - T.s3.start + 1.0) {
      const bx = 330, by = 300, bw = 1260, bh = 460;
      panel(ctx, bx - 40, by - 60, bw + 80, bh + 150, { alpha: pD });
      axisBox(ctx, bx, by, bw, bh, pD);
      text(ctx, '三代配方的 λ 采样密度(解析式)', bx, by - 24, { size: 28, color: C.ink2, alpha: pD });
      const X = i => bx + i / (S3.lams.length - 1) * bw;
      const Y = v => by + bh - clamp01(v / 0.22) * bh;
      polyline(ctx, S3.pCos.map((v, i) => [X(i), Y(v)]), { color: C.ddpm, width: 5, alpha: pD, frac: pD });
      polyline(ctx, S3.pEDM.map((v, i) => [X(i), Y(v)]), { color: C.edm, width: 5, alpha: pD, frac: pD });
      polyline(ctx, S3.pFM.map((v, i) => [X(i), Y(v)]), { color: C.fm, width: 5, alpha: pD, frac: pD });
      text(ctx, 'DDPM uniform-t(副产品)', bx + 30, by + 50, { size: 25, color: C.ddpm, font: mono, alpha: pD });
      text(ctx, 'EDM lognormal', bx + 30, by + 88, { size: 25, color: C.edm, font: mono, alpha: pD });
      text(ctx, 'SD3 logit-normal', bx + 30, by + 126, { size: 25, color: C.fm, font: mono, alpha: pD });
      spectrumStrip(ctx, bx, by + bh + 40, bw, 10, { alpha: pD * 0.8 });
      // decision band shading mid
      ctx.save(); ctx.globalAlpha = pD * 0.12; ctx.fillStyle = C.ink;
      ctx.fillRect(bx + bw * 0.35, by, bw * 0.3, bh); ctx.restore();
      text(ctx, '决策带', bx + bw / 2, by + 34, { size: 26, align: 'center', color: C.ink2, alpha: pD });
    }
    // binned loss over steps
    const pL = ease(ph(local, seg[2].start - T.s3.start + 1.2, 2));
    if (pL > 0) {
      const bx = 330, by = 300, bw = 1260, bh = 500;
      panel(ctx, bx - 40, by - 60, bw + 80, bh + 160, { alpha: pL });
      axisBox(ctx, bx, by, bw, bh, pL);
      text(ctx, '真实训练:每个 λ 桶的 x̂₀-loss 随步数(log 轴)', bx, by - 24, { size: 28, color: C.ink2, alpha: pL });
      const steps = S3.curve.map(c => c.step);
      const nb = 24;
      const X = i => bx + steps[i] / steps[steps.length - 1] * bw;
      const Y = v => by + bh - clamp01((Math.log10(v) + 4) / 4.8) * bh;
      const growFrac = ease(ph(local, seg[2].start - T.s3.start + 1.2, seg[2].dur - 1));
      for (let b = 2; b < nb - 1; b++) {
        const lamB = -9 + (b + 0.5) * 19 / nb;
        const ys = S3.curve.map(c => c.x0_per_bin[b]).map(v => v == null ? null : v);
        const pts = [];
        for (let i = 0; i < steps.length; i++) if (ys[i] != null && isFinite(ys[i])) pts.push([X(i), Y(Math.max(1e-4, ys[i]))]);
        polyline(ctx, pts, { color: lamColor(Math.max(-9, Math.min(10, lamB))), width: 3.5, alpha: pL * 0.95, frac: growFrac });
      }
      text(ctx, '紫 = 高噪桶:立刻贴地板(那是数据的条件方差,不是没学好)', bx + 24, by + bh - 96, { size: 26, color: '#9DA0EE', alpha: pL * ease(ph(local, seg[2].start - T.s3.start + 3.5, 1.5)) });
      text(ctx, '金 = 低噪桶:量级天生小', bx + 24, by + bh - 56, { size: 26, color: C.edm, alpha: pL * ease(ph(local, seg[2].start - T.s3.start + 5, 1.5)) });
      text(ctx, '中段桶:整个训练期持续下降 → 质量的进步全在这里', bx + 24, by + 60, { size: 28, color: C.ink, weight: 'bold', alpha: pL * ease(ph(local, seg[2].start - T.s3.start + 7, 1.5)) });
      text(ctx, '训练步数 →', bx + bw / 2, by + bh + 44, { size: 24, align: 'center', color: C.ink3, font: mono, alpha: pL });
    }
  }

  // ============================================================ S4 no-network generation
  const S4 = (() => {
    // unpack mnist 4-bit
    const M = window.HF_MNIST;
    const raw = atob(M.b64);
    const n = M.n, px = 784;
    const X = new Float32Array(n * px);
    for (let i = 0; i < raw.length; i++) {
      const byte = raw.charCodeAt(i);
      X[i * 2] = (byte >> 4) / 15;
      X[i * 2 + 1] = (byte & 15) / 15;
    }
    // empirical-Bayes denoiser in VE frame: E[x0|xt] = softmax(-d2/(2s^2)) . X
    function denoiseVE(xt, sig) {
      const w = new Float64Array(n);
      let mx = -1e30;
      for (let i = 0; i < n; i++) {
        let d2 = 0;
        for (let j = 0; j < px; j++) { const d = xt[j] - X[i * px + j]; d2 += d * d; }
        const l = -d2 / (2 * sig * sig);
        w[i] = l; if (l > mx) mx = l;
      }
      let Z = 0;
      for (let i = 0; i < n; i++) { w[i] = Math.exp(w[i] - mx); Z += w[i]; }
      const out = new Float32Array(px);
      let top = 0, topi = 0;
      for (let i = 0; i < n; i++) {
        const r = w[i] / Z;
        if (r > top) { top = r; topi = i; }
        for (let j = 0; j < px; j++) out[j] += r * X[i * px + j];
      }
      return { out, top, topi };
    }
    // DDIM (eta=0) in VE frame on karras grid, record x0h per step
    const sig = HF.karrasSigmas(22, 0.03, 6.0);
    function run(seed) {
      const rng = HF.mulberry32(seed);
      let x = new Float32Array(px);
      for (let j = 0; j < px; j++) x[j] = sig[0] * HF.gauss(rng);
      const recs = [];
      for (let k = 0; k < sig.length - 1; k++) {
        const s = sig[k], sn = sig[k + 1];
        const { out: x0h, top, topi } = denoiseVE(x, s);
        recs.push({ x0h, top, topi, sig: s });
        const nx = new Float32Array(px);
        for (let j = 0; j < px; j++) { const eps = (x[j] - x0h[j]) / s; nx[j] = x0h[j] + sn * eps; }
        x = nx;
      }
      // nearest neighbor of final
      let best = 1e30, bi = 0;
      for (let i = 0; i < n; i++) {
        let d2 = 0;
        for (let j = 0; j < px; j++) { const d = x[j] - X[i * px + j]; d2 += d * d; }
        if (d2 < best) { best = d2; bi = i; }
      }
      return { recs, final: x, nnIdx: bi, nnL2: Math.sqrt(best / px) };
    }
    const runs = [run(101), run(202), run(303)];
    return { X, n, px, runs, denoiseVE };
  })();
  // offscreen canvas helper to blit 28x28 grayscale
  const _tile = document.createElement('canvas'); _tile.width = 28; _tile.height = 28;
  const _tctx = _tile.getContext('2d');
  function blit28(ctx, arr, x, y, size, alpha = 1, tint = null) {
    const img = _tctx.createImageData(28, 28);
    for (let j = 0; j < 784; j++) {
      const v = Math.max(0, Math.min(255, arr[j] * 255));
      img.data[j * 4] = tint ? v * tint[0] : v;
      img.data[j * 4 + 1] = tint ? v * tint[1] : v;
      img.data[j * 4 + 2] = tint ? v * tint[2] : v;
      img.data[j * 4 + 3] = 255;
    }
    _tctx.putImageData(img, 0, 0);
    ctx.save(); ctx.globalAlpha = alpha; ctx.imageSmoothingEnabled = false;
    ctx.drawImage(_tile, x, y, size, size); ctx.restore();
  }
  function drawS4(ctx, t, T) {
    const local = t - T.s4.start, seg = T.s4.segs;
    const fadeOld = 1 - ease(ph(local, seg[3].start - T.s4.start + 0.2, 0.9));
    const p0 = ease(ph(local, 0.5, 1.5)) * Math.max(0.0, fadeOld);
    // left: gallery of training digits (16x8 thumbnails)
    ctx.save(); ctx.globalAlpha = 0.55 * p0;
    for (let i = 0; i < 96; i++) {
      const r = (i / 12) | 0, c2 = i % 12;
      blit28(ctx, S4.X.subarray(i * 784, i * 784 + 784), 130 + c2 * 46, 170 + r * 46, 42, 0.5);
    }
    ctx.restore();
    text(ctx, '400 张真 MNIST(全部"参数")', 130 + 6 * 46, 560, { size: 26, align: 'center', color: C.ink3, alpha: p0 });
    // center: crystallization — x0h over steps for run 0
    const cryStart = seg[2].start - T.s4.start;
    const pc = ph(local, 1.0, (seg[2].start - T.s4.start) + seg[2].dur - 2.0);
    const run = S4.runs[0];
    const fi = clamp01(pc) * (run.recs.length - 1);
    const i0 = Math.min(run.recs.length - 1, Math.floor(fi));
    const rec = run.recs[i0];
    blit28(ctx, rec.x0h, 810, 200, 300, p0);
    text(ctx, `x̂₀ = 加权平均(σ = ${rec.sig.toFixed(2)})`, 960, 540, { size: 28, align: 'center', color: C.ink2, font: mono, alpha: p0 });
    text(ctx, `最大权重 ${(rec.top * 100).toFixed(0)}%`, 960, 580, { size: 26, align: 'center', color: C.ink3, font: mono, alpha: p0 });
    // right: current x_t? show weights bar of top responsibilities
    const pW = ease(ph(local, seg[1].start - T.s4.start, 2));
    if (pW > 0) {
      text(ctx, 'softmax 权重(前 12 名)', 1420, 190, { size: 26, color: C.ink2, alpha: pW });
      // top-12 by weight: recompute cheap—use rec.top only? draw responsibilities heat of first 12 via denoise? Precomputed per rec? keep simple: show top weight bar growing
      const barw = 300 * rec.top;
      ctx.save(); ctx.globalAlpha = pW; ctx.fillStyle = C.edm; ctx.fillRect(1420, 220, Math.max(6, barw), 26); ctx.restore();
      text(ctx, '高噪:权重摊开(平均一大片)→ 低噪:锁死一张', 1420, 286, { size: 24, color: C.ink3, alpha: pW });
      blit28(ctx, S4.X.subarray(rec.topi * 784, rec.topi * 784 + 784), 1420, 330, 180, pW * 0.9);
      text(ctx, '当前权重最大的训练图', 1510, 546, { size: 24, align: 'center', color: C.ink3, alpha: pW });
    }
    // verdict strip: three finals with NN pairs (s4d..s4e)
    const pV = ease(ph(local, seg[3].start - T.s4.start + 0.5, 1.4));
    if (pV > 0) {
      panel(ctx, 220, 250, 1480, 470, { alpha: pV });
      text(ctx, '对质:生成图(左) vs 最近的训练图(右)', 260, 306, { size: 32, color: C.ink, alpha: pV });
      S4.runs.forEach((r, i) => {
        const gx = 300 + i * 480;
        blit28(ctx, r.final, gx, 350, 205, pV);
        blit28(ctx, S4.X.subarray(r.nnIdx * 784, r.nnIdx * 784 + 784), gx + 225, 350, 205, pV);
        text(ctx, `逐像素 L2 = ${r.nnL2.toFixed(3)}`, gx + 215, 610, { size: 27, align: 'center', color: C.bad, font: mono, alpha: pV });
      });
      text(ctx, '几乎逐像素重合 —— 复读机', 960, 676, { size: 30, align: 'center', color: C.bad, alpha: pV });
      const pF = ease(ph(local, seg[4].start - T.s4.start + 0.8, 2));
      text(ctx, '完美目标 + 完美优化 = 完美复读机', 960, 790, { size: 40, align: 'center', color: C.ink, weight: 'bold', alpha: pF });
      text(ctx, '创造力 = 网络对这张查表的背叛(泛化 = 归纳偏置的礼物)', 960, 856, { size: 32, align: 'center', color: C.edm, alpha: pF });
    }
  }

  window.HF_SCENES_CORE = { C, MODE_COLORS, lamColor, text, mono, panel, spectrumStrip, axisBox, polyline, arrow, blit28, clamp01, ease, lerp, ph, gmm, models, denA, denU, W, H, drawS0, drawS1, drawS2, drawS3, drawS4 };
})();
