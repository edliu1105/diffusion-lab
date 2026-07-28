// scenes_late.js — scenes S5–S9. Pure draw(t); all stochastic content precomputed with seeds.
(function () {
  const K = window.HF_SCENES_CORE;
  const { C, MODE_COLORS, lamColor, text, mono, panel, spectrumStrip, axisBox, polyline, arrow,
    clamp01, ease, lerp, ph, gmm, models, denA, W, H } = K;
  const HF = window.HF, D = window.HF_DATA;

  // ============================================================ S5 sampling = integration
  const S5 = (() => {
    // DDIM vs FM-Euler race, same seeds, analytic denoiser (exact field)
    const N = 160, STEPS = 40;
    const rng = HF.mulberry32(4242);
    const x0s = new Float32Array(N * 2);
    for (let i = 0; i < N * 2; i++) x0s[i] = HF.gauss(rng);
    // record trajectories in VP display coords
    function recordDDIM(steps) {
      const s = HF.ddim(denA, Float32Array.from(x0s), N, steps, { eta: 0, lamMin: -8.5, lamMax: 9 });
      const traj = [Float32Array.from(s.xs)];
      while (!s.done) { s.step(); traj.push(Float32Array.from(s.xs)); }
      return traj;
    }
    function recordFME(steps) {
      const s = HF.fmEuler(denA, Float32Array.from(x0s), N, steps, { tStart: 0.9858350, tEnd: 0.0110090 });
      const lams = [-8.5];
      const traj = [Float32Array.from(s.xs)];
      while (!s.done) { const r = s.step(); traj.push(Float32Array.from(s.xs)); lams.push(r.lam); }
      return { traj, lams };
    }
    const ddimT = recordDDIM(STEPS);
    const fme = recordFME(STEPS);
    // gap vs N (endpoint median distance), few Ns
    const gaps = [8, 32, 128].map(n => {
      const a = HF.ddim(denA, Float32Array.from(x0s), N, n, { eta: 0, lamMin: -8.5, lamMax: 9 });
      HF.runToEnd(a);
      const b = HF.fmEuler(denA, Float32Array.from(x0s), N, n, { tStart: 0.9858350, tEnd: 0.0110090 });
      HF.runToEnd(b);
      const bv = new Float32Array(N * 2);
      for (let i = 0; i < N * 2; i++) bv[i] = HF.fmToVP(b.xs[i], 9);
      return { n, gap: HF.endpointGap(a.xs, bv, N) };
    });
    return { N, STEPS, ddimT, fme, gaps };
  })();
  function drawS5(ctx, t, T) {
    const local = t - T.s5.start, seg = T.s5.segs;
    const p0 = ease(ph(local, 0.4, 1.4));
    // twin panes race (starts with s5b)
    const raceStart = seg[1].start - T.s5.start + 0.5;
    const pr = ph(local, raceStart, seg[1].dur - 1.5);
    const panes = [[560, 'DDIM(VP 坐标)', C.ddim, S5.ddimT, 'vp'], [1360, 'FM-Euler(直线坐标)', C.fm, S5.fme.traj, 'fm']];
    const cy = 430, sc = 118;
    for (const [cx, label, col, traj, frame] of panes) {
      // data dots
      ctx.save();
      for (let i = 0; i < 800; i += 2) {
        const p = D.data_points[i];
        ctx.globalAlpha = 0.22 * p0; ctx.fillStyle = '#39414E';
        ctx.beginPath(); ctx.arc(cx + p[0] * sc / 1.6, cy - p[1] * sc / 1.6, 2.5, 0, 7); ctx.fill();
      }
      ctx.restore();
      text(ctx, label, cx, cy - 320, { size: 30, align: 'center', color: col, alpha: p0 });
      const fi = clamp01(pr) * (traj.length - 1);
      const i0 = Math.min(traj.length - 2, Math.floor(fi)), f = fi - i0;
      const lam = frame === 'fm' ? lerp(S5.fme.lams[i0], S5.fme.lams[Math.min(i0 + 1, S5.fme.lams.length - 1)], f) : lerp(-8.5, 9, fi / (traj.length - 1));
      ctx.save();
      for (let i = 0; i < S5.N; i++) {
        let x = lerp(traj[i0][i * 2], traj[i0 + 1][i * 2], f);
        let y = lerp(traj[i0][i * 2 + 1], traj[i0 + 1][i * 2 + 1], f);
        if (frame === 'fm') { x = HF.fmToVP(x, Math.min(9, lam)); y = HF.fmToVP(y, Math.min(9, lam)); }
        ctx.globalAlpha = 0.9 * p0; ctx.fillStyle = lamColor(Math.max(-9, Math.min(9, lam)));
        ctx.beginPath(); ctx.arc(cx + x * sc / 1.6, cy - y * sc / 1.6, 4, 0, 7); ctx.fill();
      }
      ctx.restore();
    }
    // gap readout
    const pG = ease(ph(local, seg[1].start - T.s5.start + seg[1].dur * 0.55, 2)) *
               (1 - ease(ph(local, seg[2].start - T.s5.start + 0.2, 0.8)));
    if (pG > 0.01) {
      panel(ctx, 610, 780, 700, 150, { alpha: pG });
      text(ctx, '同种子端点中位距离(→ 同一条曲线)', 960, 822, { size: 26, align: 'center', color: C.ink2, alpha: pG });
      const parts = S5.gaps.map(g => `${g.n}步: ${g.gap.toFixed(3)}`).join('    ');
      text(ctx, parts, 960, 880, { size: 30, align: 'center', color: C.ink, font: mono, alpha: pG });
    }
    // knee chart on s5c/s5d
    const pK = ease(ph(local, seg[2].start - T.s5.start + 0.4, 2));
    if (pK > 0) {
      const bx = 330, by = 660, bw = 1260, bh = 330;
      panel(ctx, bx - 30, by - 50, bw + 60, bh + 120, { alpha: pK });
      axisBox(ctx, bx, by, bw, bh, pK);
      text(ctx, '质量-NFE:膝盖在十几步;之后是模型误差的天下', bx, by - 20, { size: 28, color: C.ink2, alpha: pK });
      // schematic-but-shaped knee: err = disc(N) + floor ; disc ~ c/N
      const pts = [];
      for (let i = 0; i <= 100; i++) {
        const n = 1 + i * 0.63; // 1..64
        const err = 1.6 / n + 0.12;
        pts.push([bx + Math.log2(n) / 6 * bw, by + bh - clamp01((Math.log10(err) + 1.1) / 1.6) * bh]);
      }
      polyline(ctx, pts, { color: C.ink, width: 5, alpha: pK, frac: pK });
      ctx.save(); ctx.globalAlpha = pK * 0.8; ctx.strokeStyle = C.bad; ctx.setLineDash([8, 8]); ctx.lineWidth = 3;
      const fy = by + bh - clamp01((Math.log10(0.12) + 1.1) / 1.6) * bh;
      ctx.beginPath(); ctx.moveTo(bx, fy); ctx.lineTo(bx + bw, fy); ctx.stroke(); ctx.restore();
      text(ctx, '模型误差地板(积分再准也穿不过去)', bx + bw - 30, fy - 26, { size: 26, align: 'right', color: C.bad, alpha: pK });
      text(ctx, 'NFE(log)→', bx + bw / 2, by + bh + 42, { size: 24, align: 'center', color: C.ink3, font: mono, alpha: pK });
      const marks = [[1, '1'], [4, '4'], [16, '16'], [64, '64']];
      for (const [n, s] of marks) text(ctx, s, bx + Math.log2(n) / 6 * bw, by + bh + 20, { size: 22, align: 'center', color: C.ink3, font: mono, alpha: pK });
    }
  }

  // ============================================================ S6 CFG geometry
  const S6 = (() => {
    const N = 400, cls = 0;
    function guided(w) {
      return (x, lam, n) => {
        const dc = gmm.denoiseVP(x, lam, n, cls);
        if (w === 1) return dc;
        const du = gmm.denoiseVP(x, lam, n);
        const out = new Float32Array(n * 2);
        for (let i = 0; i < n * 2; i++) out[i] = du[i] + w * (dc[i] - du[i]);
        return out;
      };
    }
    function cloud(w, seed) {
      const rng = HF.mulberry32(seed);
      const x1 = new Float32Array(N * 2);
      for (let i = 0; i < N * 2; i++) x1[i] = HF.gauss(rng);
      const s = HF.fmEuler(guided(w), x1, N, 24);
      HF.runToEnd(s);
      const out = new Float32Array(N * 2);
      let mx = 0, my = 0;
      for (let i = 0; i < N; i++) {
        out[i * 2] = HF.fmToVP(s.xs[i * 2], 9); out[i * 2 + 1] = HF.fmToVP(s.xs[i * 2 + 1], 9);
        mx += out[i * 2] / N; my += out[i * 2 + 1] / N;
      }
      let rms = 0;
      const mu = D.gmm.mu[cls];
      for (let i = 0; i < N; i++) rms += ((out[i * 2] - mu[0]) ** 2 + (out[i * 2 + 1] - mu[1]) ** 2) / N;
      return { pts: out, mean: [mx, my], rms: Math.sqrt(rms) };
    }
    const w1 = cloud(1, 97531), w3 = cloud(3, 97531);
    const trueRms = D.gmm.std[cls] * Math.SQRT2;
    return { w1, w3, cls, trueRms };
  })();
  function drawS6(ctx, t, T) {
    const local = t - T.s6.start, seg = T.s6.segs;
    const p0 = ease(ph(local, 0.4, 1.4));
    const cx = 700, cy = 480, sc = 210;
    const mu = D.gmm.mu[S6.cls];
    // data scatter + target ring
    ctx.save();
    for (let i = 0; i < 1200; i += 2) {
      const p = D.data_points[i];
      ctx.globalAlpha = 0.2 * p0; ctx.fillStyle = '#39414E';
      ctx.beginPath(); ctx.arc(cx + p[0] * sc / 1.9, cy - p[1] * sc / 1.9, 2.6, 0, 7); ctx.fill();
    }
    ctx.restore();
    ctx.save(); ctx.globalAlpha = p0; ctx.strokeStyle = MODE_COLORS[S6.cls]; ctx.setLineDash([7, 7]); ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(cx + mu[0] * sc / 1.9, cy - mu[1] * sc / 1.9, D.gmm.std[S6.cls] * 2 * sc / 1.9, 0, 7); ctx.stroke(); ctx.restore();
    text(ctx, '目标类的真实 2σ 范围', cx + mu[0] * sc / 1.9, cy - mu[1] * sc / 1.9 - D.gmm.std[S6.cls] * 2 * sc / 1.9 - 26, { size: 24, align: 'center', color: MODE_COLORS[S6.cls], alpha: p0 });
    // clouds: w=1 then morph to w=3 (during s6c)
    const pM = ease(ph(local, seg[2].start - T.s6.start + 0.5, 2.5));
    ctx.save();
    for (let i = 0; i < 400; i++) {
      const x = lerp(S6.w1.pts[i * 2], S6.w3.pts[i * 2], pM);
      const y = lerp(S6.w1.pts[i * 2 + 1], S6.w3.pts[i * 2 + 1], pM);
      ctx.globalAlpha = 0.85 * p0; ctx.fillStyle = C.fm;
      ctx.beginPath(); ctx.arc(cx + x * sc / 1.9, cy - y * sc / 1.9, 4.5, 0, 7); ctx.fill();
    }
    ctx.restore();
    // centroid arrow (outward push)
    if (pM > 0.15) {
      arrow(ctx, cx + S6.w1.mean[0] * sc / 1.9, cy - S6.w1.mean[1] * sc / 1.9,
        cx + lerp(S6.w1.mean[0], S6.w3.mean[0], pM) * sc / 1.9, cy - lerp(S6.w1.mean[1], S6.w3.mean[1], pM) * sc / 1.9,
        { color: C.bad, width: 7, alpha: clamp01(pM * 1.5) });
      text(ctx, '质心被推过数据,向外', cx + S6.w3.mean[0] * sc / 1.9 + 30, cy - S6.w3.mean[1] * sc / 1.9 - 26, { size: 28, color: C.bad, alpha: clamp01(pM * 1.4) });
    }
    // readout card
    panel(ctx, 1180, 260, 580, 420, { alpha: p0 });
    text(ctx, '零模型误差 · 实测', 1210, 308, { size: 28, color: C.ink2, alpha: p0 });
    const wNow = lerp(1, 3, pM);
    text(ctx, `w = ${wNow.toFixed(1)}`, 1210, 380, { size: 44, color: C.ink, font: mono, alpha: p0 });
    const rmsNow = lerp(S6.w1.rms, S6.w3.rms, pM);
    text(ctx, `类内 rms / 真条件`, 1210, 456, { size: 26, color: C.ink2, alpha: p0 });
    text(ctx, `${(rmsNow / S6.trueRms * 100).toFixed(0)}%`, 1210, 516, { size: 56, color: pM > 0.4 ? C.bad : C.ink, font: mono, alpha: p0, weight: 'bold' });
    text(ctx, pM > 0.4 ? '不缩,反涨 —— 外推' : '诚实条件模型', 1210, 590, { size: 28, color: pM > 0.4 ? C.bad : C.ok, alpha: p0 });
    // fixes strip (s6d)
    const pF = ease(ph(local, seg[3].start - T.s6.start + 0.3, 1.6));
    if (pF > 0) {
      panel(ctx, 240, 780, 1440, 210, { alpha: pF });
      const fixes = [['interval CFG', '只在决策带开引导'], ['rescale', '把统计量拉回来'], ['autoguidance', '用更差的自己当无条件'], ['guidance 蒸馏', '把 w 焙进权重']];
      fixes.forEach(([nm, de], i) => {
        const fx = 320 + i * 350;
        text(ctx, nm, fx, 850, { size: 30, color: C.edm, font: mono, alpha: pF });
        text(ctx, de, fx, 905, { size: 26, color: C.ink2, alpha: pF });
      });
    }
  }

  // ============================================================ S7 distillation
  const S7 = (() => {
    // teacher vs reflow trajectories (real nets), 46 seeds
    const N = 24, STEPS = 40;
    const rng = HF.mulberry32(2718);
    const x1 = new Float32Array(N * 2);
    for (let i = 0; i < N * 2; i++) x1[i] = HF.gauss(rng);
    function record(model) {
      const den = (x, lam, n) => model.denoiseVP(x, lam, n);
      const s = HF.fmEuler(den, Float32Array.from(x1), N, STEPS);
      const lams = [-8.4]; const traj = [Float32Array.from(s.xs)];
      while (!s.done) { const r = s.step(); traj.push(Float32Array.from(s.xs)); lams.push(Math.min(9, r.lam)); }
      void lams;
      return traj;  // native FM coordinates: straightness is visible as-is
    }
    const teacher = record(models.u_fm);
    const reflow = record(models.u_reflow);
    // naive one-step: x0 = denoise at lam_start (posterior mean) => all collapse to ~global mean
    const lamsA = new Float32Array(N).fill(-8.4);
    const naive = denA(Float32Array.from(x1), lamsA, N);
    // real mean-velocity student endpoints (1 jump)
    const lam1 = 2 * Math.log((1 - 0.985) / 0.985);
    const lams1 = new Float32Array(N).fill(lam1);
    const uu = models.u_1step.forward(Float32Array.from(x1), lams1, N);
    const student = new Float32Array(N * 2);
    for (let i = 0; i < N * 2; i++) student[i] = x1[i] - (0.985 - 0.002) * uu[i];
    return { N, teacher, reflow, x1, naive, student };
  })();
  function drawS7(ctx, t, T) {
    const local = t - T.s7.start, seg = T.s7.segs;
    const p0 = ease(ph(local, 0.4, 1.2));
    const cx = 640, cy = 470, sc = 128;
    // base data
    ctx.save();
    for (let i = 0; i < 1000; i += 2) {
      const p = D.data_points[i];
      ctx.globalAlpha = 0.2 * p0; ctx.fillStyle = '#39414E';
      ctx.beginPath(); ctx.arc(cx + p[0] * sc / 1.6, cy - p[1] * sc / 1.6, 2.6, 0, 7); ctx.fill();
    }
    ctx.restore();
    // phase: naive collapse (s7b)
    const pN = ph(local, seg[1].start - T.s7.start + 0.4, seg[1].dur - 1);
    if (local < seg[2].start - T.s7.start) {
      text(ctx, '天真一步:x̂₀ 是均值,不是样本', cx, 150, { size: 32, align: 'center', color: C.ink, alpha: p0 });
      const pp = ease(clamp01(pN));
      ctx.save();
      for (let i = 0; i < S7.N; i++) {
        const x0v = HF.fmToVP(S7.x1[i * 2], -8.4), y0v = HF.fmToVP(S7.x1[i * 2 + 1], -8.4);
        const x = lerp(x0v, S7.naive[i * 2], pp), y = lerp(y0v, S7.naive[i * 2 + 1], pp);
        ctx.globalAlpha = 0.9; ctx.fillStyle = pp > 0.8 ? C.bad : lamColor(lerp(-8.4, 9, pp));
        ctx.beginPath(); ctx.arc(cx + x * sc / 1.6, cy - y * sc / 1.6, 5, 0, 7); ctx.fill();
      }
      ctx.restore();
      if (pp > 0.75) text(ctx, '全部塌向均值 → 糊', cx, cy + 300, { size: 30, align: 'center', color: C.bad, alpha: (pp - 0.75) * 4 });
    } else {
      // trajectories: teacher (curved) vs reflow (straight) — s7c
      const tc0 = seg[2].start - T.s7.start;
      const pT = ease(ph(local, tc0 + 0.2, 1.2));
      const fracT = ease(ph(local, tc0 + 0.4, 2.4));
      const fracR = ease(ph(local, tc0 + 3.2, 2.4));
      const dimT = 1 - 0.6 * ease(ph(local, tc0 + 3.2, 1.2));
      ctx.save(); ctx.globalAlpha = pT; ctx.fillStyle = C.fm; ctx.fillRect(cx - 380, 148, 22, 22); ctx.restore();
      text(ctx, '教师 u_fm:弯路', cx - 344, 159, { size: 28, color: C.fm, alpha: pT });
      if (fracR > 0) {
        ctx.save(); ctx.globalAlpha = pT; ctx.fillStyle = C.student; ctx.fillRect(cx + 60, 148, 22, 22); ctx.restore();
        text(ctx, 'ReFlow:直路', cx + 96, 159, { size: 28, color: C.student, alpha: clamp01(fracR * 2) });
      }
      for (let i = 0; i < S7.N; i++) {
        const tp = S7.teacher.map(xs => [cx + xs[i * 2] * sc / 1.6, cy - xs[i * 2 + 1] * sc / 1.6]);
        polyline(ctx, tp, { color: C.fm, width: 2.2, alpha: 0.5 * pT * dimT, frac: fracT });
      }
      if (fracR > 0) for (let i = 0; i < S7.N; i++) {
        const rp = S7.reflow.map(xs => [cx + xs[i * 2] * sc / 1.6, cy - xs[i * 2 + 1] * sc / 1.6]);
        polyline(ctx, rp, { color: C.student, width: 2.4, alpha: 0.65 * pT, frac: fracR });
      }
      const p132 = ease(ph(local, seg[2].start - T.s7.start + seg[2].dur * 0.62, 1.5));
      if (p132 > 0) {
        panel(ctx, 350, 830, 580, 130, { alpha: p132 });
        text(ctx, '2 步端点漂移:0.714 → 0.005', 640, 878, { size: 30, align: 'center', color: C.ink, font: mono, alpha: p132 });
        text(ctx, '132× —— 配对拆掉了条件方差', 640, 924, { size: 27, align: 'center', color: C.student, alpha: p132 });
      }
    }
    // right rail: routes table (s7a on) + costs (s7d)
    const pR = ease(ph(local, 0.8, 1.6));
    panel(ctx, 1180, 200, 600, 480, { alpha: pR });
    text(ctx, '四条路线 = 给终点映射列方程', 1210, 248, { size: 27, color: C.ink2, alpha: pR });
    const routes = [
      ['轨迹回归', 'PD 一脉;如今多当初始化', C.ink],
      ['分布匹配', 'DMD / ADD → schnell、Turbo', C.edm],
      ['一致性 / flow map', 'sCM → rCM(14B 视频)', C.ddim],
      ['平均速度', 'MeanFlow;从头一步训练', C.student],
    ];
    routes.forEach(([nm, de, col], i) => {
      const yy = 310 + i * 90;
      text(ctx, nm, 1210, yy, { size: 29, color: col, alpha: pR });
      text(ctx, de, 1210, yy + 38, { size: 23, color: C.ink3, font: mono, alpha: pR });
    });
    const pC = ease(ph(local, seg[3].start - T.s7.start + 0.3, 1.6));
    if (pC > 0) {
      panel(ctx, 1180, 710, 600, 280, { alpha: pC });
      text(ctx, '少步模型共同签的三笔账', 1210, 758, { size: 27, color: C.ink2, alpha: pC });
      ['CFG 已蒸进权重,w 不可调', '往返轨迹没了 → 反演编辑失灵', '多样性地板:fidelity 与 coverage 都要验'].forEach((s, i) => {
        text(ctx, '· ' + s, 1210, 812 + i * 52, { size: 25, color: C.ink, alpha: pC });
      });
    }
  }

  // ============================================================ S8 time & world models
  const DR = window.HF_DRIFT;
  const DR_SM = (() => {
    const win = 15, out = {};
    for (const nm of ['A', 'B', 'C']) {
      const d = DR[nm].radial.drift, sm = [];
      for (let i = 0; i < d.length; i++) {
        let a = 0, c = 0;
        for (let j = Math.max(0, i - win); j <= i; j++) { a += d[j]; c++; }
        sm.push(a / c);
      }
      out[nm] = sm;
    }
    return out;
  })();
  function drawS8(ctx, t, T) {
    const local = t - T.s8.start, seg = T.s8.segs;
    const p0 = ease(ph(local, 0.4, 1.2));
    if (local < seg[2].start - T.s8.start) {
      // Diffusion Forcing frame strips
      const rows = [
        ['a. 离线整段:全帧同一噪声级,一起去噪', [0, 0, 0, 0, 0, 0, 0, 0], 0],
        ['b. Diffusion Forcing:每帧独立噪声级', [9, 7, 5, 2.5, 0, -3, -6, -8.5], seg[1].start - T.s8.start],
        ['c. 流式:干净历史(KV)+ 新帧少步出图', [9, 9, 9, 9, 9, 2, -4, -8.5], seg[1].start - T.s8.start + 3.5],
      ];
      rows.forEach(([label, lams, t0], r) => {
        const pR = ease(ph(local, (r === 0 ? 0.6 : t0), 1.6));
        if (pR <= 0) return;
        const yy = 260 + r * 220;
        text(ctx, label, 330, yy - 46, { size: 30, color: C.ink, alpha: pR });
        lams.forEach((lm, i) => {
          ctx.save(); ctx.globalAlpha = pR;
          ctx.fillStyle = lamColor(lm); ctx.beginPath();
          ctx.roundRect(330 + i * 165, yy, 140, 110, 10); ctx.fill();
          if (r === 2 && i < 5) { ctx.strokeStyle = C.ok; ctx.lineWidth = 3; ctx.stroke(); }
          ctx.restore();
          text(ctx, lm >= 8.9 ? '干净' : lm <= -8 ? '纯噪' : `λ=${lm}`, 400 + i * 165, yy + 55, { size: 22, align: 'center', color: 'rgba(12,15,20,0.85)', font: mono, alpha: pR });
        });
      });
      text(ctx, '一个改动买三件事:条件生成 · 容忍瑕疵 · 滑窗无限续写', 960, 960, { size: 30, align: 'center', color: C.ink2, alpha: ease(ph(local, seg[1].start - T.s8.start + 5, 2)) });
    } else {
      // drift experiment: circle + real baked rollouts
      const pD = ease(ph(local, seg[2].start - T.s8.start + 0.2, 1.4));
      const cx = 620, cy = 540, sc2 = 88;
      ctx.save(); ctx.globalAlpha = pD; ctx.strokeStyle = 'rgba(154,163,178,0.5)'; ctx.setLineDash([8, 8]); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(cx, cy, DR.R * sc2, 0, 7); ctx.stroke(); ctx.restore();
      text(ctx, '数据流形(圆轨道)· 每步注入 0.02 径向误差', cx, 190, { size: 30, align: 'center', color: C.ink2, alpha: pD });
      const frac = ph(local, seg[2].start - T.s8.start + 0.8, seg[2].dur + seg[3].dur - 3);
      const runs = [['A', '干净上下文训练', C.bad], ['B', '加噪上下文(疫苗)', C.ok], ['C', '自回滚重标注', C.edm]];
      const nF = DR.A.radial.pts.length;
      const upto = Math.max(2, Math.floor(clamp01(frac) * nF));
      for (const [nm, label, col] of runs) {
        const pts = DR[nm].radial.pts.slice(0, upto).map(p => [cx + p[0] * sc2, cy - p[1] * sc2]);
        polyline(ctx, pts, { color: col, width: 4, alpha: 0.9 * pD });
        const last = pts[pts.length - 1];
        ctx.save(); ctx.globalAlpha = pD; ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(last[0], last[1], 9, 0, 7); ctx.fill(); ctx.restore();
      }
      // drift curves panel
      const bx = 1150, by = 300, bw = 620, bh = 380;
      panel(ctx, bx - 30, by - 60, bw + 60, bh + 150, { alpha: pD });
      axisBox(ctx, bx, by, bw, bh, pD);
      text(ctx, '离轨距离(15 帧滑动平均)', bx, by - 24, { size: 28, color: C.ink2, alpha: pD });
      const maxD = 0.4;
      for (const [nm, label, col] of runs) {
        const dr = DR_SM[nm];
        const pts = [];
        for (let i = 0; i < upto; i++) pts.push([bx + i / (nF - 1) * bw, by + bh - clamp01(dr[i] / maxD) * bh]);
        polyline(ctx, pts, { color: col, width: 4.5, alpha: pD });
      }
      const labels = [['A:累积到注入量的 ~15×,锁进坏轨道', C.bad, DR_SM.A[nF - 1]], ['C:居中', C.edm, DR_SM.C[nF - 1] + 0.03], ['B:钉死在单步注入量', C.ok, DR_SM.B[nF - 1]]];
      labels.forEach(([s2, col, dv], i) => {
        text(ctx, s2, bx + bw - 10, by + bh - clamp01(dv / maxD) * bh - 20, { size: 24, align: 'right', color: col, alpha: pD * ease(ph(local, seg[3].start - T.s8.start + 0.5 + i * 0.6, 1)) });
      });
      text(ctx, '帧 →', bx + bw / 2, by + bh + 40, { size: 24, align: 'center', color: C.ink3, font: mono, alpha: pD });
      const pM = ease(ph(local, seg[3].start - T.s8.start + 2.5, 2));
      text(ctx, '收缩映射:误差不获准过夜', bx + bw / 2, by + bh + 100, { size: 30, align: 'center', color: C.ok, alpha: pM, weight: 'bold' });
      // world model status (s8e)
      const pW = ease(ph(local, seg[4].start - T.s8.start + 0.3, 1.6));
      if (pW > 0) {
        panel(ctx, 240, 900, 1440, 120, { alpha: pW });
        text(ctx, '世界模型 = 视频先验 + 因果接口 + 记忆    瓶颈:长程记忆 · 误差漂移 · 每帧算力', 960, 960, { size: 32, align: 'center', color: C.ink, alpha: pW });
      }
    }
  }

  // ============================================================ S9 finale
  function drawS9(ctx, t, T) {
    const local = t - T.s9.start, seg = T.s9.segs;
    const cards = [
      ['M1', '一族分布,多套坐标', 'formulation 之争 = 坐标之争'],
      ['M2', 'λ 轴是唯一的横轴', '一切预算都花在它上面'],
      ['M3', '输出是均值,不是样本', '糊不是 bug,是定义'],
      ['M4', '采样 = 数值积分', '随机性是另开的预算'],
      ['M5', '最优解是复读机', '创造力 = 对最优的背叛'],
    ];
    cards.forEach(([id, tt, de], i) => {
      const p = ease(ph(local, 0.6 + i * (seg[0].dur - 2) / 5, 1.2));
      if (p <= 0) return;
      const xx = 200 + (i % 3) * 520, yy = 240 + Math.floor(i / 3) * 300;
      panel(ctx, xx, yy, 460, 230, { alpha: p });
      text(ctx, id, xx + 36, yy + 60, { size: 34, color: C.edm, font: mono, alpha: p, weight: 'bold' });
      text(ctx, tt, xx + 36, yy + 120, { size: 33, color: C.ink, alpha: p, weight: 'bold' });
      text(ctx, de, xx + 36, yy + 172, { size: 25, color: C.ink2, alpha: p });
    });
    const pS = ease(ph(local, seg[1].start - T.s9.start + 0.4, 1.6));
    if (pS > 0) {
      panel(ctx, 1240, 540, 480, 230, { alpha: pS });
      text(ctx, '备忘', 1276, 600, { size: 30, color: C.bad, alpha: pS, weight: 'bold' });
      text(ctx, '产品名 ≠ 技术路线名', 1276, 656, { size: 30, color: C.ink, alpha: pS });
      text(ctx, 'Sora 2 已停服;词汇仍在流通', 1276, 706, { size: 25, color: C.ink2, alpha: pS });
    }
    const pE = ease(ph(local, seg[2].start - T.s9.start + 0.3, 1.6));
    if (pE > 0) {
      text(ctx, '去实验室,亲手拨', 960, 836, { size: 44, align: 'center', color: C.ink, alpha: pE, weight: 'bold' });
      text(ctx, 'edliu1105.github.io/diffusion-lab', 960, 898, { size: 30, align: 'center', color: C.ddim, font: mono, alpha: pE });
    }
  }

  window.HF_SCENES_LATE = { drawS5, drawS6, drawS7, drawS8, drawS9 };
})();
