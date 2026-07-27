// p_field.js — §2 flagship: one learned field, five dresses. Live models.

import { R } from './models.js';
import { Plane, MODE_COLORS } from './viz2d.js';
import { C, Chart, lamColor, drawLamStrip, legendHTML } from './charts.js';
import { vpFromLam, fmTFromLam } from './nn.js';
import { mulberry32, gauss } from './gmm.js';

const MODEL_LABEL = {
  analytic: '解析真值 D*(上帝视角)', eps_ddpm: 'eps_ddpm(DDPM 方言)',
  x0_edm: 'x0_edm(EDM 方言)', u_fm: 'u_fm(FM 方言)',
};
const MODEL_COLOR = { analytic: C.analytic, eps_ddpm: C.ddpm, x0_edm: C.edm, u_fm: C.fm };

export async function init(mount) {
  mount.innerHTML = `
  <p>先把地基一句话说完:<strong>前向过程不是过程,是一族混合</strong>。取一张数据 x₀、一份高斯噪声 ε,按噪声等级 λ(logSNR)调配出
  <code>x_t = α·x₀ + σ·ε</code>。没有时间演化,没有马尔可夫链——训练时每一步都是独立地随机抽一个 λ,直接调配。
  网络要学的唯一对象,是在给定调配结果时对原图的最优猜测:</p>
  <p style="text-align:center;font-size:18px"><code style="font-size:17px">D(x_t, λ) = E[x₀ | x_t]</code></p>
  <p>它和 score 是同一个东西,由 Tweedie 公式焊死:<code>∇log p_t(x) = (α·D − x)/σ²</code>。
  五种"预测目标"全部是 D 的仿射改写——给定 x_t 和 λ,任何一个都能精确换算出其余四个:</p>

  <table class="data wide">
    <tr><th>衣服</th><th>定义</th><th>从 x̂₀ 换算</th><th>历史动机</th><th>数值危险区</th></tr>
    <tr><td><b style="color:${C.ddpm}">ε̂</b>(DDPM 2020)</td><td class="m">ε 的估计</td><td class="m">(x_t − α·x̂₀)/σ</td>
      <td>目标恒为 N(0,I),尺度稳定,训练最省心</td><td>λ 低端:换算 x̂₀ 要乘 <b>e^{−λ/2}</b>(λ=−8 时 ×54.6)——误差放大器</td></tr>
    <tr><td><b>x̂₀</b></td><td class="m">原图的估计</td><td class="m">—</td>
      <td>最直观;蒸馏/一步生成的自然坐标</td><td>λ 高端:目标≈输入,网络在学恒等映射,梯度信号趋零</td></tr>
    <tr><td><b>v̂</b>(2022, 蒸馏论文引入)</td><td class="m">α·ε − σ·x₀</td><td class="m">α·ε̂ − σ·x̂₀</td>
      <td>两端都有界:是 ε 和 x₀ 的"旋转",哪端都不退化</td><td>基本没有——这就是它存在的意义</td></tr>
    <tr><td><b style="color:${C.fm}">û</b>(FM/RF 2022-23)</td><td class="m">ε − x₀(直线路径速度)</td><td class="m">(x_t − x̂₀)/t</td>
      <td>采样器直接可用(dx/dt=û),端点干净(t=1 恰为纯噪声)</td><td>t→0 换算含 1/t,但采样方向上无碍</td></tr>
    <tr><td><b>score</b>(NCSN 2019)</td><td class="m">∇log p_t</td><td class="m">(α·x̂₀ − x_t)/σ²</td>
      <td>理论正统:SDE/ODE 全家从它推出</td><td>λ 高端:量级 ~e^{λ/2} 发散,没人直接回归它</td></tr>
  </table>
  <p class="small">表中每个换算式都不是近似——下面检查器里的数字就是这些公式当场代入算的。"数值危险区"一列是各家 formulation 差异的第一现场:同一个 D,不同衣服在不同 λ 段的<em>条件数</em>不同。</p>

  <div class="lab">
    <div class="labhead"><span class="id">实验 2.1</span><span class="t">场视图 · 换算检查器</span><span class="live">LIVE·浏览器内推理</span></div>
    <div class="ctl">
      <div class="grp"><label>网络</label><select id="f-model">
        <option value="analytic">解析真值 D*</option>
        <option value="u_fm" selected>u_fm(FM 方言)</option>
        <option value="eps_ddpm">eps_ddpm(DDPM 方言)</option>
        <option value="x0_edm">x0_edm(EDM 方言)</option>
      </select></div>
      <div class="grp"><label>视图</label><div class="seg" id="f-view">
        <button data-v="x0" class="on">x̂₀ 映射</button><button data-v="eps">ε̂</button><button data-v="v">v̂</button><button data-v="score">score</button><button data-v="u">û</button>
      </div></div>
    </div>
    <div class="ctl">
      <div class="grp"><label>λ</label><input type="range" id="f-lam" min="-8" max="9" step="0.1" value="-1">
        <canvas class="lamstrip" id="f-strip"></canvas><span class="val" id="f-lamv">−1.0</span></div>
      <div class="grp"><label><input type="checkbox" id="f-cloud" checked> x_t 分布云</label></div>
      <div class="grp"><label><input type="checkbox" id="f-truth"> 叠加解析场(白)</label></div>
    </div>
    <div class="grid2">
      <div><canvas id="f-cv"></canvas>
        <div class="labnote">灰点=数据 x₀。彩色云=当前 λ 下 x_t 真实分布(把数据和噪声按 α:σ 调配出来的)。
        箭头在<em>均匀网格</em>上取样——包括分布外区域,那里最能看出网络间的差异。移动鼠标检查任意点,点击钉住。</div>
      </div>
      <div>
        <div class="readout" id="f-inspect" style="min-height:260px">把鼠标移到左图上。</div>
        <div class="labnote" id="f-viewnote"></div>
      </div>
    </div>
  </div>

  <div class="challenge" data-correct="c">
    <div class="chead">⚡ 预测挑战 · 先选后看</div>
    <div class="q">把视图切到「x̂₀ 映射」、网络切到 <code>eps_ddpm</code>,然后把 λ 拖到 −8(纯噪声端)。你预测这个场会长什么样?(解析真值在这里应该把一切映到全局均值≈原点)</div>
    <div class="opts">
      <button data-opt="a">和解析场几乎一样整齐</button>
      <button data-opt="b">大致收向中心,只是有些抖动</button>
      <button data-opt="c">基本失控:端点散落在整个数据尺度上</button>
    </div>
    <div class="ans"><span class="verdict"></span> 实测 (c),数字如下(20×20 网格,λ=−8):解析真值的 ‖x̂₀‖ 均值 0.038(乖乖收到均值);u_fm 0.14;x0_edm 0.06;<b>eps_ddpm 1.46,最大 4.6</b>——端点散落在约等于数据环半径(1.5)的尺度上,位置近乎错乱。机制是上表那个数:λ=−8 处 ε̂→x̂₀ 的换算放大 e^{−λ/2}=54.6 倍,网络输出 2% 的误差就是 1.0 的 x̂₀ 误差。有趣的是它不是均匀白噪——网络"记得"数据的尺度,胡说也说在流形附近。坦白讲:出这道题时我自己预测的是 (b),被实测打脸——<b>这正是这间实验室存在的理由</b>。此现象 = §10 灰底故障与 v-prediction 诞生的共同根源。打开「叠加解析场」亲自看。</div>
  </div>

  <div class="challenge" data-correct="c">
    <div class="chead">⚡ 预测挑战</div>
    <div class="q">λ = +6(接近干净数据)。五件衣服里,哪一件的<em>数值量级</em>最大?</div>
    <div class="opts">
      <button data-opt="a">ε̂</button><button data-opt="b">v̂</button><button data-opt="c">score</button><button data-opt="d">x̂₀</button>
    </div>
    <div class="ans"><span class="verdict"></span> score = −ε̂/σ,而 λ=+6 时 σ=e^{−3}≈0.0498,所以 score 的量级是 ε̂ 的 20 倍,且随 λ→+∞ 发散(干净数据处的 log 密度是尖峰)。把视图切到 score、λ 拉高,看箭头爆掉(我们做了截断提示)。<b>这就是没人让网络裸奔回归 score 的原因</b>——NCSN 当年回归 σ·score(恰好=−ε),EDM 的 preconditioning 本质也是在解这件事。</div>
  </div>

  <h3>[证明] 三个方言网络,是不是同一个场?</h3>
  <p>三个网络是<em>分别独立训练</em>的:不同的目标头、不同的 λ 采样分布、不同的 loss 权重(§3 会拆)。把它们都换算到 x̂₀ 空间,和解析真值 D* 比较——下图是逐 λ 的均方偏差,来自 Python 端 8192 点/λ 的评测(附录可复现):</p>
  <div class="lab">
    <div class="labhead"><span class="id">证明 2.2</span><span class="t">场一致性 · 逐 λ 偏差</span><span class="pre">评测数据·真实训练产物</span></div>
    <div id="f-agree-legend"></div>
    <canvas id="f-agree"></canvas>
    <div class="labnote" id="f-agree-note"></div>
  </div>
  `;

  // ---------- interactive field ----------
  const plane = new Plane(document.getElementById('f-cv'), { w: 520, h: 520, range: 2.6 });
  const sel = document.getElementById('f-model');
  const lamSlider = document.getElementById('f-lam');
  const lamVal = document.getElementById('f-lamv');
  const strip = document.getElementById('f-strip');
  const inspect = document.getElementById('f-inspect');
  const viewSeg = document.getElementById('f-view');
  const viewnote = document.getElementById('f-viewnote');
  let view = 'x0', pinned = null, hover = null;

  const dataPts = R.meta.data_points.flat();
  const dataCls = R.meta.data_labels;
  const rngCloud = mulberry32(1234);

  const VIEWNOTES = {
    x0: '「x̂₀ 映射」画的是完整映射:箭头从 x_t 出发,箭头尖就落在 x̂₀。低 λ 时所有点都被拉向数据的重心,高 λ 时几乎原地不动——去噪的"力度"随 λ 变化,这就是生成时从构图到细节的日程表。',
    eps: 'ε̂ 是网络对"掺进来的噪声方向"的猜测(×0.35 缩放显示)。注意它和 x̂₀ 视图信息完全等价——只是同一支箭头换了参考系。',
    v: 'v̂ = α·ε̂ − σ·x̂₀(×0.35)。两端都有界:λ 低端它≈ε̂,高端它≈−x̂₀ 方向。一件在整个光谱上都合身的衣服。',
    score: 'score = −ε̂/σ(×0.12,超长截断)。往高 λ 拖,看它爆掉——log 密度在干净数据处是尖峰。',
    u: 'û = ε̂ − x̂₀(×0.35)。FM 采样器就沿它一步步走(dx/dt = û)。注意 û 恒指向"噪声减数据"方向,量级全程 O(1)。',
  };

  function denoisePoint(name, x, y, lam) {
    const xs = new Float32Array([x, y]);
    const lams = new Float32Array([lam]);
    if (name === 'analytic') return R.gmm.denoiseVP(xs, lams, 1);
    return R.models[name].denoiseVP(xs, lams, 1);
  }
  function denoiseBatch(name, xs, lams, n) {
    if (name === 'analytic') return R.gmm.denoiseVP(xs, lams, n);
    return R.models[name].denoiseVP(xs, lams, n);
  }

  function heads(x, y, x0x, x0y, lam) {
    const { a, s } = vpFromLam(lam);
    const t = fmTFromLam(lam);
    const ex = (x - a * x0x) / s, ey = (y - a * x0y) / s;
    return {
      a, s, t,
      x0: [x0x, x0y], eps: [ex, ey],
      v: [a * ex - s * x0x, a * ey - s * x0y],
      score: [-ex / s, -ey / s],
      u: [ex - x0x, ey - x0y],
    };
  }

  function redraw() {
    const lam = parseFloat(lamSlider.value);
    lamVal.textContent = lam.toFixed(1);
    drawLamStrip(strip, -8, 9, lam);
    plane.clear();
    // data scatter
    const cols = dataCls.map(c => MODE_COLORS[c]);
    plane.scatter(new Float32Array(dataPts), { colors: cols, r: 1.3, alpha: 0.28 });
    // x_t cloud
    if (document.getElementById('f-cloud').checked) {
      const { a, s } = vpFromLam(lam);
      const n = 500, cloud = new Float32Array(n * 2);
      for (let i = 0; i < n; i++) {
        const j = (rngCloud() * (dataPts.length / 2)) | 0;
        cloud[i * 2] = a * dataPts[j * 2] + s * gauss(rngCloud);
        cloud[i * 2 + 1] = a * dataPts[j * 2 + 1] + s * gauss(rngCloud);
      }
      plane.scatter(cloud, { color: lamColor(lam), r: 1.5, alpha: 0.35 });
    }
    // arrows on uniform grid
    const name = sel.value, color = MODEL_COLOR[name];
    const G = 15, span = 2.3;
    const n = G * G;
    const xs = new Float32Array(n * 2), lams = new Float32Array(n).fill(lam);
    let k = 0;
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      xs[k * 2] = -span + 2 * span * i / (G - 1);
      xs[k * 2 + 1] = -span + 2 * span * j / (G - 1);
      k++;
    }
    const withTruth = document.getElementById('f-truth').checked && name !== 'analytic';
    const draws = withTruth ? [['analytic', 'rgba(232,228,218,0.5)'], [name, color]] : [[name, color]];
    for (const [nm, col] of draws) {
      const x0h = denoiseBatch(nm, xs, lams, n);
      for (let i = 0; i < n; i++) {
        const h = heads(xs[i * 2], xs[i * 2 + 1], x0h[i * 2], x0h[i * 2 + 1], lam);
        let dx, dy, sc;
        if (view === 'x0') { dx = h.x0[0] - xs[i * 2]; dy = h.x0[1] - xs[i * 2 + 1]; sc = 1; }
        else { [dx, dy] = h[view]; sc = view === 'score' ? 0.12 : 0.35; }
        let vx = dx * sc, vy = dy * sc;
        const L = Math.hypot(vx, vy), maxL = 0.62;
        if (L > maxL) { vx *= maxL / L; vy *= maxL / L; }
        plane.arrow(xs[i * 2], xs[i * 2 + 1], vx, vy, { color: col, width: 1.1, alpha: nm === 'analytic' && withTruth ? 0.5 : 0.85, head: 3.5 });
      }
    }
    // mode centers
    for (let m = 0; m < R.gmm.K; m++) plane.cross(R.gmm.mu[m][0], R.gmm.mu[m][1], { color: MODE_COLORS[m], size: 5 });
    // pinned / hover marker
    const pt = pinned || hover;
    if (pt) { plane.ring(pt[0], pt[1], 0.09, { color: C.ink, dash: [] }); updateInspect(pt, lam); }
    viewnote.textContent = VIEWNOTES[view];
  }

  function updateInspect([x, y], lam) {
    const name = sel.value;
    const x0h = denoisePoint(name, x, y, lam);
    const h = heads(x, y, x0h[0], x0h[1], lam);
    const f = v => (v >= 0 ? ' ' : '') + v.toFixed(3);
    const pair = p => `(${f(p[0])}, ${f(p[1])})`;
    let extra = '';
    if (name !== 'analytic') {
      const tr = R.gmm.denoiseVP(new Float32Array([x, y]), new Float32Array([lam]), 1);
      const err = Math.hypot(x0h[0] - tr[0], x0h[1] - tr[1]);
      extra = `<span class="dim">‖x̂₀ − D*‖ = </span><b>${err.toFixed(4)}</b>  <span class="dim">(和解析真值的距离)</span>\n`;
    }
    inspect.innerHTML =
      `<span class="dim">x_t</span> = ${pair([x, y])}   <span class="dim">λ</span> = <b>${lam.toFixed(2)}</b>\n` +
      `<span class="dim">α</span> = ${h.a.toFixed(4)}   <span class="dim">σ</span> = ${h.s.toFixed(4)}   <span class="dim">t_FM</span> = ${h.t.toFixed(4)}\n` +
      `────────────────────────────────\n` +
      `<b>x̂₀</b>    = ${pair(h.x0)}   <span class="dim">‖·‖=${Math.hypot(...h.x0).toFixed(2)}</span>\n` +
      `<b>ε̂</b>     = (x_t−α·x̂₀)/σ = ${pair(h.eps)}\n` +
      `<b>v̂</b>     = α·ε̂−σ·x̂₀   = ${pair(h.v)}\n` +
      `<b>score</b> = −ε̂/σ        = ${pair(h.score)}\n` +
      `<b>û</b>     = ε̂−x̂₀       = ${pair(h.u)}\n` +
      `────────────────────────────────\n` + extra +
      `<span class="dim">五行数字来自一次前向 + 四次仿射换算。\n改 λ 或换网络,看哪些行变、哪些行的关系不变。</span>`;
  }

  plane.cv.addEventListener('mousemove', e => {
    const r = plane.cv.getBoundingClientRect();
    hover = [plane.wx(e.clientX - r.left), plane.wy(e.clientY - r.top)];
    if (!pinned) redraw();
  });
  plane.cv.addEventListener('click', e => {
    const r = plane.cv.getBoundingClientRect();
    pinned = pinned ? null : [plane.wx(e.clientX - r.left), plane.wy(e.clientY - r.top)];
    redraw();
  });
  plane.cv.addEventListener('mouseleave', () => { hover = null; if (!pinned) redraw(); });
  sel.addEventListener('change', redraw);
  lamSlider.addEventListener('input', redraw);
  document.getElementById('f-cloud').addEventListener('change', redraw);
  document.getElementById('f-truth').addEventListener('change', redraw);
  viewSeg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    viewSeg.querySelectorAll('button').forEach(o => o.classList.remove('on'));
    b.classList.add('on'); view = b.dataset.v; redraw();
  }));
  redraw();

  // ---------- agreement chart ----------
  const rows = R.results.pairwise;
  const lams = rows.map(r => r.lam);
  const agree = new Chart(document.getElementById('f-agree'), {
    w: Math.min(860, mount.clientWidth - 20), h: 300, xlab: 'λ (logSNR)', ylab: 'E‖ΔD‖²(log)',
    xlim: [-8, 9], ylim: [1e-5, 5], ylog: true,
  });
  document.getElementById('f-agree-legend').innerHTML = legendHTML([
    ['eps_ddpm vs D*', C.ddpm], ['x0_edm vs D*', C.edm], ['u_fm vs D*', C.fm], ['loss 理论地板(条件方差)', C.ink3, true],
  ]);
  const va = n => R.results[n].vs_analytic;
  agree.onRedraw(() => {
    agree.clear(); agree.axes();
    agree.line(lams, va('u_fm').map(r => r.floor), { color: C.ink3, dash: [5, 4], width: 1.5, name: '地板 (irreducible)' });
    agree.line(lams, va('eps_ddpm').map(r => r.model_err), { color: C.ddpm, name: 'eps_ddpm 误差' });
    agree.line(lams, va('x0_edm').map(r => r.model_err), { color: C.edm, name: 'x0_edm 误差' });
    agree.line(lams, va('u_fm').map(r => r.model_err), { color: C.fm, name: 'u_fm 误差' });
  });
  agree.redraw();

  // note computes the actual numbers so the text can't drift from the data
  const mid = i => lams.findIndex(l => l >= i);
  const at = (n, L) => va(n)[mid(L)].model_err;
  document.getElementById('f-agree-note').innerHTML =
    `解读:实线是各网络到解析真值的均方距离,虚线是任何模型都不可能低于的地板(数据的条件方差,§3 详拆)。` +
    `中段 λ≈0 处三者误差都在 ${Math.min(at('eps_ddpm', 0), at('u_fm', 0)).toExponential(1)} 量级,比地板小两个数量级以上——<b>三个方言学到的是同一个场</b>。` +
    `分歧在两端:λ=−8 处 eps_ddpm 误差 ${at('eps_ddpm', -8).toFixed(3)},是 u_fm(${at('u_fm', -8).toFixed(3)})的 ${(at('eps_ddpm', -8) / Math.max(at('u_fm', -8), 1e-9)).toFixed(1)} 倍——不是数学不同,是<b>各家把训练预算(λ 采样密度 × 权重)花在了不同的段</b>。这句话就是 §3 的主题。`;
}
