// p_train.js — §3: one training step, three dialects, line by line, live numbers.
// Then the real economics: where each dialect spends its gradient budget on the lam axis.

import { R } from './models.js';
import { C, Chart, legendHTML, fmt, lamColor } from './charts.js';
import { vpFromLam } from './nn.js';
import { mulberry32, gauss } from './gmm.js';

export async function init(mount) {
  mount.innerHTML = `
  <p>把任何一家的训练循环写全,是同样的六行。真正的分歧只有两处,我用金色标出:</p>
  <pre>for step in range(N):
    x0    = sample_data(B)
    <b>λ     = sample_noise_level()        # ① 各家不同:往哪些噪声级投预算</b>
    ε     = randn_like(x0)
    x_t   = α(λ)·x0 + σ(λ)·ε            # 调配,不是"扩散过程"
    pred  = net(x_t, λ)                 # 网络永远只见 (x_t, λ)
    loss  = <b>w(λ)</b>·‖pred − target(λ)‖²    # ② 目标头 + 权重:同一 x̂₀-loss 的不同汇率</b></pre>
  <p>下面这台显微镜同时执行三种方言的"一步":<strong>同一批 x₀、同一份 ε</strong>,各自按自家规矩抽 λ、构造目标、算 loss。
  所有数字来自你浏览器里的真实前向传播。多按几次「重抽一步」,注意 λ 那一行——三家抽到的位置分布完全不同,这不是噪声,是策略。</p>

  <div class="lab">
    <div class="labhead"><span class="id">实验 3.1</span><span class="t">一步训练 · 三方言并排</span><span class="live">LIVE</span>
      <button class="primary" id="t-resample" style="margin-left:auto">重抽一步 ↻</button></div>
    <div style="overflow-x:auto"><table class="data wide" id="t-table" style="min-width:840px"></table></div>
    <div class="labnote">网络是三张<em>独立训练</em>的网(§2 的那三张)。loss 行给了两个数:各家原生 loss,和统一换算到 x̂₀ 空间之后的等效值——
    换算用的就是右列的汇率 w(λ)。同一个几何目标,三种记账法。灰色行是训练里"看不见"的对照信息(真目标、真 x₀),实际训练只用得到黑色行。</div>
  </div>

  <h3>① 的真面目:λ 采样密度,就是预算曲线</h3>
  <p>三家的「抽 λ」策略画在同一根轴上(解析式,非拟合),再叠上各家权重换算成 x̂₀ 空间后的<em>有效训练压强</em> ∝ 密度 × 汇率:</p>
  <div class="lab">
    <div class="labhead"><span class="id">图 3.2</span><span class="t">λ 预算:密度 · 汇率 · 压强</span></div>
    <div id="t-press-legend"></div>
    <canvas id="t-density"></canvas>
    <canvas id="t-press" style="margin-top:14px"></canvas>
    <div class="labnote">上:p(λ) 采样密度。下:密度 × w(λ) 归一化后的压强(log 轴)。
    读法:<b>压强曲线在哪里高,网络就在哪个噪声段被逼着学得准</b>。回头对照 §2 的偏差图——
    eps_ddpm 在 λ&lt;−5 误差爆炸,正是它的压强在那里塌掉的位置(e^λ 汇率在低 λ 处趋零:ε-loss 对 x̂₀ 误差近乎免税)。
    EDM 的 lognormal 密度 + λ(σ) 权重、FM 的 logit-normal,都是把压强集中在"任务既非平凡也非不可能"的中段——两家 2022/2024 年各自独立收敛到同一个设计判断。</div>
  </div>

  <div class="challenge" data-correct="c">
    <div class="chead">⚡ 预测挑战</div>
    <div class="q">这套 2D 数据上,x̂₀-空间的 loss 理论地板(不可约条件方差,2 维合计)在 λ=0 处大约是多少?提示:数据 σ_data≈1.08/维,λ=0 意味着信噪各半。</div>
    <div class="opts"><button data-opt="a">≈ 0.1</button><button data-opt="b">≈ 0.5</button><button data-opt="c">≈ 1.0</button><button data-opt="d">≈ 2.3(=全方差)</button></div>
    <div class="ans"><span class="verdict"></span> 解析值 <b>1.004</b>(fp64 蒙特卡洛 1.004,附录自检表)。λ=0 时观测已含一半信息,后验方差约为先验方差(2×1.08²≈2.34)的四成。
    要点不是这个数,而是:<b>loss 有一条永远碰不到的地板,而且地板的形状(随 λ 的曲线)是数据的性质,与模型无关</b>。训练监控里"loss 不降了"必须先问:是撞地板了,还是没学动?只有减去地板(或分 λ 桶对比)才能区分。下一张图就是三张网离地板的真实距离。</div>
  </div>

  <h3>② 的后果:分桶看 loss,别看平均</h3>
  <div class="lab">
    <div class="labhead"><span class="id">图 3.3</span><span class="t">训练过程 · 每个 λ 桶的 x̂₀-loss 随步数</span><span class="pre">训练日志·真实</span></div>
    <div class="ctl"><div class="grp"><label>网络</label><select id="t-net">
      <option value="u_fm">u_fm</option><option value="eps_ddpm">eps_ddpm</option><option value="x0_edm">x0_edm</option>
    </select></div></div>
    <canvas id="t-curve"></canvas>
    <div class="labnote" id="t-curve-note"></div>
  </div>

  <div class="callout">
  <b>换算表(全部经 fp64 数值验证,附录 11.3):</b>任何一家的 loss 都等于 w(λ)·‖x̂₀−x₀‖² 的加权族。
  <span class="m">ε-loss: w=e^λ</span> · <span class="m">v-loss: w=1+e^λ</span> · <span class="m">FM u-loss: w=(1+e^{λ/2})²</span> · <span class="m">x₀-loss: w=1</span>。
  注意 FM 的 w 与 v-pred 的 w 全程只差 1~2 倍:<b>flow matching 的训练目标 ≈ cosine 调度下的 v-prediction</b>——这不是巧合,是 Kingma–Gao (2023) 定理的一个特例。两个"阵营"打了两年,损失函数几乎是同一条。<span class="tag f">事实</span>
  </div>
  <details><summary><b>EDM / EDM2 复杂度解压包</b>——那一堆系数各自在干什么(点开)</summary><div class="body">
    <table class="data wide">
      <tr><th>公式里的东西</th><th>它实际在干什么</th></tr>
      <tr><td class="m">c_in = 1/√(σ²+σ_d²)</td><td>输入白化:保证网络无论 σ 多大,进来的张量方差恒为 1。没有它,网络前几层要自己学会"除以一万"。</td></tr>
      <tr><td class="m">c_skip = σ_d²/(σ²+σ_d²)</td><td>让网络只预测"值得预测的部分":低噪时 D≈x(照抄输入,c_skip→1),网络只学残差;高噪时 c_skip→0,网络全权接管。本质是按信噪比在"恒等函数"与"完全预测"之间调混合比——ε-pred 和 x₀-pred 是它的两个极端特例。</td></tr>
      <tr><td class="m">c_out = σ·σ_d/√(σ²+σ_d²)</td><td>输出反白化:网络原始输出恒为单位方差,乘回真实尺度。和 c_skip 联合推导自"最小化输出方差"的一页代数。</td></tr>
      <tr><td class="m">lnσ ~ N(P_mean, P_std)</td><td>λ 预算曲线(本节主角):把梯度集中在"既非平凡也非不可能"的难度带。</td></tr>
      <tr><td class="m">λ(σ) 权重</td><td>让每个 σ 的 loss 在初始化时都 ≈1:没有哪个噪声段天生嗓门大。</td></tr>
      <tr><td class="m">EDM2:magnitude-preserving 层</td><td>把"每层激活范数=1"从祈祷变成结构保证 → 学习率/EMA 的行为在整个训练期稳定;weight decay 都省了。</td></tr>
      <tr><td class="m">EDM2:可学习 w(λ)(不确定性权重)</td><td>把"λ 预算"从手调超参变成多任务学习的不确定性估计,网络自己报告哪段该多花钱——本节压强图的自动驾驶版。</td></tr>
      <tr><td class="m">EDM2:post-hoc EMA</td><td>训练时存多条 EMA 轨迹的基,训练后再合成任意 EMA 长度——把一个不可回溯的超参变成可回溯的。</td></tr>
    </table>
    <p class="small">读 EDM 系论文的正确姿势:所有系数都是同一句话的变体——<b>"让网络在单位尺度上工作,让每个噪声段公平发声"</b>。记住这句话,公式全部可以现场重推。</p>
  </div></details>

  <p>工程推论一:<strong>对比 formulation 的消融,不对齐 λ 压强就是没控制变量</strong>——你测的是权重曲线,不是数学框架。
  工程推论二:纯 loss 权重会被 Adam 的逐参数归一化部分吃掉,所以<em>采样密度</em>比<em>乘法权重</em>更"硬"(等价重要性采样只在 SGD 期望意义下成立,二阶矩不同)。EDM2 (2024) 干脆把 w(λ) 做成可学习的不确定性权重,让网络自己报告哪段该多花预算。<span class="tag f">事实</span> 我的立场:新项目直接用 logit-normal 密度起步,别手调权重。<span class="tag j">判断</span></p>
  `;

  // ---------------- live one-step microscope ----------------
  const rng = mulberry32(20260726);
  const tbl = document.getElementById('t-table');

  function sampleStep() {
    // shared x0 (one sample shown) + shared eps; each dialect draws its own lam
    const K = R.gmm.K;
    const k = (rng() * K) | 0;
    const x0 = [R.gmm.mu[k][0] + R.gmm.std[k] * gauss(rng), R.gmm.mu[k][1] + R.gmm.std[k] * gauss(rng)];
    const eps = [gauss(rng), gauss(rng)];
    // dialect draws
    const tD = rng();                                   // ddpm: t ~ U[0,1], cosine
    const f0 = Math.cos(0.008 / 1.008 * Math.PI / 2);
    const aC = Math.max(1e-5, Math.min(1, Math.cos((tD + 0.008) / 1.008 * Math.PI / 2) / f0));
    const lamD = 2 * Math.log(aC / Math.sqrt(1 - aC * aC + 1e-12));
    const sigE = Math.exp(-0.4 + 1.2 * gauss(rng));     // edm: lognormal sigma
    const lamE = -2 * Math.log(sigE);
    const tF = 1 / (1 + Math.exp(-gauss(rng)));         // fm: logit-normal t
    const lamF = 2 * Math.log((1 - tF) / tF);
    return { x0, eps, rows: [
      { name: 'eps_ddpm', model: 'eps_ddpm', color: C.ddpm, lam: lamD, draw: `t=${tD.toFixed(3)} → cosine`, head: 'ε̂', wname: 'e^λ' },
      { name: 'x0_edm', model: 'x0_edm', color: C.edm, lam: lamE, draw: `lnσ~N(−0.4,1.2²): σ=${sigE.toFixed(3)}`, head: 'D(=x̂₀)', wname: 'λ_EDM(σ)' },
      { name: 'u_fm', model: 'u_fm', color: C.fm, lam: lamF, draw: `t~logitN: t=${tF.toFixed(3)}`, head: 'û', wname: '(1+e^{λ/2})²' },
    ] };
  }

  function renderStep() {
    const st = sampleStep();
    const p2 = v => `(${v[0].toFixed(3)}, ${v[1].toFixed(3)})`;
    const cells = { draw: [], mix: [], fwd: [], tgt: [], loss: [] };
    for (const row of st.rows) {
      const { a, s } = vpFromLam(row.lam);
      const xt = [a * st.x0[0] + s * st.eps[0], a * st.x0[1] + s * st.eps[1]];
      const m = R.models[row.model];
      // native forward in the model's own frame — reuse denoiseVP then convert back to native head
      const x0h = m.denoiseVP(new Float32Array(xt), new Float32Array([row.lam]), 1);
      const epsh = [(xt[0] - a * x0h[0]) / s, (xt[1] - a * x0h[1]) / s];
      const t = 1 / (1 + Math.exp(row.lam / 2));
      let pred, tgt, nat;
      if (row.model === 'eps_ddpm') { pred = epsh; tgt = st.eps; }
      else if (row.model === 'x0_edm') { pred = [x0h[0], x0h[1]]; tgt = st.x0; }
      else { pred = [epsh[0] - x0h[0], epsh[1] - x0h[1]]; tgt = [st.eps[0] - st.x0[0], st.eps[1] - st.x0[1]]; }
      nat = (pred[0] - tgt[0]) ** 2 + (pred[1] - tgt[1]) ** 2;
      const sd = R.meta.sigma_data;
      const w = row.model === 'eps_ddpm' ? Math.exp(row.lam)
        : row.model === 'u_fm' ? (1 + Math.exp(row.lam / 2)) ** 2
        : (Math.exp(-row.lam) + sd * sd) / (Math.exp(-row.lam) * sd * sd);
      const x0loss = (x0h[0] - st.x0[0]) ** 2 + (x0h[1] - st.x0[1]) ** 2;
      const wNat = row.model === 'x0_edm' ? w * nat : nat;  // edm applies w explicitly, others implicitly
      cells.draw.push(`<td class="m" style="color:${row.color}">${row.draw}<br>λ = <b>${row.lam.toFixed(2)}</b>  α=${a.toFixed(3)} σ=${s.toFixed(3)}</td>`);
      cells.mix.push(`<td class="m">x_t = ${p2(xt)}</td>`);
      cells.fwd.push(`<td class="m">${row.head} = ${p2(pred)}</td>`);
      cells.tgt.push(`<td class="m" style="color:${C.ink3}">${row.head.replace('̂', '')}* = ${p2(tgt)}</td>`);
      cells.loss.push(`<td class="m">原生 = ${fmt(wNat)}<br><span style="color:${C.ink3}">x̂₀-空间: ‖Δx₀‖²=${fmt(x0loss)} × w=${fmt(w)}</span></td>`);
      void t;
    }
    tbl.innerHTML = `
      <tr><th style="min-width:110px"></th><th style="color:${C.ddpm}">DDPM 方言</th><th style="color:${C.edm}">EDM 方言</th><th style="color:${C.fm}">FM 方言</th></tr>
      <tr><td class="m">x₀(共享)</td><td class="m" colspan="3">${p2(st.x0)} <span style="color:${C.ink3}">← 数据集抽一个;ε(共享)= ${p2(st.eps)}</span></td></tr>
      <tr><td class="m">① 抽噪声级</td>${cells.draw.join('')}</tr>
      <tr><td class="m">调配 x_t</td>${cells.mix.join('')}</tr>
      <tr><td class="m">前向(真网络)</td>${cells.fwd.join('')}</tr>
      <tr><td class="m" style="color:${C.ink3}">目标(训练时已知)</td>${cells.tgt.join('')}</tr>
      <tr><td class="m">② loss</td>${cells.loss.join('')}</tr>`;
  }
  document.getElementById('t-resample').addEventListener('click', renderStep);
  renderStep();

  // ---------------- density + pressure charts ----------------
  const lams = [];
  for (let l = -9; l <= 10.001; l += 0.15) lams.push(l);
  const nPDF = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  // cosine-uniform-t density over lam: p(lam) = |dt/dlam| numerically
  const tOfLam = lam => {
    const a2 = 1 / (1 + Math.exp(-lam));
    const s = 0.008, f0 = Math.cos(s / (1 + s) * Math.PI / 2);
    return Math.max(0, Math.min(1, (2 / Math.PI) * Math.acos(Math.sqrt(a2) * f0) * (1 + s) - s));
  };
  const pCos = lams.map(l => Math.abs((tOfLam(l + 0.01) - tOfLam(l - 0.01)) / 0.02));
  const pFM = lams.map(l => nPDF(l, 0, 2));
  const pEDM = lams.map(l => nPDF(l, 0.8, 2.4));
  const sd = R.meta.sigma_data;
  const wEps = lams.map(l => Math.exp(l));
  const wU = lams.map(l => (1 + Math.exp(l / 2)) ** 2);
  const wEDM = lams.map(l => (Math.exp(-l) + sd * sd) / (Math.exp(-l) * sd * sd));
  const wV = lams.map(l => 1 + Math.exp(l));
  const norm = arr => { const s2 = arr.reduce((a, b) => a + (isFinite(b) ? b : 0), 0) * 0.15; return arr.map(v => v / s2); };
  const press = (p, w) => norm(p.map((v, i) => v * w[i]));

  document.getElementById('t-press-legend').innerHTML = legendHTML([
    ['DDPM: uniform-t·cosine + ε-loss', C.ddpm], ['EDM: lognormal σ + λ(σ)', C.edm],
    ['FM: logit-normal + u-loss', C.fm], ['v-pred·cosine(参照)', C.ink3, true],
  ]);
  const w1 = Math.min(860, mount.clientWidth - 20);
  const den = new Chart(document.getElementById('t-density'), { w: w1, h: 210, xlab: 'λ', ylab: 'p(λ) 采样密度', xlim: [-9, 10], ylim: [0, 0.23] });
  den.onRedraw(() => {
    den.clear(); den.axes();
    den.line(lams, pCos, { color: C.ddpm, name: 'DDPM 密度' });
    den.line(lams, pEDM, { color: C.edm, name: 'EDM 密度' });
    den.line(lams, pFM, { color: C.fm, name: 'FM 密度' });
  });
  den.redraw();
  const pr = new Chart(document.getElementById('t-press'), { w: w1, h: 240, xlab: 'λ', ylab: '压强 = p·w 归一化(log)', xlim: [-9, 10], ylim: [1e-5, 3], ylog: true });
  pr.onRedraw(() => {
    pr.clear(); pr.axes();
    pr.line(lams, press(pCos, wEps), { color: C.ddpm, name: 'DDPM 压强' });
    pr.line(lams, press(pEDM, wEDM), { color: C.edm, name: 'EDM 压强' });
    pr.line(lams, press(pFM, wU), { color: C.fm, name: 'FM 压强' });
    pr.line(lams, press(pCos, wV), { color: C.ink3, dash: [5, 4], name: 'v·cosine 压强' });
  });
  pr.redraw();

  // ---------------- per-bin training curves ----------------
  const sel = document.getElementById('t-net');
  const cvC = document.getElementById('t-curve');
  const note = document.getElementById('t-curve-note');
  function drawCurve() {
    const name = sel.value;
    const curve = R.results[name].curve;
    const steps = curve.map(c => c.step);
    const nb = curve[0].x0_per_bin.length;
    const ch = new Chart(cvC, { w: w1, h: 300, xlab: '训练步数', ylab: '该 λ 桶的 x̂₀-loss(log)', xlim: [0, steps[steps.length - 1]], ylim: [1e-4, 6], ylog: true });
    ch.onRedraw(() => {
      ch.clear(); ch.axes();
      for (let b = 0; b < nb; b++) {
        const lamB = -9 + (b + 0.5) * 19 / nb;
        if (lamB < -8.6 || lamB > 9.2) continue;
        const ys = curve.map(c => c.x0_per_bin[b]);
        if (ys.every(v => v == null)) continue;
        ch.line(steps, ys, { color: lamColorSafe(lamB), width: 1.4, alpha: 0.9, name: `λ≈${lamB.toFixed(1)}` });
      }
    });
    ch.redraw();
    note.innerHTML = `每条线是一个 λ 桶(颜色即光谱条位置:紫=高噪,金=低噪)。三个读法:
    ① 高 λ(金)桶的 loss 天生小几个数量级——不是学得好,是任务本身简单(地板低);
    ② 低 λ(紫)桶几乎立刻贴地板(高噪端最优解≈整体均值,一层线性就够);
    ③ <b>真正在整个训练期持续下降的是中段桶</b>——生成质量的进步全在那里发生,总 loss 的早期平台完全掩盖了这件事。看训练日志只看总 loss,等于闭着眼睛开车。`;
  }
  function lamColorSafe(l) { return lamColor(Math.max(-9, Math.min(10, l))); }
  sel.addEventListener('change', drawCurve);
  drawCurve();
}
