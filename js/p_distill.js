// p_distill.js — §7: distillation. Teacher (instantaneous velocity, curved),
// ReFlow (instantaneous velocity, straightened), 1-step (mean velocity).
// All three run live; the seam map makes "why one step is hard" visible.

import { R } from './models.js';
import { Plane, MODE_COLORS } from './viz2d.js';
import { C, legendHTML } from './charts.js';
import { mulberry32, gauss } from './gmm.js';
import { fmEuler, runToEnd, fmToVP, endpointGap } from './diff.js';

// yield to the event loop without setTimeout (hidden-tab timer throttling stalls it)
const tick = () => new Promise(r => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });

export async function init(mount) {
  let D2 = null;
  try { D2 = await (await fetch('data/distill2d_results.json')).json(); } catch { }
  const haveStudents = R.models.u_reflow && R.models.u_1step;

  mount.innerHTML = `
  <p>先把"蒸馏"这个词拆掉。教师模型定义了一个<strong>确定性映射</strong>:噪声 x₁ → ODE 终点 x₀(§5 已证)。
  采样的几十步,只是在用迭代法<em>求值</em>这个映射。所以蒸馏不创造信息——它把「推理时的迭代计算」搬进「权重里的直接查表」。
  问题在于这个映射本身的性质:它把一整团高斯捏到几个离散的 mode 上,<strong>在 mode 交界处剧烈弯折、近乎不连续</strong>。
  多步法用"每步只走一点、走完再看"来绕开曲率;一步法必须正面硬学这个恶劣函数。三条路线,对应三种回答:</p>
  <table class="data wide">
    <tr><th>路线</th><th>学生回归什么</th><th>代表</th><th>我们的 2D 对应物</th></tr>
    <tr><td><b>把路修直</b></td><td>还是瞬时速度,但换成配对数据 (x₀,x₁) 的直线插值</td><td>ReFlow / Rectified Flow(SD3 血统)</td><td class="m">u_reflow</td></tr>
    <tr><td><b>学平均速度</b></td><td>区间平均速度 ū(x,r,t)=位移/时间;一步=一次求值</td><td>Consistency 系(CM/CTM/sCM)、Shortcut、<b>MeanFlow</b></td><td class="m">u_1step(=MeanFlow 取 r=0,t=1 的特例)</td></tr>
    <tr><td><b>只对分布负责</b></td><td>不回归轨迹,直接让"学生一步样本的分布"匹配教师(score 差/判别器)</td><td>DMD/DMD2、ADD(SDXL-Turbo)、f-distill</td><td class="m">(见叙事——需要第二张网络,略)</td></tr>
  </table>

  <div class="lab">
    <div class="labhead"><span class="id">实验 7.1</span><span class="t">轨迹直化:教师 vs ReFlow</span><span class="live">LIVE</span></div>
    <div class="ctl">
      <button class="primary" id="ds-traj">画 60 条轨迹</button>
      <div class="grp"><label>少步端点漂移</label><button id="ds-drift">测 2/4/8 步 vs 128 步</button></div>
    </div>
    <div class="grid2">
      <div><canvas id="ds-cv1"></canvas><div class="labnote">同一批噪声种子。<span style="color:${C.fm}">玫红=教师 u_fm</span>,<span style="color:${C.student}">薄荷=ReFlow 学生</span>。教师的轨迹先冲向"全局均值"再拐向具体 mode(条件期望的宿命);ReFlow 用教师自己的(噪声,终点)配对重训,路几乎变直——<b>同样的终点分布,更好积分的路径</b>。</div></div>
      <div><div class="readout" id="ds-read1" style="min-height:200px">「画轨迹」看几何;「测漂移」看数字。</div></div>
    </div>
  </div>

  <div class="callout">
  <b>一个训练日志里的深洞察:</b>本仓库教师 u_fm 的训练 loss 收敛在 ~0.8(那是地板——u 目标带着 Var(ε−x₀|x_t) 的条件方差,§3);
  而 ReFlow 学生的 loss 直落 <b>6×10⁻⁵</b>。不是学生更强——是<b>配对把地板拆了</b>:教师的目标是"给定 x_t,对所有可能 (x₀,ε) 配对的平均速度"(有方差→期望→轨迹弯);
  ReFlow 的数据是教师 ODE 给出的一一配对,目标几乎确定(除了极少的轨迹交叉处)。
  <b>曲率的来源和 loss 地板的来源是同一个东西:条件不确定性。</b>把不确定性从数据里挤掉,路自然就直了。这一句是 rectified flow 全部数学的物理内容。</div>

  <div class="lab">
    <div class="labhead"><span class="id">实验 7.2</span><span class="t">一步映射的缝(seam map)</span><span class="live">LIVE·点击计算</span></div>
    <div class="ctl"><button class="primary" id="ds-seam">计算映射图(≈4s)</button><span class="val" id="ds-prog"></span></div>
    <div class="grid2">
      <div><canvas id="ds-cv2"></canvas><div class="labnote"><b>左:教师(32 步)</b>。噪声平面上每个点按"落进哪个 mode"着色,亮度=离 mode 中心距离(暗=贴中心)。</div></div>
      <div><canvas id="ds-cv3"></canvas><div class="labnote"><b>右:1 步学生(平均速度)</b>。看缝:色块边界处出现亮带——学生在不连续处输出了"两边的折中",这些点落在 mode 之间、流形之外。<b>一步生成的一切困难浓缩在这几条缝里</b>。图像模型里,"缝"对应語义模糊的噪声种子,L2 学生给糊图,GAN/DMD 损失逼学生"选边站"。</div></div>
    </div>
    <div class="readout" id="ds-read2"></div>
  </div>

  <div class="challenge" data-correct="c">
    <div class="chead">⚡ 预测挑战</div>
    <div class="q">ReFlow 之后,2 步 Euler 的端点漂移(相对各自 128 步)会降到教师的几分之一?</div>
    <div class="opts"><button data-opt="a">差不多(路径又不是误差主源)</button><button data-opt="b">~10 倍</button><button data-opt="c">100 倍以上</button></div>
    <div class="ans"><span class="verdict"></span> 本仓库实测(8192 点,Python 端):教师 2 步漂移 <b>0.714</b> → ReFlow <b>0.0054</b>,<b>132 倍</b>。按「测漂移」在浏览器里复测一遍。
    但把这个数字直接搬去图像世界会栽跟头:这里的 6-mode 玩具几乎没有轨迹交叉,配对后条件方差≈0,路能修到近乎笔直;真实图像数据的 reflow 典型收益是一个数量级,且 SD3 报告过多轮 reflow 收益快速递减——<b>交叉与条件不确定性越多,"边际直"离"逐条直"越远</b>。玩具给机制,真实给刻度,两个都要拿。<span class="tag f">事实</span></div>
  </div>

  <div id="ds-mnist"></div>

  <h3>代价清单:压掉的到底是什么</h3>
  <p>把"几十步→1-4 步"的账逐项算清(每一项都能在上面的实验里指出对应物):</p>
  <ul>
    <li><b>误差修正消失。</b>多步采样每步都重新问一次场("我现在在哪,该往哪走"),走偏了下一步会拉回来;一步学生没有第二次机会。SDE 采样的 churn 更是主动纠错(§5)。这就是少步模型对 out-of-distribution 起点(inpainting、img2img、编辑)明显更脆的机制。<span class="tag i">推断+实验一致</span></li>
    <li><b>多样性税。</b>缝上的种子在教师那里各有归属,在 L2 学生这里得到均值(糊),在 GAN/DMD 学生那里被推向某一边(mode 收缩)。DMD2 论文自己报告了多样性下降,靠 GAN 项和更大学生缓解。<span class="tag f">事实</span></li>
    <li><b>引导固化。</b>教师采样时 w 可调;蒸馏时 w(或其分布)被焙进权重。Flux 把 w 做成输入蒸(guidance embedding)来保住旋钮。<span class="tag f">事实</span></li>
    <li><b>组合性变差。</b>多步过程中途可以换 prompt、加控制、拼接场——一步映射没有"中途"。实时视频那条线(§8)对此格外敏感。</li>
  </ul>
  <div class="callout">上限之争(§9 分歧二的预告):2024 年的共识还是"少步=有损压缩";2025-26 的 MeanFlow(从零训练 1-NFE ImageNet-256 FID 3.43,变体已到 ~2.0)和 rCM/sCM 大规模蒸馏正在把"损"逼到测量误差内。<span class="tag f">事实·检索 2026-07</span> 我的立场:在"分布保真"意义上少步会追平多步;但"推理时补计算"的能力(纠错、组合、search)是结构性的,不会被追平——这决定两者长期分工。<span class="tag j">判断</span></div>
  `;

  const el = id => document.getElementById(id);
  if (!haveStudents) {
    el('ds-read1').innerHTML = '学生模型尚未训练完成(distill2d.py 运行中)。刷新后可用。';
  }

  // ---- 7.1 trajectories + drift ----
  const p1 = new Plane(el('ds-cv1'), { w: 460, h: 460, range: 2.6 });
  function baseScatter(p) {
    p.clear();
    p.scatter(new Float32Array(R.meta.data_points.flat()), { color: '#39414E', r: 1.1, alpha: 0.3 });
    for (let m = 0; m < R.gmm.K; m++) p.cross(R.gmm.mu[m][0], R.gmm.mu[m][1], { color: MODE_COLORS[m], size: 5 });
  }
  baseScatter(p1);
  el('ds-traj').addEventListener('click', () => {
    if (!haveStudents) return;
    baseScatter(p1);
    const N = 60, rng = mulberry32(31415);
    const x1 = new Float32Array(N * 2);
    for (let i = 0; i < N * 2; i++) x1[i] = gauss(rng);
    for (const [name, color] of [['u_fm', C.fm], ['u_reflow', C.student]]) {
      const m = R.models[name];
      const s = fmEuler((x, lam, n) => m.denoiseVP(x, lam, n), Float32Array.from(x1), N, 48);
      const trails = Array.from({ length: N }, () => []);
      let lam = -8.4;
      for (let i = 0; i < N; i++) trails[i].push(fmToVP(s.xs[i * 2], lam), fmToVP(s.xs[i * 2 + 1], lam));
      while (!s.done) {
        const r = s.step(); lam = r.lam;
        for (let i = 0; i < N; i++) trails[i].push(fmToVP(s.xs[i * 2], lam), fmToVP(s.xs[i * 2 + 1], lam));
      }
      for (const t of trails) p1.path(new Float32Array(t), { color, width: 1, alpha: 0.5 });
    }
  });
  el('ds-drift').addEventListener('click', async () => {
    if (!haveStudents) return;
    const N = 300, seed = 2718;
    const lines = ['少步端点漂移(同 300 种子,vs 各自 128 步):', ''];
    for (const [name, label] of [['u_fm', '教师'], ['u_reflow', 'ReFlow']]) {
      const m = R.models[name];
      const d = (x, lam, n) => m.denoiseVP(x, lam, n);
      const mk = steps => { const rng = mulberry32(seed); const x1 = new Float32Array(N * 2); for (let i = 0; i < N * 2; i++) x1[i] = gauss(rng); const s = fmEuler(d, x1, N, steps); runToEnd(s); return s.xs; };
      const ref = mk(128);
      const gaps = [2, 4, 8].map(k => endpointGap(mk(k), ref, N));
      lines.push(`<b>${label}</b>  2步: ${gaps[0].toFixed(4)}   4步: ${gaps[1].toFixed(4)}   8步: ${gaps[2].toFixed(4)}`);
      el('ds-read1').innerHTML = lines.join('\n');
      await tick();
    }
    if (D2) lines.push('', `<span class="dim">Python 端复测(8192 点):教师 2步 ${D2.straightness.teacher.gap_2step.toFixed(4)} → ReFlow ${D2.straightness.reflow.gap_2step.toFixed(4)}</span>`);
    lines.push('', '漂移 ∝ 轨迹曲率 × 步长²(Euler 局部误差)。修路修的就是那个曲率项。');
    el('ds-read1').innerHTML = lines.join('\n');
  });

  // ---- 7.2 seam map ----
  const G = 100, span = 2.8;
  const p2 = new Plane(el('ds-cv2'), { w: 420, h: 420, range: span });
  const p3 = new Plane(el('ds-cv3'), { w: 420, h: 420, range: span });
  p2.clear(); p3.clear();
  el('ds-seam').addEventListener('click', async () => {
    if (!haveStudents) return;
    const teacher = R.models.u_fm, student = R.models.u_1step;
    const dT = (x, lam, n) => teacher.denoiseVP(x, lam, n);
    const n = G * G;
    const xs = new Float32Array(n * 2);
    let k = 0;
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      xs[k * 2] = -span + 2 * span * i / (G - 1);
      xs[k * 2 + 1] = span - 2 * span * j / (G - 1);
      k++;
    }
    // teacher endpoints, 32 steps, chunked for UI
    const endT = new Float32Array(n * 2);
    const CH = 2500;
    for (let off = 0; off < n; off += CH) {
      const m = Math.min(CH, n - off);
      const sub = xs.slice(off * 2, (off + m) * 2);
      const s = fmEuler(dT, sub, m, 32);
      runToEnd(s);
      endT.set(s.xs, off * 2);
      el('ds-prog').textContent = `教师 ${Math.min(off + CH, n)}/${n}`;
      await tick();
    }
    // student: single mean-velocity step x0 = x1 - (t1-t0)*uu; trained with span (0.985-0.002)
    const lam1 = 2 * Math.log((1 - 0.985) / 0.985);
    const lams = new Float32Array(n).fill(lam1);
    const uu = new Float32Array(n * 2);
    for (let off = 0; off < n; off += CH) {
      const m = Math.min(CH, n - off);
      const sub = xs.slice(off * 2, (off + m) * 2);
      const out = student.forward(sub, lams.slice(off, off + m), m);
      uu.set(out, off * 2);
      el('ds-prog').textContent = `学生 ${Math.min(off + CH, n)}/${n}`;
      await tick();
    }
    const endS = new Float32Array(n * 2);
    for (let i = 0; i < n * 2; i++) endS[i] = xs[i] - (0.985 - 0.002) * uu[i];
    // paint both
    let offCnt = 0, offAcc = 0;
    for (const [pl, end] of [[p2, endT], [p3, endS]]) {
      const ctx = pl.ctx, cw = pl.w / G;
      for (let idx = 0; idx < n; idx++) {
        let best = 1e9, bk = 0;
        for (let m2 = 0; m2 < R.gmm.K; m2++) {
          const d = Math.hypot(end[idx * 2] - R.gmm.mu[m2][0], end[idx * 2 + 1] - R.gmm.mu[m2][1]);
          if (d < best) { best = d; bk = m2; }
        }
        if (end === endS) { offAcc += best; offCnt++; }
        const bright = Math.min(1, best / 0.8);
        const base = MODE_COLORS[bk];
        const rr = parseInt(base.slice(1, 3), 16), gg = parseInt(base.slice(3, 5), 16), bb = parseInt(base.slice(5, 7), 16);
        // near a mode center: deep saturated wedge color; off-manifold: burns to white
        const mix = c => Math.round(c * 0.72 * (1 - bright) + 252 * bright);
        ctx.fillStyle = `rgb(${mix(rr)},${mix(gg)},${mix(bb)})`;
        const j = Math.floor(idx / G), i = idx % G;
        ctx.fillRect(i * cw, j * cw, cw + 0.5, cw + 0.5);
      }
    }
    // seam width metric: fraction of grid cells landing >3*std from every mode center
    const off3 = end => { let c2 = 0; for (let i = 0; i < n; i++) { let b = 1e9; for (let m2 = 0; m2 < R.gmm.K; m2++) b = Math.min(b, Math.hypot(end[i * 2] - R.gmm.mu[m2][0], end[i * 2 + 1] - R.gmm.mu[m2][1])); if (b > 3 * R.gmm.std[0]) c2++; } return c2 / n; };
    el('ds-prog').textContent = '完成';
    el('ds-read2').innerHTML =
      `离群率(落点距一切 mode 中心 &gt; 3σ 的噪声占比):教师 <b>${(off3(endT) * 100).toFixed(2)}%</b> → 一步学生 <b>${(off3(endS) * 100).toFixed(2)}%</b>\n` +
      `<span class="dim">这些多出来的离群点几乎全部排在色块边界上——L2 回归在不连续处输出均值,产出"哪个 mode 都不是"的样本。图像世界里这就是那批"说不清是猫是狗"的糊图。</span>`;
  });

  // ---- MNIST student assets ----
  try {
    const S = await (await fetch('data/mnist/student.json')).json();
    el('ds-mnist').innerHTML = `
    <div class="lab">
      <div class="labhead"><span class="id">实验 7.3</span><span class="t">MNIST:朴素一步蒸馏的真实结果</span><span class="pre">预渲染</span></div>
      <div class="imgrow">
        <div class="imgcell"><img src="data/mnist/student_teacher.png" width="272"><div class="cap">教师 · ${S.teacher_nfe} NFE</div></div>
        <div class="imgcell"><img src="data/mnist/student_1step.png" width="272"><div class="cap">一步学生 · ${S.student_nfe} NFE(同种子)</div></div>
      </div>
      <div class="labnote">高频能量(相邻像素差的方差):教师 <b>${S.hf_teacher.toFixed(4)}</b> vs 学生 <b>${S.hf_student.toFixed(4)}</b>
      (${(S.hf_student / S.hf_teacher * 100).toFixed(0)}%)——32 倍加速,换 ~12% 的锐度税,而且逐列对比能看到学生真的学到了"同种子同姿态"的映射。
      为什么朴素 L2 在这里如此能打?<b>因为类条件把缝切薄了</b>:一步映射最恶劣的不连续在"这团噪声该当 3 还是 8"的选择处,而标签把选择题改成了填空题。
      把这个观察放大就是一条生产规律:<b>条件越强,蒸馏越容易</b>——文生图蒸馏(prompt 几乎唯一确定内容)比无条件生成容易得多;反过来,蒸馏后模型的多样性损失也最先出现在条件约束不到的自由度上。剩下 12% 的税(和 2D 缝图上那 1.2% 的离群增量)就是 DMD/对抗项要挣回来的部分。</div>
    </div>`;
  } catch { }
}
