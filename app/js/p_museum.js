// p_museum.js — §10: failure museum. Exhibit A is a REAL reproduction:
// the leaky-schedule model was trained in this repo, gray backgrounds and all.

import { C, Chart, legendHTML } from './charts.js';

const TREE = [
  {
    sym: '样本发灰 / 对比度塌 / 黑不下去', tensor: '训练调度的 λ_min、√ᾱ_T;生成图的像素均值直方图',
    check: '√ᾱ_T > 0.01 且 ε-pred?几乎实锤。再看:x_T 统计 vs N(0,I) 的均值/方差差。',
    causes: '① 非零 terminal SNR + ε-pred(SD1/2 的历史 bug,本页展品)② 起点分布与训练不符 ③ VAE decode 后的 clamp/normalize 错',
    fix: 'zero-terminal-SNR rescale + 换 v-pred;或直接 FM(端点天生干净)。offset-noise 是止痛药不是手术。',
  },
  {
    sym: '构图对但细节糊', tensor: '分 λ 桶的 loss(高 λ 段);VAE 重建对照',
    check: '把真图过一遍 encode→decode:如果也糊,锅在 VAE 上限,扩散再训也没用。分桶 loss 高 λ 段贴地板则是采样/蒸馏问题。',
    causes: '① VAE/latent 上限 ② 高 λ(低噪)段压强不足(σ 采样分布偏左)③ 蒸馏税 ④ EMA 没开或衰减错',
    fix: 'P_mean 右移 / logit-normal 中心右移;换更好的 autoencoder;确认 EMA;蒸馏加判别器项。',
  },
  {
    sym: '构图崩:多肢、物体拼贴、全局不连贯', tensor: '前 20% 步的 x̂₀ 序列(构图期);低 λ 段分桶 loss',
    check: '只跑前几步,把 x̂₀ 存下来看:构图期就崩 = 高噪段能力不足,不是细节问题。',
    causes: '① 分辨率提升但 schedule 没 shift(同 σ 在高分辨率下破坏力骤降,λ 有效右移)② 低 λ 压强不够 ③ 全局注意力不足',
    fix: 'timestep/σ shift(SD3:按 √(像素数比) 平移 logSNR)、加大高噪采样密度、构图期用更强 CFG。',
  },
  {
    sym: '训练 NaN / loss 尖刺', tensor: '分 λ 桶的梯度范数(找爆点在哪一段);attention logits 幅值',
    check: '爆点集中在 λ 高端 → 权重发散(ε 系换算 1/σ、EDM 权重 ~1/σ²,σ_min 没 clamp);随机步爆 → 数值/数据问题。',
    causes: '① σ_min 太小 + 无界权重 ② fp16(而非 bf16)攒出的溢出 ③ attention 无 qk-norm ④ 坏样本',
    fix: 'clamp σ∈[σ_min,σ_max]、bf16、qk-norm、grad-clip 1.0、EMA 之外再存"上一个好检查点"。',
  },
  {
    sym: '加采样步数,指标不再涨', tensor: 'ODE 收敛曲线(§5.2 那条)vs 分布指标曲线',
    check: '同 NFE 换 2 阶采样器:若无差 → 离散误差已收敛,剩下全是模型误差,加步数当然没用。',
    causes: '① 模型误差主导(正常!)② λ 网格端点没覆盖(起点太低/终点太早)③ 指标饱和',
    fix: '这时买质量的手段:churn/SDE(用随机性对冲模型误差)、autoguidance、更好的模型。别再堆步数。',
  },
  {
    sym: 'CFG 拉高后过曝、塑料感、千人一面', tensor: '像素值直方图的两端占比(§6 饱和曲线)',
    check: 'w 扫描:饱和占比单调上升、类内方差单调下降 → 教科书式 CFG 失真(§6 已在完美模型上复现)。',
    causes: 'CFG 的算术本性,不是 bug',
    fix: 'interval CFG(只在中段开)、CFG rescale、autoguidance、蒸馏时把 w 做成条件输入。',
  },
  {
    sym: '棋盘纹 / 网格状伪影', tensor: 'VAE 重建图(排除法第一步)',
    check: '真图 encode→decode 也有纹?→ decoder 的转置卷积/上采样问题,与扩散无关。',
    causes: '① decoder 上采样核重叠 ② latent 归一化错 ③ patch 边界(DiT patchify)',
    fix: '换 resize+conv 上采样;检查 latent scale 常数(那个著名的 0.18215 一类)。',
  },
];

export async function init(mount) {
  let M = null;
  try { M = await (await fetch('data/mnist/museum.json')).json(); } catch { }

  mount.innerHTML = `
  <p>这一节的镇馆展品是一次三幕剧,全部由隔壁真实训练的模型出演:照着 Stable Diffusion 1.x 的真实配置
  (scaled-linear β,ε-prediction)训出来的泄漏模型,√ᾱ_T = <b>${M ? M.terminal_sqrt_abar.toFixed(4) : '0.068'}</b>,
  终端 logSNR = <b>${M ? M.terminal_logsnr.toFixed(2) : '-5.4'}</b> ≠ −∞。
  翻译成人话:<strong>它训练时从未见过纯噪声</strong>——最"噪"的样本里也永远残留着 6.8% 的原图,
  等于永远给模型递着一张写有全局信息(整体亮度)的小抄。推理从纯 N(0,I) 出发时,小抄消失了。</p>

  <h4>第一幕(阴性对照):bug 在,但在原版 MNIST 上咬不动</h4>
  <div class="readout" id="mu-neg" style="max-width:640px"></div>
  <p>为什么?<b>泄漏偷走的是"从初始噪声决定全局统计"的能力</b>。原版 MNIST 每张图的全局亮度几乎相同(黑底白字,均值全在 −0.80 附近)——
  没有多样性可偷,于是统计上无恙。这个阴性结果值一记重点:<strong>terminal-SNR bug 的咬合条件是数据存在全局统计多样性</strong>。
  自然图像恰好满足(白天/夜景/高调/低调),所以 SD1.x 中招:出不了纯黑画面、夜景发雾。同一个 bug,不同数据,毒性天差地别——判断力比背结论值钱的地方。</p>

  <h4>第二幕(追凶):给数据加上亮度多样性 β~U(0,1),再分别用两种采样器审问泄漏模型</h4>
  <div class="lab">
    <div class="labhead"><span class="id">展品 A</span><span class="t">亮度多样 MNIST:对照组 × 采样器</span><span class="pre">真实训练产物</span></div>
    <div class="imgrow" id="mu-imgs"></div>
    <div id="mu-hist-wrap"><div id="mu-hist-legend"></div><canvas id="mu-hist"></canvas></div>
    <div class="labnote" id="mu-note"></div>
  </div>

  <h4>第三幕(判决):为什么在这里只是擦伤,在 SD1.x 那里是重伤</h4>
  <p>两条实测线索:① <b>ancestral sampling 几乎完全掩盖此 bug</b>(每步重新注噪,不断重掷"全局亮度"这枚硬币,链条中段泄漏可读后自然长出多样性);
  ② 确定性 DDIM 下出现压缩但依然温和。真正的钥匙是一个数:<b>全局统计量的有效 logSNR = 像素 λ + log(参与平均的维度数)</b>。
  本馆画布 32×32=1024 维,log(1024)≈6.9——泄漏调度的 λ_min=−5.4 对单像素是"深噪",对全图均值却相当于 λ≈+1.5,<em>亮度信息本来就还活着</em>,模型不完全依赖小抄。
  换到 SD1.x 的 512×512(或更高维 latent),同一个 λ_min 距离"全局量不可读"近了好几个 nat,加上用户真的会要"纯黑画面"这种分布尾部——bug 就从擦伤变成重伤。</p>
  <div class="callout"><b>带走的判断力,比"zero-SNR 修复"本身值钱:</b>terminal-SNR 泄漏是一个<b>随分辨率变毒</b>的 bug——这与 §9 心智模型二(分辨率上升 = λ 轴左移)和拷问 Q7 是同一件事的三个入口。凡是"低分辨率试验没问题、上了高分辨率就翻车"的低频伪影(发灰、色偏、出不了极端亮度),第一嫌疑人都是它。修法不变:zero-terminal-SNR rescale + v/x₀/FM 头 + 起点合同校验(Lin et al. 2024)。<span class="tag f">事实(机制)+判断(排查优先级)</span></div>

  <div class="challenge" data-correct="c">
    <div class="chead">⚡ 预测挑战</div>
    <div class="q">把本馆这套泄漏配置(λ_min=−5.4)原样搬到 512×512 像素空间训练,亮度塌缩会?</div>
    <div class="opts"><button data-opt="a">一样轻微(bug 就这点毒性)</button><button data-opt="b">消失(大图信息更多)</button><button data-opt="c">严重得多(全局量的有效 λ 掉了 ~4.8 个 nat)</button></div>
    <div class="ans"><span class="verdict"></span> 512²=26 万维,log(262144/1024)≈5.5——同一个 λ_min 下,全图均值的有效 logSNR 从 +1.5 掉到 −4.0,"亮度"从可读变成基本不可读,模型只剩小抄可依赖;推理时小抄消失,塌缩全面发作。这正是本馆没法在 32×32 上"完整复刻" SD1.x 惨案的原因——<b>不是 bug 不在,是画布太小救了它</b>。(这一幕的三次实测把我最初的两版预测都推翻了,推理过程原样保留在展品文字里——这比一个顺利的演示更接近真实的研究。)</div>
  </div>

  <div class="lab">
    <div class="labhead"><span class="id">展品 B</span><span class="t">一次真实的 NaN(就发生在建造本页的过程中)</span><span class="pre">事故存档</span></div>
    <pre>[pairs] 0/40000 ... [pairs] 30000/40000     # 教师蒸馏对生成:无报错,安静完成
[student] 0 loss nan
[student] 1000 loss nan ...                  # 学生从第 0 步就是 NaN</pre>
    <p style="max-width:none">尸检:生成蒸馏配对时,t 网格写成了 <code>linspace(1, 0)</code>——第一步 t=1.0 整,
    λ = 2·log((1−t)/t) = log(0) = <b>−∞</b>,进 sinusoidal time embedding 变成 sin(−∞) = NaN,教师输出 NaN,
    40000 个"教师端点"全军覆没。<strong>最危险的部分是没有任何一步报错</strong>:生成循环安静跑完,数据集被无声毒化,直到下游 loss 才显形。
    (同样的代码在 2D 版里没事,因为那里从 t=0.985 起步——两份代码差一个字符。)</p>
    <dl class="kv">
      <dt>对应手册条目</dt><dd>「训练 NaN」:爆点与步数无关、从第 0 步就 NaN → 先怀疑数据/条件里的无界量,再怀疑优化</dd>
      <dt>结构性教训</dt><dd>λ 是个<b>两端无界</b>的量,任何吃 λ 的组件(embedding、权重、换算)都必须问一句"端点给我什么"。所有真实系统的起点都截断在有限 λ(EDM 的 σ_max、FM 的 t_start&lt;1)——§5 讲过,这不是数值洁癖,是硬约束。</dd>
      <dt>修复 + 疫苗</dt><dd>t 网格 [0.985, 0.002];配对生成后 <code>assert isfinite(X0).all()</code>——一行断言,值一下午。</dd>
    </dl>
  </div>

  <h3>值班手册:症状 → 第一诊断动作</h3>
  <p>排查的元规则只有一条:<strong>先定位坏在 λ 轴的哪一段,再问为什么</strong>。低频/构图/整体色调=高噪段;纹理/边缘=低噪段;对比度/亮度极值=终端;然后去看那一段的分桶 loss、梯度范数、和采样器覆盖。以下每行都按"最快证伪"排序:</p>
  <div id="mu-tree"></div>

  <div class="callout blue">
  <b>三条老兵纪律</b>(代价换来的,拿去):
  ① 任何"formulation 换了就变好"的结论,先检查是不是只是 λ 压强曲线变了(§3)——90% 是。<span class="tag j">判断</span>
  ② 训练日志里必须有分 λ 桶的 loss 和梯度范数,没有等于没监控;总 loss 是安慰剂。
  ③ 采样器出问题先跑"完美模型对照"(解析玩具/教师模型),把"积分器的锅"和"模型的锅"分开——本实验室从 §5 起一直在给你演示这个方法论本身。</div>
  `;

  // act 1: negative control numbers (plain MNIST)
  if (M) {
    document.getElementById('mu-neg').innerHTML =
      `原版 MNIST,256 样本的像素统计(泄漏模型,250 步 ancestral sampling,从 N(0,I) 起):\n` +
      `均值   数据 <b>${M.mean_data.toFixed(4)}</b>   FM <b>${M.mean_fm.toFixed(4)}</b>   泄漏 <b>${M.mean_leaky.toFixed(4)}</b>\n` +
      `<span class="dim">三者几乎重合 —— bug 存在(小抄确实在训练里),但没有可偷的多样性,统计无恙。</span>`;
  } else {
    document.getElementById('mu-neg').innerHTML = '统计生成中……';
  }

  // act 2+3: luminance-diverse assets
  let L = null;
  try { L = await (await fetch('data/mnist/museum_lum.json')).json(); } catch { }
  if (L) {
    document.getElementById('mu-imgs').innerHTML = `
      <div class="imgcell"><img src="data/mnist/lum_data.png" width="204"><div class="cap"><b>数据</b>(亮度多样)<br>背景亮度 std <b>${L.std_data.toFixed(3)}</b></div></div>
      <div class="imgcell"><img src="data/mnist/lum_fm.png" width="204"><div class="cap"><b>健康 FM</b>·确定性采样<br>std <b>${L.std_fm.toFixed(3)}</b></div></div>
      <div class="imgcell"><img src="data/mnist/lum_leaky.png" width="204"><div class="cap"><b>患者</b>·泄漏+DDIM(η=0)<br>std <b style="color:${C.bad}">${L.std_leaky.toFixed(3)}</b></div></div>
      <div class="imgcell"><img src="data/mnist/lum_leaky_fixedinit.png" width="204"><div class="cap"><b>法证</b>·同患者同采样器,塞回小抄<br>std <b style="color:${C.ok}">${L.std_cheat.toFixed(3)}</b></div></div>`;
    const xs = Array.from({ length: L.bins }, (_, i) => L.range[0] + (L.range[1] - L.range[0]) * (i + 0.5) / L.bins);
    document.getElementById('mu-hist-legend').innerHTML = legendHTML([
      ['数据', C.ink3], ['健康 FM', C.fm], ['泄漏 + ancestral(η=1)', C.ddpm], ['泄漏 + DDIM(η=0)', C.bad], ['法证起点 + DDIM', C.ok, true],
    ]);
    const ymax = Math.max(...L.bg_leaky, ...L.bg_data, ...L.bg_leaky_anc) * 1.2;
    const ch = new Chart(document.getElementById('mu-hist'), {
      w: Math.min(760, mount.clientWidth - 20), h: 250, xlab: '每图背景亮度(10 分位像素值)', ylab: '占比',
      xlim: [L.range[0], L.range[1]], ylim: [0, ymax],
    });
    ch.onRedraw(() => {
      ch.clear(); ch.axes();
      ch.line(xs, L.bg_data, { color: C.ink3, name: '数据' });
      ch.line(xs, L.bg_fm, { color: C.fm, name: '健康 FM' });
      ch.line(xs, L.bg_leaky_anc, { color: C.ddpm, name: '泄漏·祖先' });
      ch.line(xs, L.bg_leaky, { color: C.bad, width: 2.5, name: '泄漏·DDIM' });
      ch.line(xs, L.bg_cheat, { color: C.ok, dash: [5, 4], name: '法证·DDIM' });
    });
    ch.redraw();
    document.getElementById('mu-note').innerHTML =
      `实测(背景亮度 std;数据=${L.std_data.toFixed(3)}):健康 FM <b>${L.std_fm.toFixed(3)}</b>;
      泄漏+ancestral <b>${L.std_leaky_anc.toFixed(3)}</b>(几乎无恙——随机性掩盖了 bug);
      泄漏+确定性 DDIM <b style="color:${C.bad}">${L.std_leaky.toFixed(3)}</b>(压缩显形,但仍是擦伤级);
      塞回小抄后 <b style="color:${C.ok}">${L.std_cheat.toFixed(3)}</b>(恢复到健康水平——确认压缩确实来自"合同违约"而非模型能力)。
      两个反直觉读数(ancestral 为何免疫、为何只是擦伤)的机制在下方第三幕——那才是这件展品真正要卖的东西。`;
  } else {
    document.getElementById('mu-imgs').innerHTML = '<div class="small">亮度多样版模型训练中(mnist.py lum),展品稍后自动出现。</div>';
  }

  // diagnostic tree
  document.getElementById('mu-tree').innerHTML = TREE.map(t => `
    <details><summary><b>${t.sym}</b></summary><div class="body">
      <dl class="kv">
        <dt>先看什么</dt><dd>${t.tensor}</dd>
        <dt>判据</dt><dd>${t.check}</dd>
        <dt>根因排序</dt><dd>${t.causes}</dd>
        <dt>修复</dt><dd>${t.fix}</dd>
      </dl>
    </div></details>`).join('');
}
