// p_s8.js — §8: real-time video & world models. Narrative + precise SVG schematics.

import { lamColor } from './charts.js';

function frameRect(x, y, lam, w = 26, h = 34) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${lamColor(lam)}" opacity="0.9"/>`;
}

export async function init(mount) {
  // three-panel schematic: offline / diffusion forcing / streaming
  const P = (x0, title, frames, arrows, note) => {
    let s = `<g transform="translate(${x0},0)"><text x="0" y="14" fill="var(--ink2)" font-size="12" font-family="Consolas">${title}</text>`;
    frames.forEach((lam, i) => { s += frameRect(i * 32, 26, lam); });
    s += arrows + `<text x="0" y="92" fill="var(--ink3)" font-size="10.5" font-family="DengXian">${note}</text></g>`;
    return s;
  };
  const arrBi = (() => {
    let a = '';
    for (let i = 0; i < 5; i++) for (const j of [i - 1, i + 1]) {
      if (j < 0 || j > 5) continue;
      a += `<path d="M ${i * 32 + 13} 70 Q ${(i + j) * 16 + 13} 82 ${j * 32 + 13} 70" stroke="var(--ink3)" fill="none" stroke-width="0.8" opacity="0.7"/>`;
    }
    return a;
  })();
  const arrCausal = (() => {
    let a = '';
    for (let i = 1; i < 6; i++) for (let j = Math.max(0, i - 2); j < i; j++) {
      a += `<path d="M ${j * 32 + 13} 70 Q ${(i + j) * 16 + 13} ${78 + (i - j) * 4} ${i * 32 + 13} 70" stroke="var(--ink3)" fill="none" stroke-width="0.8" opacity="0.7" marker-end="url(#ah)"/>`;
    }
    return a;
  })();

  mount.innerHTML = `
  <p>先看清楚"为什么离线视频模型天生不实时"。Sora/Veo 形状的模型把整段视频压成一个 3D latent 块,时空双向注意力,<strong>所有帧在同一个噪声等级上一起去噪几十步</strong>。三个结构性障碍:第 1 帧要看第 120 帧才能定稿(非因果);延迟=整段生成时间;用户动作没有插口。把它变成 24fps 可交互,是三步手术:</p>

  <svg viewBox="0 0 720 100" style="width:100%;max-width:760px;display:block;margin:10px 0 4px">
    <defs><marker id="ah" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 z" fill="var(--ink3)"/></marker></defs>
    ${P(0, 'a. 离线:整段·双向·同噪声级', [0, 0, 0, 0, 0, 0], arrBi, '全帧同 λ,一起走完几十步')}
    ${P(250, 'b. Diffusion Forcing:每帧自己的 λ', [9, 7, 4, 0.5, -3, -7], arrCausal, '过去干净、未来更噪;帧间因果注意')}
    ${P(500, 'c. 流式:KV cache + 帧内少步', [9, 9, 9, 9, 2, -7], arrCausal, '历史=缓存的世界状态;新帧 1–4 步出图')}
  </svg>
  <p class="small">b 图是关键一步的精确表达:训练时给<b>每一帧独立采样噪声等级</b>(色=λ,紫噪金净)。这一个改动同时买到三件东西:模型学会"以干净历史为条件生成带噪未来"(=自回归的定义)、学会容忍历史里的瑕疵(把上下文当"微噪"处理→对抗误差累积)、推理时可以滑动窗口逐帧推进。<span class="tag f">事实(Diffusion Forcing, 2024)</span></p>

  <h3>三步手术的谱系(2024 → 2026-07)</h3>
  <table class="data wide">
    <tr><th>步骤</th><th>代表工作</th><th>核心动作</th><th>没它会怎样</th></tr>
    <tr><td><b>① 架构因果化</b></td><td>CausVid、各家流式底座</td><td>帧内双向、帧间因果(block-causal);KV cache 成为"世界状态"</td><td>每帧都要重算全段注意力,无法流式</td></tr>
    <tr><td><b>② 训练自回归化</b></td><td>Diffusion Forcing(2024)、Rolling/滑窗族</td><td>逐帧独立 λ + 因果条件;历史噪声增强</td><td>exposure bias:训练只见真历史,推理全是自己的输出,10 秒内漂移</td></tr>
    <tr><td><b>③ 时间轴蒸馏</b></td><td>Self-Forcing(2025)→ Causal Forcing(ICML 2026)→ Causal-rCM(2026-06,开源统一配方)</td><td>帧内几十步→1–4 步(DMD/rCM 系);训练时 rollout 自己的生成当上下文</td><td>单帧 500ms+,凑不出 24fps</td></tr>
  </table>
  <p>2026 年这条线上最有信息量的一场架构官司:Self-Forcing 系用<b>双向教师</b>蒸馏自回归学生;Causal Forcing 指出这从原理上就错——双向教师的去噪映射依赖未来,拿它监督因果学生违反 frame-level injectivity,于是先做一个自回归教师(ODE 初始化)再做分布匹配,质量与动态性双升。<span class="tag f">事实(检索 2026-07)</span> 这场官司值得记住的通用教训:<b>蒸馏的前提是教师和学生解同一个问题;条件结构不同就不是同一个问题</b>。</p>

  <h3>误差累积:对策谱系</h3>
  <table class="data wide">
    <tr><th>对策</th><th>机制</th><th>代价</th></tr>
    <tr><td>上下文噪声增强</td><td>训练时把历史帧加微噪→自生成瑕疵落在训练分布内(等价于把"我可能错了"写进条件)</td><td>历史信息被轻度模糊,精细一致性略降</td></tr>
    <tr><td>Self-Forcing 类后训练</td><td>直接在自己的 rollout 上训练(train=test)</td><td>训练成本高;要防坍缩需配分布匹配损失</td></tr>
    <tr><td>注意力锚 / 帧 sink</td><td>少数"锚帧"常驻 KV(DySink 2026 动态选锚)</td><td>锚选错=错误也被锚定</td></tr>
    <tr><td>长时记忆 / 空间记忆</td><td>Genie 3 的分钟级一致性;回访同一地点不变(机制未公开,推断为长上下文+显式记忆混合)<span class="tag i">推</span></td><td>记忆容量与算力线性冲突</td></tr>
  </table>

  <h3>现状快照(2026-07)与差距清单</h3>
  <p><b>现状</b>:Genie 3(2026-01 公测):文本→可走动世界,24fps/720p、数分钟一致性、promptable events;Decart Oasis 证明了纯神经游戏世界的可行性;Vidu S1 做到实时交互视频;NVIDIA Cosmos 把"世界基础模型"卖给机器人/自驾管线;World Labs Marble 走显式 3D 的另一条路。<span class="tag f">事实(检索 2026-07)</span></p>
  <p><b>到"真实世界模拟器"还差什么</b>(按我判断的难度排序,难→易):<span class="tag j">判断</span></p>
  <ol>
    <li><b>持久状态。</b>分钟级→小时级;离开房间十分钟回来,东西还在原位。这是记忆架构问题,不是画质问题——当前最硬的伤。</li>
    <li><b>因果接口的 grounding。</b>"我推了箱子"和"箱子恰好动了"在视频先验里难以区分;需要海量(动作→后果)配对数据,而互联网视频没有动作标注。机器人/游戏遥测是稀缺资产——这也是 NVIDIA 押 Cosmos 的逻辑。</li>
    <li><b>物理的长链一致性。</b>单步物理已经很像样(Sora 2 的卖点),守恒律级别的长时物理仍靠运气。</li>
    <li><b>算力经济学。</b>24fps × 大模型前向 × 并发用户;蒸馏(③)+稀疏注意力(Light Forcing 类)+小帧模型是必需品,不是优化。</li>
    <li><b>评测真空。</b>没有公认的"世界性"基准(一致性/可控性/反事实),领域仍在用 demo 对轰。</li>
  </ol>
  <div class="callout">立场:视频生成与世界模型正在合流成一个物种——<b>视频先验 + 因果接口 + 记忆</b>。图像/视频侧学到的一切(本实验室 §2–7)原封不动地是它的地基:世界模型的每一帧仍然是"一个 E[x₀|x_t] 加一个积分器",只是条件里多了历史与动作,预算里多了"每帧 41ms"这条死线。往这个方向押注的技能组合:蒸馏 + KV/记忆系统 + 交互数据管线。<span class="tag j">判断</span></div>
  <p class="small">2026-07 检索来源(标注「检索」的事实):Genie 3 <a href="https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/">DeepMind 官方</a>;
  Causal Forcing <a href="https://arxiv.org/abs/2602.02214">arXiv:2602.02214</a>(ICML 2026)与 <a href="https://github.com/thu-ml/Causal-Forcing">thu-ml 代码</a>;
  Causal-rCM <a href="https://arxiv.org/abs/2606.25473">arXiv:2606.25473</a>;Vidu S1 <a href="https://arxiv.org/pdf/2607.03118">arXiv:2607.03118</a>;
  DySink <a href="https://arxiv.org/pdf/2605.21028">arXiv:2605.21028</a>;Light Forcing <a href="https://arxiv.org/pdf/2602.04789">arXiv:2602.04789</a>;
  MeanFlow <a href="https://openreview.net/forum?id=uWj4s7rMnR">OpenReview</a> 及 RAE 变体(CVPR 2026);rCM <a href="https://arxiv.org/html/2510.08431">arXiv:2510.08431</a>;
  视频产品格局综述见 aimlapi/atlascloud/wavespeed 2026 对比文。</p>`;
}
