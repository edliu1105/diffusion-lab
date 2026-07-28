// p_s1.js — §1: the map. Four axes generate the field; priorities; who ships what.

export async function init(mount) {
  mount.innerHTML = `
  <h3>1.1 领域的生成语法:一个对象 × 六根轴</h3>
  <p>这个领域的论文看起来有一千篇,自由度只有六根。学会这张表,任何新名词出现时自动落位——它必然是在某根轴上拨了一格:</p>
  <table class="data wide">
    <tr><th>轴</th><th>问题</th><th>选项谱系(左=老,右=新)</th><th>2026 生产默认</th></tr>
    <tr><td><b>A 路径</b></td><td>α(λ),σ(λ) 怎么走、端点在哪</td><td class="m">VP-linear → VP-cosine → VE → FM-linear(+分辨率 shift)</td><td class="m">FM-linear + shift</td></tr>
    <tr><td><b>B 参数化</b></td><td>网络吐哪件衣服</td><td class="m">ε → x₀ → v → EDM preconditioning → u</td><td class="m">u(或 EDM 式 preconditioning)</td></tr>
    <tr><td><b>C λ 预算</b></td><td>训练把梯度花在哪段噪声</td><td class="m">uniform-t → SNR 权重族 → lognormal σ → logit-normal t → EDM2 可学习</td><td class="m">logit-normal(±shift)</td></tr>
    <tr><td><b>D 积分器</b></td><td>推理怎么走回去</td><td class="m">ancestral → DDIM → Heun/DPM-Solver++/UniPC → 蒸馏(1–8 步)</td><td class="m">蒸馏 or 二阶 ODE + 少量 churn</td></tr>
    <tr><td><b>E 空间</b></td><td>在哪个表示上做生成</td><td class="m">像素 → VAE latent → 更压缩的 latent / RAE(表征自编码器)</td><td class="m">VAE latent(→RAE 苗头)</td></tr>
    <tr><td><b>F 条件</b></td><td>怎么听话</td><td class="m">CFG → interval/rescale → autoguidance → 蒸馏进权重</td><td class="m">CFG 蒸馏 + interval</td></tr>
  </table>
  <p><b>30 秒读论文协议</b>:①它动了哪根轴?②给的证据是"该轴上的对照"还是"顺便重训了别的轴"(消融干净吗——90% 的 formulation 论文死在这,见 §3)?③动机是新场景(实时/长视频/编辑)还是老场景刷点?
  三问下来,决定精读/略读/跳过。<span class="tag j">判断</span></p>

  <h3>1.2 学习优先级:什么值得长进肌肉</h3>
  <table class="data wide">
    <tr><th>梯队</th><th>内容</th><th>为什么</th></tr>
    <tr><td><b>Ⅰ 肌肉记忆</b></td>
      <td>统一对象 E[x₀|x_t] 与五衣换算(§2/§4);λ 预算观(§3);FM 方言全套(训练+采样);采样=积分 + CFG 算术(§5/§6)</td>
      <td>这四样是"生成语法"本身。掌握后,DDPM/EDM/RF 的一切公式都变成推论,不用背。</td></tr>
    <tr><td><b>Ⅱ 原理级</b></td>
      <td>EDM preconditioning 与 σ 网格的设计逻辑;EDM2 的权重自学习;DPM-Solver++/UniPC 的 exponential-integrator 思想;蒸馏四路线(§7);zero-SNR 类战伤(§10);视频的因果化(§8)</td>
      <td>做系统决策、读代码、排障需要;但细节可查表,不必默写。</td></tr>
    <tr><td><b>Ⅲ 知道存在</b></td>
      <td>离散扩散(文本/代码侧有生命力)、Schrödinger bridge、cold/blurring diffusion、测度论级的 score-SDE 理论</td>
      <td>图像/视频主干用不上;需要时再进。别在这里练级。<span class="tag j">判断</span></td></tr>
  </table>
  <p>一句话理由:<strong>生产主干已经收敛</strong>——latent 空间 + transformer + 直线路径 flow matching + 蒸馏加速 + CFG 工程。方言差异(DDPM vs EDM vs FM)已被证明是权重曲线差异(§3 的实验),所以把力气花在"轴"而不是"方言"上。<span class="tag j">判断,证据在 §2–3</span></p>

  <h3>1.3 生产地图(2026-07 快照)</h3>
  <p>只列真正在收钱或定义 SOTA 的。佐证等级:<span class="tag f">F=官方技报/论文/代码</span> <span class="tag i">推=社区一致推断,官方未证实</span></p>
  <table class="data wide">
    <tr><th>产品/模型</th><th>底层配方(轴 A–F)</th><th>佐证</th></tr>
    <tr><td><b>FLUX.2</b>(BFL,32B,2025-11)</td><td>latent + MMDiT 系 + <b>rectified flow</b>;guidance 蒸馏(w 做输入);schnell 版 timestep 蒸馏</td><td><span class="tag f">F</span></td></tr>
    <tr><td><b>SD3 / 3.5</b>(Stability)</td><td>RF + logit-normal + 分辨率 shift + MMDiT——生产级 RF 的公开配方书</td><td><span class="tag f">F</span></td></tr>
    <tr><td><b>Midjourney v7</b>(2026-03)</td><td>不公开;社区共识为 latent diffusion 系,壁垒在数据/美学后训练而非 formulation</td><td><span class="tag i">推</span></td></tr>
    <tr><td><b>GPT-4o / gpt-image 系</b>(OpenAI)</td><td>自回归 token 主干 + 扩散式解码;掀起 AR-图像复兴的那一枪(2025-03)</td><td><span class="tag i">推(官方仅给系统卡级描述)</span></td></tr>
    <tr><td><b>Imagen 4 / Nano Banana 系</b>(Google)</td><td>latent diffusion;与 Gemini 栈融合,编辑/一致性为卖点</td><td><span class="tag i">推</span></td></tr>
    <tr><td><b>HunyuanImage 3.0</b>(腾讯,开源)</td><td>AR 框架 + 扩散解码的统一多模态——混合派的公开样本</td><td><span class="tag f">F</span></td></tr>
    <tr><td><b>Sora 2</b>(OpenAI,<span class="deadline">已停服</span>)</td><td>时空 patch DiT 扩散 + 原生音频,曾是物理真实感标杆;2026-03-24 官宣停服、04-26 关闭应用、API 于 09-24 日落。技术遗产(spacetime patches 的词汇与物理观感标准)仍在全行业流通。教训:<b>产品名 ≠ 技术路线名,产品节奏可以快过模型生命周期</b></td><td><span class="tag f">F(停服时间线,检索 2026-07;此行曾误标为在役,经同行文档比对后订正)</span></td></tr>
    <tr><td><b>Veo 3.1</b>(Google)</td><td>latent 视频扩散 + 原生 48kHz 音频,4K;当前综合口碑第一</td><td><span class="tag f">F/推混合</span></td></tr>
    <tr><td><b>Kling 3.0/O3</b>(快手)、<b>Seedance 2.0</b>(字节)</td><td>DiT 视频扩散系;Kling 走性价比+动作控制</td><td><span class="tag i">推</span></td></tr>
    <tr><td><b>Wan 2.x / Hunyuan Video</b>(开源)</td><td>开源视频 FM/扩散基座——想读真代码从这进</td><td><span class="tag f">F</span></td></tr>
    <tr><td><b>Genie 3</b>(DeepMind,2026-01 公测)</td><td>帧级自回归世界模型,24fps/720p/分钟级一致性,提示即世界</td><td><span class="tag f">F(能力)/推(架构细节)</span></td></tr>
    <tr><td><b>NVIDIA Cosmos</b></td><td>世界基础模型平台(机器人/自驾向),扩散+AR 双线</td><td><span class="tag f">F</span></td></tr>
  </table>
  <p class="small">检索时间 2026-07(来源清单在交付说明里)。表的正确读法不是记型号,而是看列二:<b>全部落在同一个配方空间里</b>——差异化都发生在数据、后训练、产品线,formulation 本身已不是竞争维度。<span class="tag j">判断</span></p>

  <h3>1.4 源头文献(带立场的书单)</h3>
  <ul>
    <li><b>必读三篇</b>:Karras et al. 2022(EDM,"把方言拆成轴"的原点,本实验室的方法论来源);Kingma & Gao 2023(所有目标=加权 ELBO,§3 换算表的定理版);SD3 技报 2024(生产级 RF 的完整配方)。</li>
    <li><b>值得精读</b>:DDIM(2020,把采样从链里解放出来);EDM2(2024,权重自学习+架构卫生学);Flow Matching 原文 + Lipman 2024 讲义(干净的数学);DMD2 与 sCM(2024,蒸馏两大路线的成熟形态);MeanFlow(2025,一步生成从零训练);Diffusion Forcing → Self-Forcing → Causal Forcing(2024→2026,视频因果化的完整弧线)。</li>
    <li><b>历史意义大于当下实用</b>:DDPM 2020、Score-SDE 2021(读导言与图 1 即可,推导可跳);LDM 2022(latent 一页就懂,剩下是工程)。</li>
    <li><b>大胆跳过</b>:绝大多数"新 schedule/新 sampler 提点 0.x FID"论文;二手 survey;任何不做 λ-预算对照的 formulation 比较。<span class="tag j">判断</span></li>
  </ul>`;
}
