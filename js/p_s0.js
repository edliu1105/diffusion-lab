// p_s0.js — how to use the lab.

export async function init(mount) {
  mount.innerHTML = `
  <div class="grid2">
    <div>
      <h4>这是什么</h4>
      <p>一间实验室,不是一篇文章。四个 2D 网络 + 两个蒸馏学生此刻运行在你的浏览器里(还有一套 MNIST UNet 的预渲染真张量)。
      每个核心断言都配一个你能拨动的实验或一段当场跑的证明;每个概念都给三层:<em>直觉</em>(它在干什么)、<em>机制</em>(为什么成立)、<em>实现</em>(张量和数值长什么样)。任何一层撑不住,就当我没讲对。</p>
      <h4>两条路径</h4>
      <p><b>速通(≈2h)</b>:§1 优先级 → §2 场视图(必玩)→ §3 压强图 → §5 竞技场+两个证明 → §9 心智模型与拷问。<br>
      <b>全量(≈1 天)</b>:顺序走,做每一张挑战卡:<strong>先按自己的预测选,再看揭晓</strong>——错了的地方就是钱所在。</p>
    </div>
    <div>
      <h4>标注纪律</h4>
      <p>凡是叙事性结论,标注来源等级:<span class="tag f">事实</span> 公开可查/本机可验证;
      <span class="tag i">推断</span> 由公开信息推得,厂商未确认;<span class="tag j">判断</span> 我的立场,欢迎开火。
      不带标注的技术陈述 = 数学或本仓库实验,当场可复算。</p>
      <h4>徽章</h4>
      <p><span class="live" style="font-family:var(--mono);font-size:9.5px;color:var(--student);border:1px solid var(--student);border-radius:3px;padding:1px 6px">LIVE</span> = 此刻在你浏览器里跑真模型;
      <span class="pre" style="font-family:var(--mono);font-size:9.5px;color:var(--ink3);border:1px solid var(--ink3);border-radius:3px;padding:1px 6px">预渲染</span> = 本机 GPU 跑好的真张量/真样本,非示意图。
      唯一的例外是 §8 的架构示意(没法在本机训一个视频模型),已明确标出;其余一切像素来自真实计算。</p>
      <h4>符号约定(全站统一)</h4>
      <p class="m" style="font-size:13px">x_t = α·x₀ + σ·ε;λ = logSNR = 2·log(α/σ);FM:α=1−t, σ=t。λ→−∞ 纯噪,λ→+∞ 干净。</p>
    </div>
  </div>`;
}
