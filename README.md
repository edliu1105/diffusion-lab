# 扩散实验室(Diffusion Bench)

一间为"真懂"而建的实验室:所有断言由本机训练的真实模型 + 浏览器内实时计算支撑。
交付物是浏览器里的交互式大师课:`app/index.html`。

**在线版(GitHub Pages,随时可开):https://edliu1105.github.io/diffusion-lab/**

## 本地启动

```bash
python serve.py
# 打开 http://localhost:8737(no-store 头,改代码即刷即见)
```

(或在 Claude Code 里直接用预览:`.claude/launch.json` 已配置 `lab`。)
在线版部署:`app/` 目录以 `git subtree split -P app -b gh-pages` 切出并推送,Pages 指向 gh-pages 根。

## 里面有什么

| 章节 | 形态 | 核心 |
|---|---|---|
| §1 地图与优先级 | 叙事 | 四轴生成语法、生产地图(2026-07)、学习优先级 |
| §2 一个场,五件衣服 | **LIVE** | 三个原生方言网络 + 解析真值;五种预测的现场换算 |
| §3 训练循环显微镜 | **LIVE** | 三方言同批数据逐行对比;λ 预算(密度×汇率);loss 地板 |
| §4 预测目标解剖室 | 真张量 | MNIST UNet 的 x_t/x̂₀/ε̂/v̂/û/score 全光谱扫描 |
| §5 采样=数值积分 | **LIVE** | 粒子竞技场、NFE 基准、DDIM≡FM-Euler 与 SDE≡ODE 的现场证明 |
| §6 CFG 分布算术 | **LIVE** | 用零误差解析模型证明 CFG 失真是算术本性;MNIST w 扫描 |
| §7 蒸馏 | **LIVE** | 教师/ReFlow/一步学生;轨迹直化;一步映射的缝 |
| §8 实时视频与世界模型 | 叙事 | 因果化三步手术;误差累积对策;2026-07 现状与差距 |
| §9 心智模型·分歧·拷问 | 叙事 | 5 个共享心智模型、3 个真分歧、10 道深浅对照拷问 |
| §10 故障博物馆 | 真复现 | terminal-SNR 灰底(照 SD1.x 配置真训出来的)+ 值班手册 |
| §11 附录 | LIVE | 速查卡、复现命令、浏览器↔PyTorch 对拍 |

## 复现全部数据

```bash
python exp/common.py          # fp64 恒等式自检(换算、权重、采样器等价)
python exp/toy2d.py           # 2D:eps_ddpm / x0_edm / u_fm / u_fm_cond(各按原生方言训练)
python exp/distill2d.py       # ReFlow + 一步(平均速度)学生
python exp/mnist.py all       # MNIST:健康 FM / 泄漏调度 / 一步学生
python exp/mnist_export.py    # 解剖图集、采样器网格、CFG 扫描、故障展品
python exp/make_fixtures.py   # JS↔PyTorch 对拍数据
python exp/cifar.py           # (bonus)CIFAR-10 彩色条件 FM
```

统一数学核心在 `exp/common.py`(带 fp64 自检),浏览器运行时在 `app/js/{nn,gmm,diff}.js`,
两边用 `app/data/fixtures.json` 逐位对拍(§11.3 一键运行)。

## 设计原则

- 一切核心断言可亲手验证(LIVE 实验或一键证明),没有示意动画;
- 叙事结论标注 事实 / 推断 / 判断;
- 挑战卡先预测后揭晓——揭晓文本里是实测数字。
