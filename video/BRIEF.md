# BRIEF

workflow: general-video
flow: autonomous

## Deliverable
16:9 1920×1080 教学视频(~15 min),中文 TTS 旁白 + 真计算动画,融合两份教学网页
(本仓库 Diffusion Lab + 同行单文件版)的最优内容。观众:资深工程师(本任务书作者)。

## Non-negotiables
- 动画为真实计算:场景直接运行本仓库训练的模型(权重内联)、解析 GMM、经验贝叶斯 MNIST 去噪器、
  Python 预训练/预烘焙的漂移实验数据。禁止装饰性假动画。
- 视觉系统沿用 Diffusion Lab:深炭底、λ 光谱(紫→金)、方言配色(DDPM 蓝/EDM 琥珀/FM 玫红/学生青)。
- 旁白:edge-tts zh-CN-YunxiNeural;每段一个 mp3,场景时长由音频时长驱动。
- 十幕:Hook → 一根轴 → 五种读数 → 训练显微镜 → 无网络生成 → 采样=积分 → CFG → 蒸馏 → 时间轴/世界模型 → 定稿。

## Approval
autonomous:lint/check 通过 + 抽帧自检后直接渲染交付(用户异步验收)。
