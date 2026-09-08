# 自动研究的一手文献核查

核查日期：2026-09-05。这里区分论文结论、官方实现事实与本项目的推断；自动指标不能替代“本人觉得像不像”的回答。

| 一手来源 | 实际支持什么 | 对本项目的用法与边界 |
|---|---|---|
| [AdaFace，CVPR 2022；官方代码](https://github.com/mk-minchul/AdaFace) | 根据图像质量调整训练损失，改善人脸识别；官方提供 WebFace4M 的 IR50 权重，要求 112×112、BGR、均值和标准差均为 0.5。 | 增加不同训练方案的识别器，检验结论是否依赖 LVFace。它仍然测识别表征，不能据此宣称本人更满意。检测器共用 SCRFD 是本项目控制变量的选择，并非 AdaFace 原始示例的 MTCNN。 |
| [Face2Diffusion，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shiohara_Face2Diffusion_for_Fast_and_Editable_Face_Personalization_CVPR_2024_paper.html) | 组合多尺度身份编码、表情指导和去噪正则，改善身份保持与可编辑性的取舍。 | 支持把“身份”和“表情响应”分别检查；不支持把所有失真归因于编码器最后一层。 |
| [PhotoMaker 官方实现](https://github.com/TencentARC/PhotoMaker) | 支持多参考照片及 SDXL 基础模型；官方将 V2 发布说明与 CVPR 2024 的 PhotoMaker 论文分开。 | V2 的具体路径以锁定代码为准，不能把 V1 论文中的全部训练细节直接套给 V2；官方“多图改善身份”的建议也不是本项目的人评结果。 |
| [PuLID 论文](https://arxiv.org/abs/2404.16022)、[官方代码](https://github.com/ToTheBeginning/PuLID) | 用对比对齐及身份损失，在插入身份时尽量减少对原模型行为的扰动。 | 为第二生成系统提供依据。与 PhotoMaker 的整套系统比较不能单独证明某个注入模块造成差异；v1.1 的实际参数仍需记录。 |
| [F-Bench，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_F-Bench_Rethinking_Human_Preference_Evaluation_Metrics_for_Benchmarking_Face_Generation_ICCV_2025_paper.html) | 建立包含多维人类评分的人脸生成评测，分别研究质量、真实性、身份与文字对应。 | 支持分开评估这些维度。论文中的人类总体偏好不能代替本项目尚未收集的本人评分；本轮不采用其模型评分充当人评。 |
| [WithAnyone，作者论文](https://arxiv.org/abs/2510.14975) | 讨论重建式训练产生参考脸复制倾向，以及身份相似度与自然变化之间的取舍。 | 提醒我们：距离参考更近不自动等于更好。本轮只报告连续参考距离和精确重复，不从 LPIPS 阈值推造“复制率”。 |
| [LPIPS 官方实现](https://github.com/richzhang/PerceptualSimilarity) | 学习图像的感知距离；AlexNet v0.1 输入为 RGB、范围 −1 到 1。 | 全图距离受背景、姿态影响，所以同时报告固定人脸裁切；LPIPS 不是专门的身份或本人 likeness 指标。 |
| [MediaPipe 官方说明](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | 输出面部关键点、52 个表情系数及从标准脸到检测脸的变换矩阵。 | 连续微笑系数用于检查指令是否改变表情。矩阵角度依赖坐标约定，不等于经过相机标定的真实头部角度。 |
| [Diffusers 0.29.2 AutoencoderKL 文档](https://huggingface.co/docs/diffusers/v0.29.2/en/api/models/autoencoderkl) | VAE 负责图像与潜变量之间的转换；扩散模型使用的潜变量存在缩放约定，分块编解码另有配置。 | 确定性重建使用 posterior mode，直接解码原始 VAE 潜变量。它检查基础编解码误差，不能与其他路径误差相加来“分账”。 |

本项目据此提出的**待检验解释**是：参考照片提供的信息、识别表征保留的信息、生成模型接受的信息，以及最终画出来的细节，未必一致。辨认出“属于谁”可能已经成功，但本人依赖的脸部比例、表情习惯或细节仍可能不足。文献让这些解释值得检验；最终判断仍需要独立真实照片参照，以及分别来自本人和陌生人的盲评。
