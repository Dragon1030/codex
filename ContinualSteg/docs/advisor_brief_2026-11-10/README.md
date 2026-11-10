# ContinualSteg 导师讨论材料

- [可编辑 PowerPoint（8页）](ContinualSteg_advisor_brief.pptx)
- [PDF 预览](ContinualSteg_advisor_brief.pdf)
- [汇报所用原始指标](metrics_source.csv)
- [PPT 生成脚本](build_slides.py)

白底、黑字、单一强调色；无独立封面、目录、动画或致谢页。表格、模型流程和文字均可编辑。每页备注保存补充解释与证据，便于一对一讨论。

## 页序

1. 研究问题、当前完成情况及本次讨论重点。
2. 任务流、网络结构与冻结策略：持续学习实际体现在哪里。
3. 五任务最终指标：载密质量、恢复质量、裁剪瓶颈。
4. 跨阶段恢复 PSNR 矩阵及旧任务保留的边界。
5. 短消融的解释：当前回放蒸馏为何不能证明独立贡献。
6. Prompt 初始化、真实退化、指标统计三个具体问题。
7. 修订后最小对照实验计划（尚未完成的实验）。
8. 与导师确认场景、方法主线和原始基线的先后顺序。

## 数据与结论来源

- `metrics_source.csv` 原样来自用户上传的 `metrics [2].csv`，包含5阶段、15条评估记录。第3页取最终阶段，第4页取全部阶段，数值由脚本读取计算，非人工构造。
- 数据规模、分辨率、50轮/任务来自会话中的服务器命令和运行记录。CSV本身不记录超参数，正式复现实验仍应保存参数快照。
- 第5页的5轮短消融区间来自用户终端截图的四组结果，不是完整正式消融CSV，不能用于论文方法排名。
- 代码审阅基于仓库版本 `f60e05e` 的 `ContinualSteg/train.py`、`continual_steg/models.py`、`tasks.py`、`metrics.py`、`replay.py`。这次只生成讨论材料，没有修改训练实现或声称完成新的训练。
- 本次没有可供推理的权重和验证图片，因此不包含生成的载密/恢复效果图。计划中的六联图必须由真实checkpoint和固定验证样本生成。

## 对此前实验解释的修正

旧任务近零遗忘主要符合冻结共享骨干与旧路径的设计。蒸馏分支通过已冻结的旧任务路径计算，未连接当前任务的可训练参数；`no_distill` 也没有单独的旧样本监督回放损失。因此现有三组不能证明Replay/Distillation的有效性。正式实验应先修正损失与参数的连接，再比较梯度、性能和开销。

当前Prompt向量与线性投影权重同时零初始化，按照链式求导，两者均无法获得非零梯度，bias仍可更新。第6页提出梯度与参数变化检查；这不是已经执行过的GPU实测。

`jpeg`任务实际为像素量化而非真实JPEG；当前PSNR先根据批内整体MSE计算，再对batch等权平均，不能直接当成逐图PSNR平均值；Gaussian评估未固定逐图噪声。论文对照应统一这些口径。

## 参考来源

- HiNet，ICCV 2021：[论文](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html)
- UDH，NeurIPS 2020：*UDH: Universal Deep Hiding for Steganography, Watermarking, and Light Field Messaging*（用户此前提供的论文和原始项目）。
- RwF：[Restoring-Without-Forgetting](https://github.com/AlifAshrafee/Restoring-Without-Forgetting)
- AIM：[Adapting Image Models](https://adapt-image-models.github.io/)；借鉴轻量适配思路，不宣称复现其视频任务。

## 重新生成

安装 `python-pptx` 后，在本目录执行：

```bash
python build_slides.py
```

脚本使用 `Noto Sans CJK SC` 中文字体。PDF由LibreOffice从PPTX导出，以便未安装该字体的电脑直接预览。导出示例：

```bash
soffice --headless --convert-to pdf --outdir . ContinualSteg_advisor_brief.pptx
```
