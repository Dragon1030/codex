# 最终实验结果汇报（2026-11-11）

本目录保存基于四份正式 `metrics.csv` 的最终统计、图表和导师讨论版汇报。PPT 只保留与研究问题直接相关的结果：持续学习是否保留旧任务、顺序微调的遗忘、载体质量与秘密恢复的权衡，以及当前证据的边界。

## 文件

- `ContinualSteg_final_results_2026-11-11.pptx`：7 页学术汇报版 PPT。
- `ContinualSteg_final_results_2026-11-11.pdf`：便于预览和打印的 PDF。
- `figures/`：PPT 中使用的四张图。
- `cleaned/`：从正式运行块重算的逐任务指标和汇总表。
- `experiment_manifest.json`：数据选择、方法映射和实验协议。
- `raw/`：用户上传的原始 CSV，保留两次运行的完整内容。
- `build_report.py`、`slide_layout.py`：重新生成清洗表、图表和 PPT 的脚本。

## 数据选择

用户确认四份 `metrics.csv` 的最后 15 行来自每任务 50 轮、训练和验证均不限制 batch 数的完整实验。每份 CSV 含两个 15 行运行块；本报告统一选择第二个完整块。第一块保留在 `raw/`，不参与最终表格。上传的 `comparison.csv` 与第一块短实验相符，因此只作为审计材料，不覆盖第二块正式运行结果。

方法映射为：`metrics_a.csv=full`、`metrics_b.csv=no_distill`、`metrics_c.csv=adapter_only`、`metrics_d.csv=sequential_finetune`。任务顺序为 `clean → quantized → gaussian → resize → crop`；代码中的 `jpeg` 是 50/255 像素量化，正文使用 quantized 避免误称为真实 JPEG。

## 结果摘要

- 最终秘密恢复宏平均 PSNR：Full CL `18.88 dB`，No distill `19.01 dB`，Adapter only `18.92 dB`，Sequential FT `10.37 dB`。
- 旧任务平均遗忘：Full CL `0.0018 dB`，No distill `0.0027 dB`，Adapter only `0.0013 dB`，Sequential FT `12.3445 dB`。
- Sequential FT 的载体质量更好（cover PSNR `25.68 dB`、SSIM `.840`、APD `9.97`），但旧任务秘密恢复大幅下降，说明不能只用载体质量评价隐写模型。
- Full、No distill、Adapter only 的绝对秘密 PSNR接近；当前一次运行不能单独证明 replay 或 distillation 的独立增益。
- 当前结果证明了“任务矩阵 + 旧任务保留”的工程原型，尚不能宣称已完成原始 HiNet/UDH 结构整合、真实 JPEG 鲁棒性或隐写安全性验证。

## 重新生成

在本目录执行：

```bash
python3 build_report.py
```

需要 `python-pptx`、`matplotlib` 和 `PyMuPDF`。脚本会重写 `cleaned/`、`figures/` 和 PPT；PDF 可用 LibreOffice 转换生成。正式论文复现还应同时保存完整训练命令、随机种子、GPU、数据路径和日志。
