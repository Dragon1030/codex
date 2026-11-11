"""Build the final, data-first advisor deck from the five uploaded CSVs.

Run from this directory after installing python-pptx and matplotlib:
    python build_report.py
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from slide_layout import (  # noqa: E402
    ACCENT, ALERT, FONT, INK, LINE, MUTED, PALE, WHITE,
    MSO_SHAPE, PP_ALIGN as PPT_ALIGN, base, color, line, note, prs, table, text,
)

METHODS = ["full", "no_distill", "adapter_only", "sequential_finetune"]
METHOD_LABELS = {
    "full": "Full CL",
    "no_distill": "No distill",
    "adapter_only": "Adapter only",
    "sequential_finetune": "Sequential FT",
}
TASKS = ["clean", "jpeg", "gaussian", "resize", "crop"]
TASK_LABELS = {"clean": "clean", "jpeg": "quantized", "gaussian": "Gaussian", "resize": "resize", "crop": "crop"}
COLORS = {"full": "234E70", "no_distill": "4F81BD", "adapter_only": "70AD47", "sequential_finetune": "C0504D"}
SOURCE_FILES = {
    "full": ROOT / "raw/metrics_a.csv",
    "no_distill": ROOT / "raw/metrics_b.csv",
    "adapter_only": ROOT / "raw/metrics_c.csv",
    "sequential_finetune": ROOT / "raw/metrics_d.csv",
}


def read_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["stage"] = int(row["stage"])
            for key in ("cover_psnr", "cover_ssim", "secret_psnr", "secret_ssim", "cover_apd"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def split_runs(rows):
    runs = [[]]
    for row in rows:
        if runs[-1] and row["stage"] < runs[-1][-1]["stage"]:
            runs.append([])
        runs[-1].append(row)
    return runs


def select_final(rows, method):
    runs = split_runs(rows)
    if [len(r) for r in runs] != [15, 15]:
        raise ValueError(f"{method}: expected two 15-row runs, got {[len(r) for r in runs]}")
    return runs[-1]


def write_csv(path, rows):
    fields = ["stage", "task", "cover_psnr", "cover_ssim", "secret_psnr", "secret_ssim", "cover_apd"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def final_rows(rows):
    return {row["task"]: row for row in rows if row["stage"] == 4}


def metrics_summary(rows):
    final = final_rows(rows)
    mean_keys = ["cover_psnr", "cover_ssim", "secret_psnr", "secret_ssim", "cover_apd"]
    return {key: sum(final[t][key] for t in TASKS) / len(TASKS) for key in mean_keys}


def forgetting(rows):
    values = {}
    for task in TASKS[:-1]:
        task_rows = [r for r in rows if r["task"] == task]
        values[task] = max(r["secret_psnr"] for r in task_rows) - task_rows[-1]["secret_psnr"]
    return values


def plot_style(ax):
    ax.grid(axis="y", color="#D8DEE3", linewidth=.7, alpha=.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("DejaVu Sans")


def save_final_psnr(data):
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    x = np.arange(len(TASKS))
    width = .19
    for i, method in enumerate(METHODS):
        vals = [data[method][t]["secret_psnr"] for t in TASKS]
        bars = ax.bar(x + (i-1.5)*width, vals, width, label=METHOD_LABELS[method], color="#" + COLORS[method])
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+.12, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [TASK_LABELS[t] for t in TASKS])
    ax.set_ylabel("Secret recovery PSNR (dB)")
    ax.set_title("Final-stage secret recovery quality")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    plot_style(ax)
    fig.tight_layout()
    fig.savefig(ROOT/"figures/final_secret_psnr.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_forgetting(data):
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = np.arange(len(METHODS))
    vals = [sum(data[m].values()) / len(data[m]) for m in METHODS]
    bars = ax.bar(x, vals, color=["#"+COLORS[m] for m in METHODS], width=.62)
    ax.set_yscale("symlog", linthresh=.01)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=12, ha="right")
    ax.set_ylabel("Mean secret-PSNR forgetting (dB; lower is better)")
    ax.set_title("Backward transfer / forgetting on previously learned tasks")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.1 if v else .0012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plot_style(ax)
    fig.tight_layout()
    fig.savefig(ROOT/"figures/forgetting.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_full_matrix(rows):
    matrix = np.full((len(TASKS), len(TASKS)), np.nan)
    for row in rows:
        matrix[row["stage"], TASKS.index(row["task"])] = row["secret_psnr"]
    fig, ax = plt.subplots(figsize=(8.5, 5.1))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="YlGnBu", aspect="auto", vmin=14, vmax=21)
    ax.set_xticks(range(len(TASKS)), [TASK_LABELS[t] for t in TASKS], rotation=18, ha="right")
    ax.set_yticks(range(len(TASKS)), [f"Stage {i}" for i in range(len(TASKS))])
    ax.set_title("Full CL: secret PSNR matrix across stages")
    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            if not math.isnan(matrix[i,j]):
                ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=10,
                        color="white" if matrix[i,j] < 17.5 else f"#{INK}")
    fig.colorbar(im, ax=ax, label="Secret PSNR (dB)")
    fig.tight_layout()
    fig.savefig(ROOT/"figures/full_stage_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_final_quality(summaries):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    labels = [METHOD_LABELS[m] for m in METHODS]
    x = np.arange(len(METHODS))
    specs = [("secret_psnr", "Secret PSNR (dB)", "#234E70"), ("secret_ssim", "Secret SSIM", "#70AD47"), ("cover_apd", "Cover APD (lower)", "#C0504D")]
    for ax, (key, ylabel, c) in zip(axes, specs):
        vals = [summaries[m][key] for m in METHODS]
        bars = ax.bar(x, vals, color=c, alpha=.86)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v + (max(vals)-min(vals))*0.025, f"{v:.2f}", ha="center", fontsize=8)
        plot_style(ax)
    fig.suptitle("Macro-average final-stage quality across five tasks", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(ROOT/"figures/final_quality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# Keep the original files and derive a second, auditable final-run dataset.
data = {}
source_meta = []
for method, source in SOURCE_FILES.items():
    rows = select_final(read_rows(source), method)
    data[method] = final_rows(rows)
    write_csv(ROOT/"cleaned"/f"{method}_metrics.csv", rows)
    source_meta.append({"method": method, "source": str(source.relative_to(ROOT)), "selected": "second 15-row run"})

summaries = {method: metrics_summary([r for r in select_final(read_rows(SOURCE_FILES[method]), method)]) for method in METHODS}
forget = {method: forgetting(select_final(read_rows(SOURCE_FILES[method]), method)) for method in METHODS}
write_rows = []
for method in METHODS:
    for task in TASKS:
        row = dict(data[method][task])
        f = forget[method].get(task, "")
        write_rows.append({"run": method, **row, "forgetting_secret_psnr": f})
with (ROOT/"cleaned/comparison_recomputed.csv").open("w", newline="") as handle:
    fields = ["run", "stage", "task", "cover_psnr", "cover_ssim", "secret_psnr", "secret_ssim", "cover_apd", "forgetting_secret_psnr"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(write_rows)

manifest = {
    "selected_run": "second complete 15-row block in each metrics.csv",
    "confirmed_user_parameters": {"epochs_per_task": 50, "max_train_batches": 0, "max_val_batches": 0},
    "tasks": TASKS,
    "methods": METHODS,
    "source_files": source_meta,
    "discarded_for_final_tables": "first 15-row block in each metrics.csv (retained in raw/)",
    "comparison_upload_used_for_audit_only": "raw/comparison.csv; it describes the earlier short block and is not used for final tables",
}
(ROOT/"experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

save_final_psnr(data)
save_forgetting(forget)
save_full_matrix(select_final(read_rows(SOURCE_FILES["full"]), "full"))
save_final_quality(summaries)

# Build the final compact deck.
s = base("最终实验结果：持续学习保留旧任务，顺序微调发生遗忘", "四种方法 · 五种连续退化 · 每任务50轮 · 使用每份CSV最后15行完整运行", "数据：用户上传的四份metrics.csv；首段15行保留在raw/，不参与最终统计。")
text(s, .72, 2.03, 12.0, .70, "Full CL 的旧任务平均恢复 PSNR：19.93 dB\nSequential FT：9.16 dB；平均遗忘：12.34 dB", 24, True, ACCENT)
text(s, .73, 3.58, 11.75, .48, "核心结论：任务隔离显著保留旧任务，但 Replay / Distillation 的独立增益仍很小。", 20, True)
text(s, .73, 4.35, 11.75, .52, "当前最弱任务仍是 crop：Full CL 恢复 14.67 dB / SSIM 0.233。", 19, fill=ALERT)
text(s, .73, 5.26, 11.75, .85, "汇报边界：这是持续学习机制的工程验证，不是已完成的 HiNet / UDH 原始结构整合，也不是隐写安全性结论。", 17, fill=MUTED)
note(s, "本页数字来自第二个完整15行块。Full CL旧任务(clean/jpeg/gaussian/resize)的最终均值=19.9272dB；Sequential FT=9.1588dB；平均secret-PSNR遗忘分别为0.0018和12.3445dB。Full CL的五任务secret PSNR宏平均为18.8762dB，crop为14.6721dB。‘Full CL’是本项目的冻结共享骨干+任务Adapter+Replay+Distillation实现，不能扩大为原始HiNet或UDH。四份CSV的第一块是短实验/旧结果，已保留但排除。")

s = base("实验协议与数据审计", "同一任务顺序、同一数据路径、同一评估指标；每份CSV含两次15行运行。", "数据审计：每个metrics.csv按stage下降切分为[15,15]；最终PPT统一选第二个完整块。")
table(s, .7, 2.05, [2.65, 3.20, 3.05, 3.10], [
    ["项目", "设置", "方法", "输出"],
    ["任务流", "clean → quantized → Gaussian → resize → crop", "同一5任务顺序", "每阶段回测已见任务"],
    ["训练", "50 epochs / task；batch限制0", "Full / No distill / Adapter only / Sequential FT", "每方法15条矩阵"],
    ["数据", "DIV2K train / val；图像配对", "相同seed与路径（待核日志）", "PSNR / SSIM / APD"],
    ["审计", "首块不用于最终汇报", "第二块作为完整运行", "cleaned/*.csv可复核"],
], row_h=.72, size=15)
text(s, .73, 5.58, 11.6, .48, "comparison.csv 与首块指标不一致，因此重新从各方法 metrics.csv 计算 comparison_recomputed.csv。", 19, True, ALERT)
text(s, .73, 6.23, 11.6, .40, "正式论文仍应保存：命令行、commit、数据列表、seed、GPU和每次运行日志。", 15, fill=MUTED)
note(s, "用户确认四份metrics.csv的后15行来自每任务50轮、训练和验证不限制batch的完整实验。原comparison.csv显示的是旧的11–16dB短实验块，不能覆盖第二块结果；本报告保留原CSV并建立cleaned目录与experiment_manifest.json。metrics.csv没有存储超参数，‘同一seed与路径’只能依据运行协议，正式复现仍需保存完整命令和日志。任务名jpeg在tasks.py中是50/255像素量化，不是真实JPEG压缩；PPT正文称quantized以避免误导。")

s = base("最终秘密恢复：Full CL 与两种隔离消融接近", "按最终阶段五任务逐任务比较；纵轴越高越好。", "图：figures/final_secret_psnr.png；数值：cleaned/comparison_recomputed.csv。")
s.shapes.add_picture(str(ROOT/"figures/final_secret_psnr.png"), Inches(.72), Inches(1.92), width=Inches(8.45))
table(s, 9.42, 2.10, [1.55, 2.05], [
    ["方法", "宏平均PSNR"],
    ["Full CL", f"{summaries['full']['secret_psnr']:.2f}"],
    ["No distill", f"{summaries['no_distill']['secret_psnr']:.2f}"],
    ["Adapter only", f"{summaries['adapter_only']['secret_psnr']:.2f}"],
    ["Sequential FT", f"{summaries['sequential_finetune']['secret_psnr']:.2f}"],
], row_h=.53, size=14, emphasize=(1,))
text(s, 9.42, 5.25, 3.1, 1.10, "Full / No distill / Adapter only 的差异不到0.14 dB。\nSequential FT 的平均值低至10.37 dB。", 17, True)
note(s, "最终阶段宏平均secret PSNR：Full=18.8762；No distill=19.0070；Adapter only=18.9239；Sequential FT=10.3697。Full比No distill低0.1309dB、比Adapter only低0.0478dB，单次运行不能证明蒸馏收益；更像是当前损失连接或随机性问题。逐任务：Full clean20.29/jpeg19.56/Gaussian19.74/resize20.12/crop14.67。Sequential clean8.20/jpeg8.48/Gaussian8.17/resize11.78/crop15.21。Sequential的crop略高但旧任务严重下降，说明适应当前任务与保留旧任务存在权衡。图表不包含隐写分析检测率。")

s = base("持续学习的主要证据：旧任务矩阵与遗忘", "Full CL 的跨阶段恢复 PSNR；空白表示该任务尚未到来。", "左图：figures/full_stage_matrix.png；右图：figures/forgetting.png。")
s.shapes.add_picture(str(ROOT/"figures/full_stage_matrix.png"), Inches(.70), Inches(1.95), width=Inches(6.20))
s.shapes.add_picture(str(ROOT/"figures/forgetting.png"), Inches(6.98), Inches(2.05), width=Inches(5.65))
text(s, .78, 6.20, 11.8, .50, "Full CL旧任务平均遗忘 0.0018 dB；Sequential FT 12.3445 dB。", 20, True, ACCENT)
note(s, "遗忘按每个旧任务的best secret PSNR - 最终secret PSNR计算；crop没有后续阶段，不纳入平均遗忘。Full的clean/jpeg/gaussian/resize遗忘分别约0、0、0.007、0dB；Sequential分别12.261、12.439、13.440、11.239dB。这个差异支持‘冻结/隔离能保留旧路径’这一工程结论。它不单独证明Replay或Distillation，因为当前实现中旧任务路径被冻结，且蒸馏分支复用旧任务路径；独立贡献仍需修正训练损失后重做。")

s = base("载体质量的变化：不要只看载体PSNR", "Sequential FT 载体指标更好，但秘密恢复更差；隐写目标是多目标权衡。", "图：figures/final_quality.png；五任务宏平均，APD越低越好。")
s.shapes.add_picture(str(ROOT/"figures/final_quality.png"), Inches(.73), Inches(1.90), width=Inches(8.45))
table(s, 9.42, 2.08, [1.55, 2.05], [
    ["方法", "载体PSNR / SSIM / APD"],
    ["Full CL", "23.33 / .712 / 11.89"],
    ["No distill", "23.25 / .712 / 12.13"],
    ["Adapter only", "23.24 / .707 / 12.02"],
    ["Sequential FT", "25.68 / .840 / 9.97"],
], row_h=.58, size=13, emphasize=(1,))
text(s, 9.42, 5.38, 3.15, 1.00, "Sequential FT 让载体更接近原图，却牺牲了旧任务秘密恢复。\n这不是单指标越高越好。", 17, True)
note(s, "最终宏平均cover PSNR/SSIM/APD：Full=23.3303/.7121/11.8925；No distill=23.2529/.7122/12.1326；Adapter only=23.2411/.7066/12.0226；Sequential=25.6834/.8399/9.9749。Sequential的载体更好，但secret PSNR宏平均10.3697，说明当前loss/任务路径存在隐蔽性-恢复性与持续保留权衡。cover PSNR仅针对攻击前载密；未评估检测器/安全性。正式报告应同时给载体质量与秘密恢复质量，不能用cover质量替代隐写任务目标。")

s = base("本轮实验能支持的结论与不能支持的结论", "把结果写成可审查的研究结论，避免把机制设计等同于机制有效。", "证据范围：四份最终15行CSV；没有真实HiNet/UDH对照、显著性检验或可视化样本。")
table(s, .70, 2.02, [5.75, 6.25], [
    ["可以说", "暂时不能说"],
    ["顺序任务评估矩阵已跑通；隔离方案旧任务遗忘小。", "Full CL 已经优于原始HiNet/UDH。"],
    ["Sequential FT 在旧任务上明显退化，说明遗忘基线有效。", "Replay 或 Distillation 已被单独证明有效。"],
    ["crop 是当前恢复质量瓶颈，载体/秘密存在权衡。", "模型具有隐写安全性或对真实JPEG鲁棒。"],
    ["当前实现可作为持续学习工程原型。", "已经完成原始HiNet/UDH结构级整合。"],
], row_h=.73, size=16, emphasize=(1,))
text(s, .73, 5.77, 11.6, .56, "当前最有价值的结果：遗忘基线与任务矩阵，而不是Full CL的单次绝对PSNR。", 20, True, ACCENT)
note(s, "‘可以说’均可由CSV直接支持；‘不能说’是研究边界。尤其Full、No distill、Adapter only最终指标非常接近，且只有一次seed，不能做统计显著性声明。当前的模型是紧凑UDH风格CNN，原始HiNet可逆网络与UDH U-Net尚未替换进来。论文应先将loss/Prompt/Replay路径修正，再在统一协议下以原始UDH、HiNet为基线。")

s = base("导师讨论：只确认三个下一步决策", "若时间有限，优先修正实验有效性，再扩大模型和攻击类型。", "本页是讨论清单，不是已完成结果。")
text(s, .75, 2.00, 11.7, .49, "1  持续学习场景：恢复端是否允许已知退化标签？", 21, True, ACCENT)
text(s, 1.10, 2.62, 11.2, .54, "若未知，固定嵌入器，恢复端增加未知任务路由实验；当前CSV只测已知task_id。", 17)
text(s, .75, 3.45, 11.7, .49, "2  方法主线：先做可靠回放，还是先保留独立Adapter？", 21, True, ACCENT)
text(s, 1.10, 4.07, 11.2, .54, "把旧样本经过旧攻击的恢复损失接到当前可训练模块；重新比较 No replay / Replay / Distill。", 17)
text(s, .75, 4.90, 11.7, .49, "3  原始基线：先复现 UDH，再接入 HiNet？", 21, True, ACCENT)
text(s, 1.10, 5.52, 11.2, .54, "同一DIV2K、同一退化、同一逐图指标和同一随机种子；最终报告恢复、遗忘、参数和存储。", 17)
note(s, "建议导师确认场景、机制和基线顺序。对未知退化，当前模型的route()原型只根据prototype特征路由，尚未用分类准确率或未知检测验证。对Replay/Distillation，当前train.py的旧路径蒸馏不能代表当前任务适配器真正受约束；建议新增旧样本重放的真实cover/secret/attack损失，并将共享可训练模块作为可蒸馏目标。对原始基线，先将UDH的HidingUNet/RevealNet接入适配器机制，保持HiNet的可逆 forward/reverse一致性，再做架构对比。")

out = ROOT / "ContinualSteg_final_results_2026-11-11.pptx"
prs.save(out)
print(f"saved {len(prs.slides)} slides to {out}")
