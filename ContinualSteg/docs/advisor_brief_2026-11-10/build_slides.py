"""Generate an editable, minimal advisor discussion deck from the supplied CSV.

Requires python-pptx. Run from any directory: python build_slides.py
"""
from pathlib import Path
import csv
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
FONT = "Noto Sans CJK SC"
INK = "20262C"
MUTED = "606870"
ACCENT = "234E70"
LINE = "D8DEE3"
PALE = "F3F6F8"
ALERT = "8A442E"
WHITE = "FFFFFF"

prs = Presentation()
prs.slide_width = Inches(13.333333)
prs.slide_height = Inches(7.5)
prs.core_properties.title = "基于持续学习的鲁棒图像信息隐藏：导师讨论"
prs.core_properties.subject = "ContinualSteg 当前实现、初步结果、机制诊断与实验决策"
prs.core_properties.author = "ContinualSteg"
prs.core_properties.keywords = "ContinualSteg,持续学习,图像隐藏,导师讨论"


def color(value):
    return RGBColor.from_string(value)


def fmt(run, size, bold=False, fill=INK):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color(fill)
    prop = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        elem = prop.find(tag, prop.nsmap)
        if elem is None:
            elem = OxmlElement(tag)
            prop.append(elem)
        elem.set("typeface", FONT)


def text(slide, x, y, w, h, content, size=19, bold=False, fill=INK, align=None):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(content.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(0)
        p.space_after = Pt(6)
        p.line_spacing = 1.12
        if align is not None:
            p.alignment = align
        r = p.add_run()
        r.text = line
        fmt(r, size, bold, fill)
    return shape


def line(slide, x, y, w, fill=LINE):
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x+w), Inches(y))
    shape.line.color.rgb = color(fill)
    shape.line.width = Pt(0.75)


def base(title, subtitle, source):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(WHITE)
    text(slide, .65, .48, 12.0, .6, title, 27, True)
    text(slide, .67, 1.17, 12.0, .45, subtitle, 14, fill=MUTED)
    line(slide, .65, 1.69, 12.0)
    line(slide, .65, 6.96, 12.0)
    text(slide, .67, 7.07, 11.35, .24, source, 9, fill=MUTED)
    text(slide, 12.1, 7.05, .52, .27, f"{len(prs.slides):02d}", 11, fill=MUTED, align=PP_ALIGN.RIGHT)
    return slide


def note(slide, content):
    slide.notes_slide.notes_text_frame.text = content


def band(slide, y, content, size=19, tone=ACCENT):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.65), Inches(y), Inches(12), Inches(.65))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(PALE)
    shape.line.fill.background()
    text(slide, .86, y+.15, 11.6, .42, content, size, True, tone)


def table(slide, x, y, widths, rows, row_h=.52, size=16, emphasize=()):
    shape = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h*len(rows)))
    tab = shape.table
    for i, width in enumerate(widths):
        tab.columns[i].width = Inches(width)
    for i, row in enumerate(rows):
        tab.rows[i].height = Inches(row_h)
        for j, value in enumerate(row):
            cell = tab.cell(i, j)
            cell.margin_left = Inches(.13)
            cell.margin_right = Inches(.08)
            cell.margin_top = cell.margin_bottom = Inches(.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = color(PALE if i == 0 or i in emphasize else WHITE)
            cell.text = str(value)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                p.space_after = Pt(0)
                p.line_spacing = 1.08
                for r in p.runs:
                    fmt(r, size, i == 0 or i in emphasize, ACCENT if i == 0 else INK)
            tcpr = cell._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                el = OxmlElement(tag)
                el.set("w", "5000" if tag == "a:lnB" else "0")
                if tag == "a:lnB":
                    solid = OxmlElement("a:solidFill")
                    rgb = OxmlElement("a:srgbClr")
                    rgb.set("val", LINE)
                    solid.append(rgb)
                    el.append(solid)
                else:
                    el.append(OxmlElement("a:noFill"))
                tcpr.append(el)
    return tab


with (ROOT / "metrics_source.csv").open(newline="", encoding="utf-8-sig") as f:
    metrics = list(csv.DictReader(f))
tasks = ["clean", "jpeg", "gaussian", "resize", "crop"]
labels = {"clean": "clean", "jpeg": "像素量化*", "gaussian": "Gaussian", "resize": "resize", "crop": "crop"}
final = {r["task"]: r for r in metrics if int(r["stage"]) == 4}
lookup = {(int(r["stage"]), r["task"]): r for r in metrics}
assert len(metrics) == 15 and set(final) == set(tasks)

# 1: No separate cover or agenda: begin with the research decision.
s = base("持续学习如何服务于鲁棒图像隐藏", "ContinualSteg · 阶段汇报 / 导师讨论 · 2026.11.10", "依据：已上传训练 CSV、服务器运行截图及当前仓库代码（f60e05e）。")
text(s, .7, 2.04, 11.85, 1.0, "研究问题：新退化依次出现时，如何兼顾\n新任务恢复、旧任务保留与载密图像质量？", 24, True, ACCENT)
for y, label, content in [
    (3.39, "已完成", "五阶段训练与跨阶段评估；旧任务性能基本保持。"),
    (4.24, "当前瓶颈", "无攻击恢复质量仍有限；裁剪后的恢复明显下降。"),
    (5.09, "讨论重点", "厘清低遗忘来源，确定可验证的机制与实验主线。"),
]:
    text(s, .7, y, 1.6, .43, label, 20, True)
    text(s, 2.45, y, 10.08, .53, content, 20)
text(s, .7, 6.24, 11.9, .45, "当前定位：自建隐藏 / 恢复 CNN 的持续学习原型；尚非原始 HiNet / UDH 的完整整合。", 16, fill=MUTED)
note(s, "汇报目的：先说已完成与可验证的研究问题，不将机制组合直接称为创新。当前仓库是简化卷积隐藏/恢复模型，HiNet 与 UDH 的原始结构尚未接入。已有材料是 metrics [2].csv（15行）和运行截图，未收到实际模型权重、验证图片或完整消融CSV，故不展示虚构的图像恢复效果。完整训练50轮/任务和128像素等来自此前命令/日志，CSV本身不记录超参数。")

# 2: Editable model diagram and freeze/train protocol.
s = base("持续学习体现为：顺序任务 + 旧路径冻结", "任务变化来自退化类型；当前实验在编码端、恢复端均使用已知任务编号 t。", "代码：continual_steg/models.py · train.py；思路借鉴 RwF 的路径隔离与 AIM 的轻量适配。")
boxes = [(.70, 1.50, "载体 + 秘密"), (2.75, 2.30, "隐藏网络 H\n骨干 + 任务路径"), (5.60, 1.55, "退化 Aₜ"), (7.70, 2.30, "恢复网络 R\n骨干 + 任务路径"), (10.55, 2.00, "恢复秘密")]
for x, w, label in boxes:
    rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(2.12), Inches(w), Inches(.99))
    rect.fill.solid()
    rect.fill.fore_color.rgb = color(WHITE)
    rect.line.color.rgb = color(LINE)
    rect.line.width = Pt(1)
    text(s, x+.08, 2.31 if "\n" not in label else 2.23, w-.16, .72, label, 16, align=PP_ALIGN.CENTER)
for x in [2.25, 5.10, 7.20, 10.05]:
    text(s, x, 2.40, .45, .45, "→", 23, fill=ACCENT, align=PP_ALIGN.CENTER)
text(s, .73, 3.52, 11.9, .5, "clean  →  像素量化  →  Gaussian  →  resize  →  crop", 22, True, ACCENT)
table(s, .7, 4.30, [2.8, 3.0, 3.0, 3.0], [
    ["训练阶段", "共享骨干", "旧任务路径", "当前任务路径"],
    ["初始 clean", "训练", "—", "训练"],
    ["后续新任务", "冻结", "冻结", "训练"],
], row_h=.55, size=17)
text(s, .73, 6.25, 11.9, .44, "每任务保留独立 Adapter；以参数隔离保留旧能力，参数与存储开销随任务数增加。", 17)
note(s, "持续学习机制：数据按任务顺序到来，新任务优化不重新访问完整旧训练流，每完成一阶段回测所有已学任务。当前隔离模式冻结共享骨干与旧Adapter，因此旧路径输出理论上保持稳定；这本身属于参数隔离式持续学习。Adapter库在当前实现中预先创建，不是运行时动态扩展。训练/评估调用显式 task_id，编码器也依赖该id；route/recover_unknown仅提供接口，尚未纳入本次CSV的评估，不能宣称无任务标签推理。RwF参考 https://github.com/AlifAshrafee/Restoring-Without-Forgetting；AIM参考 https://adapt-image-models.github.io/ （适配思想借鉴，并未复现其原任务/模型）。")

# 3: All data below come directly from the uploaded CSV.
s = base("最终恢复质量：裁剪是当前最明显的瓶颈", "DIV2K：训练 800 / 验证 100 张；128×128；50轮 / 任务（运行记录），共五阶段。", "来源：metrics [2].csv，stage=4；PSNR：dB，APD：0–255；*jpeg 为像素量化，非真实 JPEG。")
headers = ["任务", "载密PSNR↑", "载密SSIM↑", "恢复PSNR↑", "恢复SSIM↑", "载密APD↓"]
fields = ["cover_psnr", "cover_ssim", "secret_psnr", "secret_ssim", "cover_apd"]
digits = [2, 3, 2, 3, 2]
rows = [headers]
for t in tasks:
    rows.append([labels[t]] + [f"{float(final[t][k]):.{d}f}" for k, d in zip(fields, digits)])
rows.append(["任务宏平均"] + [f"{sum(float(final[t][k]) for t in tasks)/5:.{d}f}" for k, d in zip(fields, digits)])
table(s, .7, 2.02, [2.15, 1.93, 1.93, 1.93, 1.93, 1.93], rows, row_h=.48, size=16, emphasize=(5,6))
text(s, .73, 5.62, 11.8, .5, "clean 恢复 20.35 dB / 0.570；crop 恢复降至 14.90 dB / 0.232。", 21, True)
text(s, .73, 6.19, 11.8, .48, "当前 PSNR 按 batch 汇总；五任务均值为宏平均。尚无原始 HiNet / UDH 同协议对照。", 15, fill=MUTED)
note(s, "指标口径：cover_* 比较原载体与攻击前载密图；secret_* 比较秘密与攻击后恢复秘密。APD在0–255尺度；当前CSV仅有载体APD，无秘密APD。原metrics.py先计算batch整体MSE再转PSNR，evaluate对batch结果等权平均；不等同于逐图PSNR算术平均，最后不足batch的权重也需统一。五任务均值仅为当前五个任务统计值的等权宏平均。crop代码保留中心75%的高和宽，再缩放回来，相当于保留56.25%面积，且造成坐标缩放；量化步长为50/255，不能标记为JPEG Q=50。不能从无对照结果宣称优于UDH或HiNet，也不能只凭PSNR/SSIM宣称抗隐写检测安全。")

# 4: Continual evaluation matrix in editable cells.
s = base("旧任务几乎不下降，但低遗忘的来源需要辨清", "恢复 PSNR 矩阵（dB）：每完成一个新阶段，重新评估所有已学任务。", "来源：metrics [2].csv；Gaussian 存在随机噪声评估波动；最后任务没有后续遗忘观测。")
matrix = [["完成阶段", "clean", "量化", "Gaussian", "resize", "crop"]]
for stage, t in enumerate(tasks):
    matrix.append([f"{stage}  {labels[t].replace('*','')}"] + [f"{float(lookup[(stage,k)]['secret_psnr']):.2f}" if (stage,k) in lookup else "—" for k in tasks])
table(s, .70, 2.14, [1.95, 1.20, 1.20, 1.40, 1.20, 1.20], matrix, row_h=.57, size=15, emphasize=(5,))
text(s, 9.35, 2.18, 3.25, .48, "观察到的遗忘", 21, True)
text(s, 9.35, 2.89, 3.25, 2.35, "clean / 量化 / resize\n接近 0 dB\nGaussian：约 0.015 dB\ncrop：暂不可评价", 17)
band(s, 5.99, "低遗忘与冻结旧路径一致，不能单独归因于 Replay / Distillation。", 20)
note(s, "CSV中每个任务最佳已观测PSNR减最终PSNR：clean=0.0000014343 dB，jpeg=0，gaussian=0.0149157416 dB，resize=0.0000030050 dB；crop仅一条观测，记N/A而非证明0遗忘。此处沿用当前脚本的非负best-to-final口径；正式对照应另外报告学习后到最终的有符号变化（BWT）以显示正迁移。旧路径的共享骨干与任务参数都冻结，所以结果稳定是架构预期；Gaussian差异不能直接解读为遗忘，因为每次调用randn_like未固定逐图噪声。")

# 5: Screenshot pilot evidence is explicitly separate from the formal CSV.
s = base("短消融暴露出：蒸馏尚未形成有效约束", "短试验：5轮 / 任务 × 20训练 batch；仅5个验证 batch。用于排错，不作正式效果比较。", "数值区间来自用户终端截图（非完整消融CSV）；机制结论来自 train.py / models.py 静态核对。")
table(s, .7, 2.01, [6.5, 5.3], [
    ["短试验组别", "各任务最终恢复 PSNR 区间"],
    ["full / no_distill / adapter_only", "均约 11.74–11.81 dB"],
    ["sequential_finetune", "约 14.29–15.61 dB"],
], row_h=.58, size=18)
text(s, .73, 4.04, 11.8, .5, "蒸馏路径：旧样本 → 已冻结的旧任务路径 → 输出匹配损失", 21, True, ACCENT)
text(s, .73, 4.77, 11.8, .62, "当前任务的可训练参数未参与这条路径，蒸馏损失对它们没有梯度。", 20)
text(s, .73, 5.48, 11.8, .59, "no_distill 也没有旧样本监督损失，不能作为“仅 Replay”的有效消融。", 19)
text(s, .73, 6.29, 11.8, .40, "骨干只学5轮便冻结也限制了后续恢复。应先修正实验定义，再安排长时间训练。", 16, fill=MUTED)
note(s, "关键更正：此前将低遗忘解释为Replay/Distillation有效是不充分的。train_one_task用old_task_id同时访问教师和当前模型的旧路径，而continual策略冻结共享骨干与旧路径；当前任务Adapter/Prompt没有参与蒸馏分支，所以该项不能约束它们。旧样本只用于该蒸馏项，lambda_distill=0时并没有监督回放损失，因此no_distill与adapter_only不是有意义的回放消融。原型三组的微小数值差异可能来自随机数消耗引起的随机噪声或采样顺序变化，不能证明贡献。截图的sequential_finetune仍保留各任务分支但更新共享骨干，不是严格的单共享分支基线。5轮结果不与之前50轮结果直接排名。")

# 6: Material correctness checks that affect the research conclusion.
s = base("形成论文结论前，需收紧三个实验边界", "这些是影响实验解释的具体问题；本次汇报不把待修复能力写成已验证成果。", "依据：models.py 的 Prompt 初始化；tasks.py 的退化实现；metrics.py 与 train.py 的评估逻辑。")
table(s, .7, 2.03, [2.2, 5.0, 4.6], [
    ["检查项", "当前实现", "下一步"],
    ["Prompt 作用", "向量 p 与投影 W 同时零初始化\np、W 的梯度为0，bias可更新", "检查梯度与参数变化\n再修正初始化"],
    ["真实退化", "jpeg 是像素量化\ncrop 是固定中心裁剪再放大", "加入真实 JPEG 与多强度退化\n固定同图、同攻击做对照"],
    ["指标可比性", "按 batch 平均指标\nGaussian 每次采样不同噪声", "统一逐图统计与评估随机性\n完整验证集 + 多随机种子"],
], row_h=.87, size=17)
text(s, .73, 6.09, 11.8, .62, "PSNR / SSIM 衡量失真与恢复；抗隐写检测性尚未评估，不能据此宣称隐写安全。", 18, True)
note(s, "Prompt代码推导：z=Wp+b，同时p=0、W=0，则dL/dp=Wᵀδ=0，dL/dW=δpᵀ=0；b可以更新，但不打破p与W的零梯度链。依据当前初始化与普通AdamW，参数可能一直保持零值；应记录梯度/参数变化进行实证核查，再考虑随机初始化p或W并保持残差输出稳定。此结论是静态审阅，不是已运行的梯度测试。当前环境未运行GPU训练，也未修改训练代码。正式鲁棒性需要真实JPEG、攻击强度和未知/混合退化；论文若主张隐写安全还需隐写分析器检测实验。")

# 7: Proposed experiments, explicitly not completed results.
s = base("下一步只做能回答问题的最小对照", "建议协议（待确认）：固定同一嵌入器，从同一收敛起点持续训练提取器。", "以下为计划；当前尚无按此修订协议完成的正式多种子结果。")
table(s, .7, 2.04, [6.3, 5.5], [
    ["实验组", "主要回答的问题"],
    ["共享提取器：直接顺序微调", "新任务适应与旧任务遗忘的基线"],
    ["冻结骨干 + 独立 Adapter", "隔离能保留多少能力？存储成本多少？"],
    ["共享可训练模块 + 监督回放", "回放是否确实改善旧任务保留？"],
    ["上一组 + 冻结教师蒸馏", "蒸馏是否带来额外收益？"],
], row_h=.59, size=17)
text(s, .73, 5.38, 11.8, .57, "统一：图像配对、退化强度、训练预算和评估噪声；完整验证集，至少3个随机种子。", 17)
text(s, .73, 6.11, 11.8, .61, "交付：恢复质量 × 遗忘 × 参数 / 存储开销；同样本六联图，并保留裁剪失败案例。", 19, True)
note(s, "这页是建议的新协议，不等同于当前run_ablation.sh已实现的四组。先建立足够好的干净隐藏/恢复基线，再从相同初始化和配对样本比较。固定嵌入器可以避免编码端预知未来攻击类型的假设；统一载密图也便于退化之间公平比较。监督回放需要对旧样本经过旧攻击后的恢复输出计算真实目标损失；蒸馏需要经过当前可训练的共享模块，而不是独立冻结的旧模块。正式报告每阶段任务矩阵、最终恢复与载密PSNR/SSIM、逐图APD、遗忘/BWT、均值±标准差、训练参数数和总存储；上限可补联合训练参考。若后续要实验任务顺序，须先修复训练器中任务id与列表位置混用，不能直接改顺序。视觉结果使用真实checkpoint与验证图片，不从CSV推造图像。")

# 8: End with decisions rather than a decorative thank-you slide.
s = base("与导师确认的三个取舍", "讨论目标：明确任务假设、方法主线和先验证哪一个原始基线。", "参考：HiNet (ICCV 2021)、UDH (NeurIPS 2020)、RwF、AIM；具体链接与证据见备注及同目录说明。")
decisions = [
    (2.00, "1  场景：嵌入后退化是否未知？", "建议固定嵌入器，持续适应提取器；\n已知任务编号的结果作为条件明确的参考。"),
    (3.54, "2  主线：隔离保留，还是共享适应？", "建议以有效回放 / 蒸馏约束共享可训练模块为主线；\n独立 Adapter 作为低遗忘与存储成本的对照。"),
    (5.08, "3  基线：先 UDH，还是直接上 HiNet？", "建议先在原始 UDH 上建立统一实验；\n随后单独处理 HiNet 可逆结构的适配。"),
]
for y, title, body in decisions:
    text(s, .73, y, 11.8, .48, title, 21, True, ACCENT)
    text(s, 1.13, y+.56, 11.3, .79, body, 18)
note(s, "请导师重点决定三件事，而不是先追加更多模块。任务场景决定是否允许编码端/恢复端使用已知任务编号；若实际传输退化未知，固定编码端更合适，恢复端未知任务路由需另做实验。方法主线应在新任务可塑性与旧任务稳定性之间有可测量权衡；冻结方案可作为可靠对照，但不能仅凭它声称蒸馏创新。基线建议先原始UDH，保留统一数据与攻击协议，再迁移HiNet并验证其可逆约束。参考来源：HiNet https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html ; UDH (NeurIPS 2020) https://proceedings.neurips.cc/ ; RwF https://github.com/AlifAshrafee/Restoring-Without-Forgetting ; AIM https://adapt-image-models.github.io/ 。")

out = ROOT / "ContinualSteg_advisor_brief.pptx"
for slide in prs.slides:
    # Disable template effects: the deck uses plain lines and flat boxes.
    for shape in slide.shapes:
        for effect in shape._element.xpath(".//a:effectRef"):
            effect.set("idx", "0")
        sppr = shape._element.find(qn("p:spPr"))
        if sppr is not None:
            sppr.append(OxmlElement("a:effectLst"))
prs.save(out)
print(f"Saved {len(prs.slides)} slides: {out}")
