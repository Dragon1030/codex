"""Layout helpers for the results discussion deck.

Imported by build_report.py. Requires python-pptx.
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
prs.core_properties.title = "ContinualSteg：正式实验结果与解释"
prs.core_properties.subject = "2026-11-11 四组50轮实验的审计、重算与可视化"
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


