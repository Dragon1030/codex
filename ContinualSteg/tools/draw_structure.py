from pathlib import Path

W, H = 1800, 1080
OUT = Path(__file__).resolve().parents[1] / "docs"
SVG = OUT / "continual_steg_structure_clean.svg"

INK = "#17324D"
MUTED = "#5E6B78"
BLUE = "#DDEAF7"
BLUE_STROKE = "#4479A7"
ORANGE = "#FFF0E4"
ORANGE_STROKE = "#E2772D"
GREEN = "#E6F3E6"
GREEN_STROKE = "#4A9655"
PINK = "#FBE3EC"
PINK_STROKE = "#C74B78"
GRAY = "#F4F6F8"
LINE = "#52606D"
WHITE = "#FFFFFF"

parts = []
def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
def text(x, y, lines, size=24, fill=INK, weight=400, anchor="middle", line_gap=31):
    if isinstance(lines, str): lines = [lines]
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Noto Sans CJK SC,Microsoft YaHei,Arial,sans-serif" font-size="{size}px" font-weight="{weight}" fill="{fill}">' + ''.join(f'<tspan x="{x}" dy="{0 if i==0 else line_gap}px">{esc(line)}</tspan>' for i,line in enumerate(lines)) + '</text>')
def rect(x,y,w,h,fill,stroke,rx=14,sw=2):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def line(x1,y1,x2,y2,stroke=LINE,width=3,dash=None,marker=True):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = ' marker-end="url(#arrow)"' if marker else ''
    parts.append(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{stroke}" stroke-width="{width}"{d}{m}/>')
def poly(points,stroke=LINE,width=3,dash=None,marker=True):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = ' marker-end="url(#arrow)"' if marker else ''
    parts.append(f'<path d="M ' + ' L '.join(f'{x} {y}' for x,y in points) + f'" fill="none" stroke="{stroke}" stroke-width="{width}"{d}{m}/>')
def badge(x,y,num):
    parts.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{INK}"/>')
    text(x,y+8,str(num),18,WHITE,700)

def box(x,y,w,h,title,body,fill=GRAY,stroke=BLUE_STROKE,title_size=24,body_size=19):
    rect(x,y,w,h,fill,stroke)
    text(x+w/2,y+38,title,title_size,INK,700)
    body_y = y + 102 if h >= 120 else y + 72
    text(x+w/2,body_y,body,body_size,MUTED,400, line_gap=25)

parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="{LINE}"/></marker></defs>
<rect width="100%" height="100%" fill="white"/>
''')
text(900,55,"ContinualSteg：持续学习鲁棒图像信息隐藏流程",34,INK,700)
text(900,92,"上方是一次前向数据流，下方是任务选择、回放、损失和参数更新",19,MUTED,400)
# Main forward row
box(35,180,180,82,"Cover x","载体图像",WHITE,"#8B98A5",22,18)
box(35,315,180,82,"Secret m","秘密图像",WHITE,"#8B98A5",22,18)
box(285,220,280,135,"HidingNet","共享主干",BLUE,BLUE_STROKE,24,19)
box(635,245,170,85,"Container s","载密图像",WHITE,"#8B98A5",22,18)
box(875,220,220,135,"攻击 / 退化 A_t","量化 / 噪声 / 缩放 / 裁剪",PINK,PINK_STROKE,24,17)
box(1165,220,280,135,"RevealNet","共享主干",BLUE,BLUE_STROKE,24,19)
box(1515,245,220,85,"Recovered m̂","恢复秘密图像",WHITE,"#8B98A5",22,18)
# Main arrows
poly([(215,221),(250,221),(250,270),(285,270)])
poly([(215,356),(250,356),(250,305),(285,305)])
line(565,287,635,287)
line(805,287,875,287)
line(1095,287,1165,287)
line(1445,287,1515,287)
# number badges
for x,n in [(245,1),(595,2),(840,3),(1125,4),(1480,5)]: badge(x,170,n)
# Adapter injection labels / arrows
rect(330,390,250,86,ORANGE,ORANGE_STROKE)
text(455,423,"当前任务路径",20,INK,700)
text(455,456,"Adapter_t + Prompt_t",19,MUTED)
rect(1170,390,250,86,ORANGE,ORANGE_STROKE)
text(1295,423,"当前任务路径",20,INK,700)
text(1295,456,"Adapter_t + Prompt_t",19,MUTED)
poly([(455,390),(455,355)],ORANGE_STROKE,3,"8 7")
poly([(1295,390),(1295,355)],ORANGE_STROKE,3,"8 7")
text(470,380,"注入",17,ORANGE_STROKE,700)
text(1310,380,"注入",17,ORANGE_STROKE,700)
# controller panel
rect(230,535,1350,455,"#F5FBF5",GREEN_STROKE,18,3)
text(260,575,"持续学习控制器",26,GREEN_STROKE,700,"start")
text(260,608,"每完成一个退化任务：冻结旧路径 → 训练当前路径 → 评估所有已见任务",18,MUTED,400,"start")
# top controller row
box(280,655,210,82,"Task t / Task ID","当前退化任务",GREEN,GREEN_STROKE,21,18)
box(560,655,230,82,"Router","选择任务路径",GREEN,GREEN_STROKE,21,18)
box(860,655,260,82,"Prototype Bank","每个任务一个原型",GREEN,GREEN_STROKE,21,18)
box(1200,655,300,82,"Adapter / Prompt Bank","旧任务冻结；只训练当前任务",GREEN,GREEN_STROKE,21,18)
line(490,696,560,696,GREEN_STROKE,3)
line(790,696,860,696,GREEN_STROKE,3)
line(1120,696,1200,696,GREEN_STROKE,3)
# route upward to top components (minimal)
poly([(675,655),(675,520),(455,520),(455,476)],GREEN_STROKE,3,"10 8")
poly([(675,655),(675,520),(1295,520),(1295,476)],GREEN_STROKE,3,"10 8")
text(685,535,"选择当前 Adapter / Prompt",17,GREEN_STROKE,700,"start")
# bottom controller row
box(280,825,250,90,"Replay Buffer","旧样本 / task_id",GREEN,GREEN_STROKE,21,18)
box(610,825,270,90,"Frozen Teacher","上一阶段模型\n只用于旧任务对照",GREEN,GREEN_STROKE,21,18)
box(960,825,250,90,"Loss","Lcover + Lsecret\n+ λ Ldistill",ORANGE,ORANGE_STROKE,22,19)
box(1290,825,250,90,"AdamW + Checkpoint","更新、评估、保存",ORANGE,ORANGE_STROKE,21,18)
line(530,870,610,870,GREEN_STROKE,3)
line(880,870,960,870,GREEN_STROKE,3)
line(1210,870,1290,870,GREEN_STROKE,3)
# connections from main outputs to loss; use vertical paths to avoid crossings
poly([(720,330),(720,780),(1085,780),(1085,825)],ORANGE_STROKE,3,"10 8")
poly([(1305,355),(1305,780),(1085,780),(1085,825)],ORANGE_STROKE,3,"10 8")
text(735,770,"载体 / 秘密重建损失",17,ORANGE_STROKE,700,"start")
# replay teacher to loss
poly([(530,870),(570,870),(570,790),(960,790),(960,825)],GREEN_STROKE,3,"10 8")
poly([(880,870),(920,870),(920,790),(960,790)],GREEN_STROKE,3,"10 8")
text(575,785,"旧任务样本 + 教师输出",17,GREEN_STROKE,700,"start")
# checkpoint back to prototype / stage
poly([(1415,825),(1415,760),(990,760),(990,737)],GREEN_STROKE,3,"10 8")
text(1230,755,"任务结束后更新原型并评估所有已见任务",17,GREEN_STROKE,700,"start")
# legend
text(35,1035,"实线：前向数据流",18,MUTED,400,"start")
line(230,1028,285,1028,LINE,3)
text(320,1035,"橙色虚线：当前任务路径与损失",18,MUTED,400,"start")
line(625,1028,680,1028,ORANGE_STROKE,3,"10 8")
text(715,1035,"绿色虚线：路由、回放和持续学习控制",18,MUTED,400,"start")
line(1125,1028,1180,1028,GREEN_STROKE,3,"10 8")
text(1215,1035,"蓝色：共享网络；绿色：控制器",18,MUTED,400,"start")
parts.append('</svg>')
SVG.write_text(''.join(parts), encoding='utf-8')
print(SVG)
