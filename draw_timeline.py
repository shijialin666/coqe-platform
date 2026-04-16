import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(20, 14), dpi=200)
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, bg, edge, fontsize=8.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=bg, edgecolor=edge, linewidth=1.2, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color='#1a1a1a', zorder=3,
            multialignment='center')

def draw_line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color='#aaaaaa', linewidth=1.0, zorder=1)

def draw_arrow_down(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.0), zorder=1)

# ==========================================
# 层1：用户交互层
# ==========================================
draw_box(ax, 4.0, 12.5, 12.0, 0.9, '用户交互层\n浏览器 / Streamlit Web 界面',
         '#E3F2FD', '#1565C0', fontsize=9)
layer1_cx = 10.0
layer1_bottom = 12.5

# ==========================================
# 层2：五大功能模块
# ==========================================
tabs = [
    ('数据仓库\n管理模块', '#E8F5E9', '#2E7D32'),
    ('单条样本\n调试模块', '#FFF3E0', '#E65100'),
    ('自动化\n批处理模块', '#F3E5F5', '#6A1B9A'),
    ('评测历史\n榜单模块', '#FFF8E1', '#F57F17'),
    ('Bad Case\n诊断模块', '#FFEBEE', '#C62828'),
]
tab_w, tab_h, tab_gap = 2.8, 0.9, 0.55
tab_y = 10.8
tab_centers = []
for i, (text, bg, edge) in enumerate(tabs):
    x = 1.0 + i * (tab_w + tab_gap)
    draw_box(ax, x, tab_y, tab_w, tab_h, text, bg, edge)
    cx = x + tab_w / 2
    tab_centers.append(cx)

# 层1 -> 层2 竖线
mid_x = (tab_centers[0] + tab_centers[-1]) / 2
draw_line(ax, layer1_cx, layer1_bottom, layer1_cx, tab_y + tab_h + 0.15)
draw_line(ax, tab_centers[0], tab_y + tab_h + 0.15, tab_centers[-1], tab_y + tab_h + 0.15)
for cx in tab_centers:
    draw_arrow_down(ax, cx, tab_y + tab_h + 0.15, tab_y + tab_h)

# ==========================================
# 层3：后端处理层
# ==========================================
backends = [
    ('TXT知识库\n解析引擎', '#E3F2FD', '#1565C0'),
    ('N-gram相似度\n检索引擎', '#E8F5E9', '#2E7D32'),
    ('LLM择优\n处理模块', '#FFF3E0', '#E65100'),
    ('LLM动态生成\n模块', '#F3E5F5', '#6A1B9A'),
    ('F1精确匹配\n评测模块', '#FFF8E1', '#F57F17'),
]
bk_w, bk_h, bk_gap = 2.8, 0.9, 0.55
bk_y = 9.0
bk_centers = []
for i, (text, bg, edge) in enumerate(backends):
    x = 1.0 + i * (bk_w + bk_gap)
    draw_box(ax, x, bk_y, bk_w, bk_h, text, bg, edge)
    cx = x + bk_w / 2
    bk_centers.append(cx)

# 层2 -> 层3
draw_line(ax, tab_centers[0], bk_y + bk_h + 0.15, tab_centers[-1], bk_y + bk_h + 0.15)
for cx in tab_centers:
    draw_line(ax, cx, tab_y, cx, bk_y + bk_h + 0.15)
for cx in bk_centers:
    draw_arrow_down(ax, cx, bk_y + bk_h + 0.15, bk_y + bk_h)

# ==========================================
# 层4：核心算法文件层
# ==========================================
algos = [
    ('llm_select_topk.py\n混合模式后端', '#E8F5E9', '#2E7D32'),
    ('onlyselect.py\n纯选择模式后端', '#FFF3E0', '#E65100'),
    ('evaluate_f1.py\nF1 评测模块', '#FFF8E1', '#F57F17'),
]
al_w, al_h, al_gap = 4.5, 0.9, 0.75
al_y = 7.2
al_centers = []
for i, (text, bg, edge) in enumerate(algos):
    x = 1.5 + i * (al_w + al_gap)
    draw_box(ax, x, al_y, al_w, al_h, text, bg, edge)
    cx = x + al_w / 2
    al_centers.append(cx)

# 层3 -> 层4
draw_line(ax, bk_centers[0], al_y + al_h + 0.15, bk_centers[-1], al_y + al_h + 0.15)
for cx in bk_centers:
    draw_line(ax, cx, bk_y, cx, al_y + al_h + 0.15)
for cx in al_centers:
    draw_arrow_down(ax, cx, al_y + al_h + 0.15, al_y + al_h)

# ==========================================
# 层5：数据存储层
# ==========================================
storage = [
    ('候选集 JSON\n待处理数据', '#E3F2FD', '#1565C0'),
    ('领域知识库 TXT\ntrain.txt', '#E8F5E9', '#2E7D32'),
    ('预测结果 JSON\n处理产出数据', '#FFF3E0', '#E65100'),
    ('评测历史 CSV\n历史记录', '#FFF8E1', '#F57F17'),
    ('Checkpoint JSON\n断点续传文件', '#F3E5F5', '#6A1B9A'),
]
st_w, st_h, st_gap = 2.8, 0.9, 0.55
st_y = 5.4
st_centers = []
for i, (text, bg, edge) in enumerate(storage):
    x = 1.0 + i * (st_w + st_gap)
    draw_box(ax, x, st_y, st_w, st_h, text, bg, edge)
    cx = x + st_w / 2
    st_centers.append(cx)

# 层4 -> 层5
draw_line(ax, al_centers[0], st_y + st_h + 0.15, al_centers[-1], st_y + st_h + 0.15)
for cx in al_centers:
    draw_line(ax, cx, al_y, cx, st_y + st_h + 0.15)
for cx in st_centers:
    draw_arrow_down(ax, cx, st_y + st_h + 0.15, st_y + st_h)

# ==========================================
# 层级标签
# ==========================================
labels = [
    (13.1, 12.95, '① 用户交互层'),
    (13.1, 11.25, '② 功能模块层'),
    (13.1, 9.45,  '③ 后端算法处理层'),
    (13.1, 7.65,  '④ 核心算法文件层'),
    (13.1, 5.85,  '⑤ 数据存储层'),
]
for x, y, label in labels:
    ax.text(x, y, label, ha='left', va='center', fontsize=8,
            color='#555555', style='italic')

# 标题
ax.text(10, 13.6, '系统功能框架图', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#1a1a1a')

plt.tight_layout()
plt.savefig('system_framework.png', bbox_inches='tight', pad_inches=0.3)
plt.show()
print("已保存: system_framework.png")