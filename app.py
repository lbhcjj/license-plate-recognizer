# -*- coding: utf-8 -*-
"""
AI车牌识别系统 - 最终融合版
动效：扫描波纹 + AI脉冲流光进度条
功能：图片识别 · 手动修正 · ZIP批量 · 统计筛选 · CSV/Excel导出
"""

import streamlit as st
import cv2
import numpy as np
from openai import OpenAI
import hyperlpr3 as lpr3
import re
import pandas as pd
from PIL import Image
from io import BytesIO
import time
from functools import wraps
import zipfile
import hashlib
import uuid
import plotly.express as px

# ====================================================================
# 工具函数
# ====================================================================
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def resize_image_if_needed(img, max_size=1920):
    h, w = img.shape[:2]
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
    return img

def to_excel(df):
    output = BytesIO()
    df_export = df.drop(columns=["id", "file_md5"], errors="ignore")
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='识别记录')
    return output.getvalue()

def draw_plate_box(img_rgb, plate_item):
    box = plate_item.get("box")
    if not box or len(box) != 4:
        return img_rgb
    try:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (120, 255, 0), 3)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (80, 200, 0), 1)
    except Exception:
        pass
    return img_rgb

def render_plate_card(fmt_plate, color_cls, conf, ptype, addr):
    badge_cls = "badge-high" if conf >= 0.85 else "badge-medium" if conf >= 0.6 else "badge-low"
    badge_txt = "高" if conf >= 0.85 else "中" if conf >= 0.6 else "低"
    st.markdown(f"""
    <div class="plate-card">
        <div class="plate-number {color_cls}">{fmt_plate}</div>
        <div class="result-metrics">
            <div><div>📊 置信度</div><span class="badge {badge_cls}">{conf:.0%} · {badge_txt}</span></div>
            <div><div>🏷️ 类型</div><div style="font-weight:600">{ptype}</div></div>
            <div><div>📍 属地</div><div style="font-weight:600">{addr}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_lpr():
    return lpr3.LicensePlateCatcher()

# ====================================================================
# 常量配置
# ====================================================================
PROVINCE_MAP = {
    "京": "北京市", "沪": "上海市", "津": "天津市", "渝": "重庆市",
    "冀": "河北省", "豫": "河南省", "云": "云南省", "辽": "辽宁省",
    "黑": "黑龙江省", "湘": "湖南省", "皖": "安徽省", "鲁": "山东省",
    "新": "新疆维吾尔自治区", "苏": "江苏省", "浙": "浙江省",
    "赣": "江西省", "鄂": "湖北省", "桂": "广西壮族自治区",
    "甘": "甘肃省", "晋": "山西省", "蒙": "内蒙古自治区",
    "陕": "陕西省", "吉": "吉林省", "闽": "福建省", "贵": "贵州省",
    "粤": "广东省", "青": "青海省", "藏": "西藏自治区",
    "川": "四川省", "宁": "宁夏回族自治区", "琼": "海南省",
    "使": "使馆号牌", "领": "领事馆号牌"
}

EV_PURE = {'D', 'A', 'B', 'C', 'E'}
EV_HYBRID = {'F', 'G', 'H', 'J', 'K'}
EV_OTHER = {'L', 'M', 'N', 'P', 'R'}
EV_ALL = EV_PURE | EV_HYBRID | EV_OTHER

PROVINCE = r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]'
CITY = r'[A-Z]'
RULE_BLUE = re.compile(f'^{PROVINCE}{CITY}[A-Z0-9]{{5}}$')
RULE_GREEN = re.compile(f'^{PROVINCE}{CITY}[A-Z0-9]{{6}}$')
RULE_YELLOW = re.compile(f'^{PROVINCE}{CITY}[A-Z0-9]{{4,5}}$')

# ====================================================================
# 核心解析函数
# ====================================================================
def retry_api(max_retry=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            err = None
            for i in range(max_retry + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err = e
                    if i < max_retry:
                        time.sleep(1)
            return f"调用失败：{type(err).__name__}"
        return wrapper
    return decorator

def smart_plate_parser(raw_plate, plate_color, conf_threshold=0.6, confidence=0.0):
    raw_plate = raw_plate.replace('·', '').strip()
    if confidence < conf_threshold:
        return None, raw_plate, confidence, "识别无效(置信度过低)", ""

    plate_color = plate_color.lower() if plate_color else 'unknown'
    plate_type = "未知类型"
    formatted_plate = raw_plate
    addr = PROVINCE_MAP.get(raw_plate[0], "未知属地") if len(raw_plate) >= 1 else ""

    if plate_color == 'green':
        if RULE_GREEN.match(raw_plate):
            energy_char = raw_plate[2] if len(raw_plate) >= 3 else ''
            if energy_char not in EV_ALL:
                plate_type = "新能源(格式异常)"
            elif energy_char in EV_PURE:
                plate_type = "新能源（纯电动·小型车）"
            elif energy_char in EV_HYBRID:
                plate_type = "新能源（插电混动·小型车）"
            elif energy_char in EV_OTHER:
                plate_type = "新能源（特殊号段·小型车）"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}"
    elif plate_color == 'blue':
        if RULE_BLUE.match(raw_plate):
            plate_type = "燃油汽车"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}"
    elif plate_color == 'yellow':
        if RULE_YELLOW.match(raw_plate):
            energy_char = raw_plate[-1] if len(raw_plate) >= 1 else ''
            if energy_char in EV_PURE:
                plate_type = "新能源（纯电动·大型车）"
            elif energy_char in EV_HYBRID:
                plate_type = "新能源（插电混动·大型车）"
            else:
                plate_type = "大型黄牌车辆"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}"
    else:
        if RULE_GREEN.match(raw_plate):
            plate_type = "新能源汽车（推断）"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}"
        elif RULE_BLUE.match(raw_plate):
            plate_type = "燃油汽车（推断）"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}"
        elif RULE_YELLOW.match(raw_plate):
            plate_type = "大型黄牌车辆（推断）"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}"

    return formatted_plate, raw_plate, confidence, plate_type, addr

def parse_lpr_results(results_tuple):
    results_dict = []
    seen_plates = set()
    for item in results_tuple:
        try:
            if len(item) < 4:
                continue
            plate_str = str(item[0])
            if plate_str in seen_plates:
                continue
            seen_plates.add(plate_str)
            conf = float(item[1])
            box = item[3] if len(item) > 3 else None
            color = item[4].lower() if len(item) >= 5 and isinstance(item[4], str) else 'unknown'
            if box and len(box) == 4:
                results_dict.append({"plate": plate_str, "confidence": conf, "color": color, "box": box})
        except Exception:
            continue
    return results_dict

@retry_api(max_retry=2)
def deepseek_analyze(context_info, api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=15.0)
    system_msg = "严格遵循GA36-2018：小型新能源第3位=D/A/B/C/E/F/G/H/J/K，第4位可以是数字。示例：粤AD38467=合法"
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": f"{context_info}\n请给出：1.归属地 2.合理性评价"}]
    )
    if not resp.choices or not resp.choices[0].message or not resp.choices[0].message.content:
        return "AI 分析暂不可用：API 返回格式异常，请稍后重试"
    return resp.choices[0].message.content

# ====================================================================
# 页面初始化
# ====================================================================
st.set_page_config(page_title="AI车牌识别系统", layout="wide", page_icon="🚗")

for key in ["history", "results_cache", "zip_cache", "api_key"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "api_key" else ""
for key in ["uploader_key", "files_processed", "last_file_count", "zip_processed", "zip_name"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key not in ("zip_processed", "zip_name") else False if key == "zip_processed" else None

# ====================================================================
st.markdown("""
<style>
    /* ========== CSS 自定义属性（主题色） ========== */
    :root {
        --bg-primary: #0b0e1a;
        --bg-card: rgba(255, 255, 255, 0.04);
        --bg-card-hover: rgba(255, 255, 255, 0.07);
        --bg-card-glass: rgba(255, 255, 255, 0.05);
        --border-card: rgba(255, 255, 255, 0.06);
        --border-card-hover: rgba(255, 255, 255, 0.12);
        --text-primary: #e0e0e0;
        --text-secondary: #c8cdd8;
        --text-muted: #8892b0;
        --text-heading: #f0f4ff;
        --sidebar-bg: rgba(11, 14, 26, 0.98);
        --table-th-bg: rgba(255,255,255,0.06);
        --table-td-border: rgba(255,255,255,0.04);
        --table-th-border: rgba(255,255,255,0.08);
        --table-hover: rgba(255,255,255,0.03);
        --upload-bg: rgba(255,255,255,0.03);
        --upload-border: rgba(255,255,255,0.12);
        --scroll-thumb: #2a2d40;
        --scroll-thumb-hover: #3a3d55;
        --loader-bg: rgba(255, 255, 255, 0.04);
        --loader-border: rgba(255, 255, 255, 0.08);
        --ai-box-bg: rgba(64, 196, 255, 0.06);
        --ai-box-border: rgba(64, 196, 255, 0.15);
        --divider-color: rgba(255,255,255,0.08);
        --input-bg: rgba(255,255,255,0.06);
        --btn-bg: rgba(255,255,255,0.06);
        --alert-bg: rgba(255,255,255,0.04);
        --sidebar-text: #c8cdd8;
        --caption-color: #8892b0;
        --expander-bg: rgba(255,255,255,0.04);
    }
    /* ---------- 全局 ---------- */
    html, body, #root {{
        background: var(--bg-primary) !important;
    }}
    .stApp {{
        background: var(--bg-primary) !important;
        color: var(--text-primary);
    }}
    .block-container {{
        padding: 1.5rem 2rem !important;
        max-width: 1400px;
        margin: 0 auto;
    }}
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
    ::-webkit-scrollbar-thumb {{ background: var(--scroll-thumb); border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--scroll-thumb-hover); }}

    /* ---------- 标题文字 ---------- */
    h1, h2, h3, h4 {{
        color: var(--text-heading) !important;
        letter-spacing: 0.3px;
    }}
    p, li, .stMarkdown {{
        color: var(--text-secondary);
    }}

    /* ---------- 标题横幅 ---------- */

/* ---------- 结果卡片 ---------- */
    .plate-card {{
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-card);
        border-radius: 16px;
        padding: 18px 16px;
        margin: 8px 0;
        transition: all 0.25s ease;
    }}
    .plate-card:hover {{
        background: var(--bg-card-hover);
        border-color: var(--border-card-hover);
    }}

    /* 车牌号码大号展示 */
    .plate-number {{
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 3px;
        font-family: 'Courier New', monospace;
        text-align: center;
        padding: 8px 0;
    }}
    .plate-number.blue {{ color: #4fc3f7; }}
    .plate-number.green {{ color: #81c784; }}
    .plate-number.yellow {{ color: #ffd54f; }}
    .plate-number.default {{ color: var(--text-primary); }}

    .badge {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }}
    .badge-high   {{ background: rgba(100, 255, 218, 0.15); color: #64ffda; border: 1px solid rgba(100,255,218,0.3); }}
    .badge-medium {{ background: rgba(255, 213, 79, 0.15);  color: #ffd54f; border: 1px solid rgba(255,213,79,0.3); }}
    .badge-low    {{ background: rgba(255, 82, 82, 0.15);   color: #ff5252; border: 1px solid rgba(255,82,82,0.3); }}

/* ---------- 加载动画 ---------- */
    @keyframes dot-bounce {{
        0%, 80%, 100% {{ transform: scale(0); }}
        40% {{ transform: scale(1); }}
    }}
    .scan-loader {{
        display: flex;
        align-items: center;
        gap: 14px;
        background: var(--loader-bg);
        border: 1px solid var(--loader-border);
        border-radius: 12px;
        padding: 14px 20px;
        margin: 10px 0;
        backdrop-filter: blur(8px);
    }}
    .scan-dots {{
        display: flex;
        gap: 5px;
    }}
    .scan-dots span {{
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        animation: dot-bounce 1.4s ease-in-out infinite both;
    }}
    .scan-dots span:nth-child(1) {{ background: #64ffda; animation-delay: -0.32s; }}
    .scan-dots span:nth-child(2) {{ background: #40c4ff; animation-delay: -0.16s; }}
    .scan-dots span:nth-child(3) {{ background: #b388ff; animation-delay: 0s; }}
    .scan-text {{
        color: var(--text-muted);
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }}
    .scan-text.active {{ color: #64ffda; }}
    .scan-text.blue   {{ color: #40c4ff; }}

    /* ---------- 侧边栏 ---------- */
    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-card);
    }}
    [data-testid="stSidebar"] .stMarkdown {{
        color: var(--text-secondary);
    }}


    /* ---------- 消息框 ---------- */
    .stAlert {{ border-radius: 10px !important; border: none !important; }}
    [data-testid="stAlert"] {{ padding: 12px 16px !important; }}
    div[data-baseweb="alert"] {{ border-radius: 10px !important; }}

    /* ---------- 按钮 ---------- */
    .stButton button {{
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: 1px solid var(--border-card) !important;
    }}
    .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }}

    /* ---------- 上传组件 ---------- */
    [data-testid="stFileUploader"] {{
        background: var(--upload-bg);
        border: 1px dashed var(--upload-border);
        border-radius: 14px;
        padding: 10px;
        transition: border-color 0.3s;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: rgba(100,255,218,0.3);
    }}

    /* ---------- 分割线 ---------- */
    .section-divider {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--divider-color), transparent);
        margin: 28px 0;
    }}

    /* ---------- 响应式 ---------- */
    @media (max-width: 768px) {{
        .block-container {{ padding: 1rem 1rem !important; }}
        .plate-number {{ font-size: 22px; letter-spacing: 2px; }}
        .plate-card {{ padding: 14px 10px; }}
    @media (max-width: 480px) {{
        .block-container {{ padding: 0.6rem 0.6rem !important; }}
        .plate-number {{ font-size: 18px; }}
    }}

    /* ================================================================
       以下覆盖 Streamlit 原生组件，确保亮色模式彻底无暗色残留
       使用双倍 class (e.g. .stTextInput.stTextInput) 提高特异性
       ================================================================ */

    /* ---------- 顶部黑杠 ---------- */
    @media (min-width: 769px) {{
        header[data-testid="stHeader"] {{
            display: none !important;
        }}
        #stDecoration, .stDecoration {{
            display: none !important;
        }}
    }}
    @media (max-width: 768px) {{
        header[data-testid="stHeader"] {{
            display: flex !important;
            background: var(--bg-primary) !important;
            backdrop-filter: blur(10px) !important;
            border: none !important;
            height: 40px !important;
            min-height: 40px !important;
        }}
        header[data-testid="stHeader"] button {{
            color: var(--text-primary) !important;
        }}
        header[data-testid="stHeader"] svg {{
            fill: var(--text-primary) !important;
        }}
    }}
    .stApp {{
        margin-top: 0 !important;
    }}

    /* ---------- 主内容区 ---------- */
    .main > div {{ background: transparent !important; }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span {{
        color: var(--sidebar-text) !important;
    }}
    section[data-testid="stSidebar"] .stCaption {{
        color: var(--caption-color) !important;
    }}

    /* ---------- 文本输入框 ---------- */
    .stTextInput.stTextInput {{
        background: transparent !important;
    }}
    .stTextInput.stTextInput > div {{
        background-color: var(--input-bg) !important;
        border-radius: 10px !important;
        border: 1px solid var(--border-card) !important;
    }}
    .stTextInput.stTextInput > div > div {{
        background-color: transparent !important;
    }}
    .stTextInput.stTextInput input {{
        background-color: transparent !important;
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
        caret-color: var(--text-primary) !important;
        border: none !important;
        box-shadow: none !important;
    }}
    .stTextInput.stTextInput input::placeholder {{
        color: var(--text-muted) !important;
        -webkit-text-fill-color: var(--text-muted) !important;
        opacity: 0.7 !important;
    }}
    .stTextInput.stTextInput label {{
        color: var(--text-secondary) !important;
    }}
    .stTextInput [data-testid="stTextInputVisibilityToggle"] {{
        color: var(--text-secondary) !important;
        opacity: 0.65 !important;
    }}
    .stTextInput [data-testid="stTextInputVisibilityToggle"] *,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg *,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg path,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg circle,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg line,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg polygon {{
        fill: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
        stroke: var(--text-secondary) !important;
    }}
    .stTextInput [data-testid="stTextInputVisibilityToggle"]:hover {{
        opacity: 1 !important;
    }}

    /* ---------- 滑块 ---------- */
    .stSlider.stSlider label {{
        color: var(--text-secondary) !important;
    }}
    .stSlider.stSlider div[data-testid="stTickBar"] {{
        color: var(--text-muted) !important;
    }}

    /* ---------- 按钮 ---------- */
    .stButton.stButton button,
    .stDownloadButton.stDownloadButton button {{
        background: var(--btn-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 10px !important;
    }}
    .stButton.stButton button:hover,
    .stDownloadButton.stDownloadButton button:hover {{
        background: var(--bg-card-hover) !important;
        border-color: var(--border-card-hover) !important;
    }}

    /* ---------- Alert ---------- */
    .stAlert.stAlert {{
        background: var(--alert-bg) !important;
        backdrop-filter: blur(8px);
        color: var(--text-secondary) !important;
    }}

    /* ---------- 图片 caption ---------- */
    .stImage.stImage figcaption {{
        color: var(--caption-color) !important;
        font-size: 13px !important;
    }}

    /* ---------- 展开器 ---------- */
    .streamlit-expanderHeader {{
        background: var(--expander-bg) !important;
        color: var(--text-primary) !important;
    }}
    .streamlit-expanderHeader:hover {{
        background: var(--bg-card-hover) !important;
    }}
    .streamlit-expanderContent {{
        background: transparent !important;
    }}

    /* ---------- 选择框 ---------- */
    .stSelectbox.stSelectbox div[data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        border: 1px solid var(--border-card) !important;
    }}

    /* ---------- 分隔线 ---------- */
    hr {{
        border-color: var(--divider-color) !important;
    }}

    /* ---------- 文件上传器 ---------- */
    [data-testid="stFileUploader"] {{
        background: var(--upload-bg) !important;
    }}
    [data-testid="stFileUploader"] section {{
        color: var(--text-secondary) !important;
    }}
    [data-testid="stFileUploader"] button {{
        background: var(--btn-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-card) !important;
    }}
    [data-testid="stFileUploader"] div {{
        background: transparent !important;
    }}
    [data-testid="stFileUploader"] [data-testid="stFileUploadDragzone"] {{
        background: var(--upload-bg) !important;
        border: 1px dashed var(--upload-border) !important;
    }}

    /* ---------- spinner ---------- */
    .stSpinner.stSpinner > div {{
        border-color: var(--text-muted) !important;
    }}

    /* ---------- tooltip ---------- */
    [data-baseweb="tooltip"] {{
        background: var(--bg-card-glass) !important;
        backdrop-filter: blur(12px);
        color: var(--text-primary) !important;
    }}

    /* ---------- 表格 ---------- */
    [data-testid="stDataFrame"] {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] > div:first-child {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] table {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] thead {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] th {{
        background: var(--table-th-bg) !important;
        color: var(--text-muted) !important;
        border-bottom: 1px solid var(--table-th-border) !important;
    }}
    [data-testid="stDataFrame"] td {{
        background: var(--bg-primary) !important;
        color: var(--text-secondary) !important;
        border-bottom: 1px solid var(--table-td-border) !important;
    }}
    [data-testid="stDataFrame"] tr {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] tbody {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] tr:hover td {{
        background: var(--table-hover) !important;
    }}
    [data-testid="stDataFrame"] [data-testid="StyledVirtuosoItem"] {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] [data-testid="StyledVirtuosoItem"] td {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] div[data-testid="StyledVirtuoso"] {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] [data-testid="StyledVirtuoso"] > div {{
        background: var(--bg-primary) !important;
    }}
    [data-testid="stDataFrame"] [class] {{
        background: transparent !important;
    }}
    [data-testid="stDataFrame"] div {{
        background: transparent !important;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ── 标题居中修复（原CSS因双花括号问题失效） ── */
.app-header {
    text-align: center !important;
    padding: 28px 0 12px 0 !important;
    margin-bottom: 20px !important;
    position: relative !important;
}
.app-header h1 {
    font-size: 38px !important;
    font-weight: 800 !important;
    margin: 0 !important;
    color: #7ec8e3 !important;
    letter-spacing: 2px !important;
}
.app-header .subtitle {
    color: #a0a0b8 !important;
    font-size: 14px !important;
    margin: 8px 0 0 0 !important;
    letter-spacing: 1px !important;
}
@media (max-width: 768px) {
    .app-header { padding: 16px 0 4px 0 !important; margin-bottom: 8px !important; }
    .app-header h1 { font-size: 26px !important; white-space: nowrap !important; letter-spacing: 1px !important; }
    .app-header .subtitle { font-size: 11px !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ── 识别结果卡片（更醒目的布局） ── */
.result-card {
    border: 1px solid var(--border-card) !important;
    background: var(--bg-card) !important;
    margin: 16px 0 !important;
    border-radius: 16px !important;
    padding: 20px 16px !important;
}
.result-header {
    text-align: center;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid var(--divider-color);
    margin-bottom: 16px;
}
.result-header .plate-number {
    font-size: 36px !important;
    letter-spacing: 4px !important;
}
.result-metrics {
    display: flex !important;
    justify-content: space-around !important;
    text-align: center !important;
    gap: 8px !important;
}
.result-metric {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    flex: 1;
}
.metric-icon { font-size: 22px; }
.metric-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}
.metric-value {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
}
@media (max-width: 768px) {
    .result-header .plate-number { font-size: 28px !important; letter-spacing: 2px !important; }
    .metric-icon { font-size: 18px; }
    .metric-label { font-size: 10px; }
    .metric-value { font-size: 13px; }
}

/* ── AI 分析加载动画（脉冲环 + 流光进度条） ── */
.ai-loader {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
    background: rgba(64, 196, 255, 0.05) !important;
    border: 1px solid rgba(64, 196, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin: 10px 0 !important;
    position: relative !important;
    overflow: hidden !important;
}
.ai-pulse-ring {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 2px solid #40c4ff;
    animation: ai-pulse 1.5s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes ai-pulse {
    0% { transform: scale(0.8); opacity: 0.6; }
    50% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(0.8); opacity: 0.6; }
}
.ai-loader-content {
    display: flex;
    flex-direction: column;
    gap: 10px;
    flex: 1;
}
.ai-loader-text {
    color: #40c4ff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    animation: ai-text-glow 2s ease-in-out infinite;
}
@keyframes ai-text-glow {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; text-shadow: 0 0 8px rgba(64,196,255,0.3); }
}
.ai-loader-bar {
    height: 3px;
    background: rgba(64, 196, 255, 0.15);
    border-radius: 2px;
    overflow: hidden;
    width: 100%;
}
.ai-loader-fill {
    height: 100%;
    width: 30%;
    background: linear-gradient(90deg, transparent, #40c4ff, transparent);
    border-radius: 2px;
    animation: ai-loading-bar 1.8s ease-in-out infinite;
}
@keyframes ai-loading-bar {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(400%); }
}

/* ── AI 分析结果框放大 ── */
.ai-box {
    background: var(--ai-box-bg) !important;
    border: 1px solid var(--ai-box-border) !important;
    border-left: 4px solid #40c4ff !important;
    border-radius: 12px !important;
    padding: 18px 22px !important;
    margin: 12px 0 !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
    color: var(--text-primary) !important;
}
.ai-box strong {
    color: #40c4ff !important;
    font-size: 18px !important;
}
@media (max-width: 768px) {
    .ai-box { font-size: 15px !important; padding: 14px 16px !important; }
    .ai-box strong { font-size: 16px !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ================================================================
   CSS 变量修复
   ================================================================ */

/* 暗色主题 */
:root, html[data-theme="dark"] {
    --bg-primary: #0b0e1a;
    --bg-card: rgba(255, 255, 255, 0.04);
    --bg-card-hover: rgba(255, 255, 255, 0.07);
    --bg-card-glass: rgba(255, 255, 255, 0.05);
    --border-card: rgba(255, 255, 255, 0.06);
    --border-card-hover: rgba(255, 255, 255, 0.12);
    --text-primary: #e0e0e0;
    --text-secondary: #c8cdd8;
    --text-muted: #8892b0;
    --text-heading: #f0f4ff;
    --sidebar-bg: rgba(11, 14, 26, 0.98);
    --table-th-bg: rgba(255,255,255,0.06);
    --table-td-border: rgba(255,255,255,0.04);
    --table-th-border: rgba(255,255,255,0.08);
    --table-hover: rgba(255,255,255,0.03);
    --upload-bg: rgba(255,255,255,0.03);
    --upload-border: rgba(255,255,255,0.12);
    --scroll-thumb: #2a2d40;
    --scroll-thumb-hover: #3a3d55;
    --loader-bg: rgba(255, 255, 255, 0.04);
    --loader-border: rgba(255, 255, 255, 0.08);
    --ai-box-bg: rgba(64, 196, 255, 0.06);
    --ai-box-border: rgba(64, 196, 255, 0.15);
    --divider-color: rgba(255,255,255,0.08);
    --input-bg: rgba(255,255,255,0.06);
    --btn-bg: rgba(255,255,255,0.06);
    --alert-bg: rgba(255,255,255,0.04);
    --sidebar-text: #c8cdd8;
    --caption-color: #8892b0;
    --expander-bg: rgba(255,255,255,0.04);
}
/* 亮色主题 */
html[data-theme="light"] {
    --bg-primary: #ffffff;
    --bg-card: rgba(0, 0, 0, 0.02);
    --bg-card-hover: rgba(0, 0, 0, 0.05);
    --bg-card-glass: rgba(0, 0, 0, 0.03);
    --border-card: rgba(0, 0, 0, 0.08);
    --border-card-hover: rgba(0, 0, 0, 0.15);
    --text-primary: #1a1a1a;
    --text-secondary: #4a4a4a;
    --text-muted: #6b7280;
    --text-heading: #0f172a;
    --sidebar-bg: #f8fafc;
    --table-th-bg: rgba(0,0,0,0.04);
    --table-td-border: rgba(0,0,0,0.06);
    --table-th-border: rgba(0,0,0,0.1);
    --table-hover: rgba(0,0,0,0.02);
    --upload-bg: rgba(0,0,0,0.02);
    --upload-border: rgba(0,0,0,0.15);
    --scroll-thumb: #d1d5db;
    --scroll-thumb-hover: #9ca3af;
    --loader-bg: rgba(0, 0, 0, 0.02);
    --loader-border: rgba(0, 0, 0, 0.08);
    --ai-box-bg: rgba(64, 196, 255, 0.08);
    --ai-box-border: rgba(64, 196, 255, 0.2);
    --divider-color: rgba(0,0,0,0.1);
    --input-bg: rgba(0,0,0,0.04);
    --btn-bg: rgba(0,0,0,0.04);
    --alert-bg: rgba(0,0,0,0.02);
    --sidebar-text: #4a4a4a;
    --caption-color: #6b7280;
    --expander-bg: rgba(0,0,0,0.02);
}

/* 修复 badge */
.result-metric .badge {
    display: inline-block !important;
    padding: 3px 12px !important;
    border-radius: 20px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border: 1px solid transparent !important;
}
.result-metric .badge-high {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #059669 !important;
    border-color: rgba(16, 185, 129, 0.3) !important;
}
.result-metric .badge-medium {
    background: rgba(245, 158, 11, 0.15) !important;
    color: #d97706 !important;
    border-color: rgba(245, 158, 11, 0.3) !important;
}
.result-metric .badge-low {
    background: rgba(239, 68, 68, 0.15) !important;
    color: #dc2626 !important;
    border-color: rgba(239, 68, 68, 0.3) !important;
}

/* 修复 metric-value */
.result-metric .metric-value {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-header"><h1>🚗 AI 车牌识别系统</h1><p style="color:#8892b0;margin:0">精准识别 · 手动修正 · 批量处理 · 智能分析</p></div>', unsafe_allow_html=True)

# ====================================================================
# 侧边栏
# ====================================================================
with st.sidebar:
    st.markdown("### ⚙️ 设置")
    st.text_input("DeepSeek API Key", type="password", placeholder="sk-...（空则跳过AI分析）", key="api_key")
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.6, 0.05)
    st.markdown("---")
    st.caption(f"📋 已识别 **{len(st.session_state.history)}** 条记录")

    if st.button("🗑️ 清空记录", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("edit_") or key.startswith("export_"):
                del st.session_state[key]
        st.session_state.history = []
        st.session_state.uploader_key += 1
        st.session_state.files_processed = False
        st.session_state.results_cache = []
        st.session_state.zip_cache = []
        st.session_state.last_file_count = 0
        st.session_state.zip_processed = False
        st.session_state.zip_name = None
        st.rerun()

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        df_export = df.drop(columns=["id", "file_md5"], errors="ignore")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 CSV", df_export.to_csv(index=False, encoding="utf-8-sig").encode(), "车牌识别记录.csv", use_container_width=True)
        with col2:
            st.download_button("📥 Excel", to_excel(df), "车牌识别记录.xlsx", use_container_width=True)

# ====================================================================
# 核心 Tab
# ====================================================================
tab1, tab2, tab3 = st.tabs(["📷 图片识别", "📦 ZIP批量", "📊 统计筛选"])

# ---------- Tab1: 图片识别 ----------
with tab1:
    st.markdown('<a id="tab1-top"></a>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "选择图片（支持批量 JPG / PNG）",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    current_file_count = len(uploaded_files) if uploaded_files else 0
    if current_file_count != st.session_state.last_file_count:
        st.session_state.files_processed = False
        st.session_state.last_file_count = current_file_count

    # 文件全部移除 → 清理缓存
    if not uploaded_files and st.session_state.results_cache:
        st.session_state.results_cache = []
        st.session_state.zip_cache = []

    if uploaded_files and not st.session_state.files_processed:
        st.session_state.results_cache = []
        lpr = load_lpr()
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file in enumerate(uploaded_files):
            status_text.text(f"正在处理: {file.name} ({idx+1}/{len(uploaded_files)})")
            progress_bar.progress((idx + 1) / len(uploaded_files))

            file_bytes = file.read()
            file_md5 = get_file_hash(file_bytes)
            img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                try:
                    img_bgr = cv2.cvtColor(np.array(Image.open(BytesIO(file_bytes))), cv2.COLOR_RGB2BGR)
                except Exception:
                    st.error(f"⚠️ [{file.name}] 无法解析")
                    continue

            img_bgr = resize_image_if_needed(img_bgr)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            draw_img = img_rgb.copy()

            # 扫描动画
            loader = st.empty()
            loader.markdown("""
            <div class="scan-loader">
                <div class="scan-dots"><span></span><span></span><span></span></div>
                <span class="scan-text active">🔍 扫描车牌中</span>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner(""):
                res_list = parse_lpr_results(lpr(img_bgr))
            loader.empty()

            if not res_list:
                st.error(f"⚠️ [{file.name}] 未检测到车牌")
                continue

            plates_info = []
            for i, one_plate in enumerate(res_list):
                draw_img = draw_plate_box(draw_img, one_plate)
                fmt_plate, raw, conf, ptype, addr = smart_plate_parser(
                    one_plate["plate"], one_plate["color"], conf_threshold, one_plate["confidence"]
                )
                if not fmt_plate:
                    continue
                plates_info.append({
                    "fmt_plate": fmt_plate, "raw": raw, "conf": conf,
                    "ptype": ptype, "addr": addr,
                    "color": one_plate["color"] if one_plate["color"] in ("blue", "green", "yellow") else "default",
                    "box": one_plate["box"], "file_md5": file_md5, "idx": i,
                })

            with st.expander(f"📷 {file.name}", expanded=True):
                st.image(draw_img, caption=f"🎯 {file.name}", use_container_width=True)

                for info in plates_info:
                    render_plate_card(info["fmt_plate"], info["color"], info["conf"], info["ptype"], info["addr"])

                    # 编辑修正
                    edit_key = f"edit_{info['file_md5']}_{info['idx']}"
                    if st.button(f"✏️ 编辑修正", key=f"btn_{edit_key}", use_container_width=True):
                        st.session_state[edit_key] = True
                        st.rerun()
                    if st.session_state.get(edit_key, False):
                        new_plate = st.text_input("修正车牌号", value=info["fmt_plate"], key=f"input_{edit_key}")
                        col_ok, col_cancel = st.columns(2)
                        with col_ok:
                            if st.button("✅ 确认修改", key=f"ok_{edit_key}"):
                                new_clean = new_plate.replace('·', '').strip()
                                new_fmt = f"{new_clean[:2]}·{new_clean[2:]}" if len(new_clean) > 2 else new_plate
                                # 按 hist_id 精准匹配历史记录
                                target_id = info.get("hist_id", "")
                                for h in st.session_state.history:
                                    if target_id and h.get("id") == target_id:
                                        h["号牌"] = new_fmt
                                        break
                                    elif not target_id and h.get("file_md5") == info["file_md5"] and h["原始号牌"] == info["raw"]:
                                        h["号牌"] = new_fmt
                                # 更新缓存中的 fmt_plate
                                for cache in st.session_state.results_cache:
                                    if cache["file_md5"] == info["file_md5"]:
                                        for p in cache["plates"]:
                                            if (target_id and p.get("hist_id") == target_id) or \
                                               (not target_id and p["fmt_plate"] == info["fmt_plate"]):
                                                p["fmt_plate"] = new_fmt
                                st.session_state[edit_key] = False
                                st.success("已修正！")
                                st.rerun()
                        with col_cancel:
                            if st.button("❌ 取消", key=f"cancel_{edit_key}"):
                                st.session_state[edit_key] = False
                                st.rerun()

                    # AI 分析动效
                    if st.session_state.api_key.strip():
                        loader2 = st.empty()
                        loader2.markdown("""
                        <div class="ai-loader">
                            <div class="ai-pulse-ring"></div>
                            <div class="ai-loader-content">
                                <span class="ai-loader-text">🧠 AI 智能分析中</span>
                                <div class="ai-loader-bar"><div class="ai-loader-fill"></div></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        ai_ret = deepseek_analyze(f"车牌号：{info['fmt_plate']}，类型：{info['ptype']}", st.session_state.api_key.strip())
                        loader2.empty()
                        st.markdown(f'<div class="ai-box"><strong>🧠 AI 分析</strong><br>{ai_ret}</div>', unsafe_allow_html=True)

                    if not any(h.get("file_md5") == info["file_md5"] and h["原始号牌"] == info["raw"] for h in st.session_state.history):
                        entry_id = str(uuid.uuid4())
                        st.session_state.history.append({
                            "id": entry_id,
                            "file_md5": info["file_md5"],
                            "图片名称": file.name,
                            "号牌": info["fmt_plate"],
                            "原始号牌": info["raw"],
                            "置信度": round(info["conf"], 2),
                            "车辆类型": info["ptype"],
                            "离线属地": info["addr"]
                        })
                        info["hist_id"] = entry_id

            # 缓存已标注图
            _, compressed = cv2.imencode('.jpg', cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            st.session_state.results_cache.append({
                "file_name": file.name,
                "file_md5": file_md5,
                "cached_img_bytes": compressed.tobytes(),
                "plates": [{
                    "box": p["box"], "fmt_plate": p["fmt_plate"],
                    "raw": p["raw"], "conf": p["conf"],
                    "ptype": p["ptype"], "addr": p["addr"],
                    "color": p["color"], "file_md5": p["file_md5"], "idx": p["idx"],
                    "hist_id": p.get("hist_id", ""),
                } for p in plates_info],
            })

        status_text.text("✅ 全部处理完成！")
        st.session_state.files_processed = True
        st.rerun()

    elif st.session_state.results_cache:
        for result in st.session_state.results_cache:
            with st.expander(f"📷 {result['file_name']}", expanded=True):
                img_bgr = cv2.imdecode(np.frombuffer(result["cached_img_bytes"], np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)

                for info in result["plates"]:
                    render_plate_card(info["fmt_plate"], info["color"], info["conf"], info["ptype"], info["addr"])

                    # 编辑修正
                    edit_key = f"edit_{info['file_md5']}_{info['idx']}"
                    if st.button(f"✏️ 编辑修正", key=f"btn_{edit_key}", use_container_width=True):
                        st.session_state[edit_key] = True
                        st.rerun()
                    if st.session_state.get(edit_key, False):
                        new_plate = st.text_input("修正车牌号", value=info["fmt_plate"], key=f"input_{edit_key}")
                        col_ok, col_cancel = st.columns(2)
                        with col_ok:
                            if st.button("✅ 确认修改", key=f"ok_{edit_key}"):
                                new_clean = new_plate.replace('·', '').strip()
                                new_fmt = f"{new_clean[:2]}·{new_clean[2:]}" if len(new_clean) > 2 else new_plate
                                target_id = info.get("hist_id", "")
                                for h in st.session_state.history:
                                    if target_id and h.get("id") == target_id:
                                        h["号牌"] = new_fmt
                                        break
                                    elif not target_id and h.get("file_md5") == info["file_md5"] and h["原始号牌"] == info["raw"]:
                                        h["号牌"] = new_fmt
                                for cache in st.session_state.results_cache:
                                    if cache["file_md5"] == info["file_md5"]:
                                        for p in cache["plates"]:
                                            if (target_id and p.get("hist_id") == target_id) or \
                                               (not target_id and p["fmt_plate"] == info["fmt_plate"]):
                                                p["fmt_plate"] = new_fmt
                                st.session_state[edit_key] = False
                                st.success("已修正！")
                                st.rerun()
                        with col_cancel:
                            if st.button("❌ 取消", key=f"cancel_{edit_key}"):
                                st.session_state[edit_key] = False
                                st.rerun()

                    # AI 分析动效
                    if st.session_state.api_key.strip():
                        loader2 = st.empty()
                        loader2.markdown("""
                        <div class="ai-loader">
                            <div class="ai-pulse-ring"></div>
                            <div class="ai-loader-content">
                                <span class="ai-loader-text">🧠 AI 智能分析中</span>
                                <div class="ai-loader-bar"><div class="ai-loader-fill"></div></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        ai_ret = deepseek_analyze(f"车牌号：{info['fmt_plate']}，类型：{info['ptype']}", st.session_state.api_key.strip())
                        loader2.empty()
                        st.markdown(f'<div class="ai-box"><strong>🧠 AI 分析</strong><br>{ai_ret}</div>', unsafe_allow_html=True)

    # ---------- 历史记录表 ----------
    if st.session_state.history and (st.session_state.files_processed or st.session_state.results_cache):
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        col_title, col_export = st.columns([3, 1])
        with col_title:
            st.markdown("### 📋 识别记录表")
        with col_export:
            df_export = pd.DataFrame(st.session_state.history)
            csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode()
            st.download_button("📥 导出 CSV", csv_bytes, "车牌识别记录.csv", mime="text/csv", use_container_width=True, key="dl_csv_tab1")
        st.dataframe(st.session_state.history, use_container_width=True, height=280)
        st.markdown('<div style="text-align:center;margin-top:12px"><a href="#tab1-top" style="color:#888;text-decoration:none;font-size:13px">⬆ 回到顶部</a></div>', unsafe_allow_html=True)

# ---------- Tab2: ZIP批量 ----------
with tab2:
    st.markdown("### 📦 ZIP批量处理")
    uploaded_zip = st.file_uploader("上传包含图片的ZIP压缩包", type="zip")
    if uploaded_zip:
        if uploaded_zip.name != st.session_state.zip_name:
            st.session_state.zip_processed = False
            st.session_state.zip_name = uploaded_zip.name
            st.session_state.zip_cache = []

        # ---- 未处理 → 显示开始按钮 ----
        if not st.session_state.zip_processed:
            with zipfile.ZipFile(uploaded_zip, 'r') as zf:
                image_files = [f for f in zf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                st.info(f"发现 {len(image_files)} 张图片")
            if st.button("🚀 开始批量识别", type="primary"):
                st.session_state.zip_processed = True
                lpr = load_lpr()
                progress_bar = st.progress(0)
                status_text = st.empty()
                with zipfile.ZipFile(uploaded_zip, 'r') as zf:
                    image_files = [f for f in zf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    for idx, img_name in enumerate(image_files):
                        status_text.text(f"正在处理: {img_name} ({idx+1}/{len(image_files)})")
                        progress_bar.progress((idx + 1) / len(image_files))
                        with zf.open(img_name) as f:
                            file_bytes = f.read()
                            file_md5 = get_file_hash(file_bytes)
                            img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                            if img_bgr is None:
                                st.error(f"⚠️ [{img_name}] 无法解析")
                                continue
                            img_bgr = resize_image_if_needed(img_bgr)
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            draw_img = img_rgb.copy()
                            loader = st.empty()
                            loader.markdown("""
                            <div class="scan-loader">
                                <div class="scan-dots"><span></span><span></span><span></span></div>
                                <span class="scan-text active">🔍 扫描车牌中</span>
                            </div>
                            """, unsafe_allow_html=True)
                            res_list = parse_lpr_results(lpr(img_bgr))
                            loader.empty()
                            plates_info = []
                            for i, one_plate in enumerate(res_list):
                                draw_img = draw_plate_box(draw_img, one_plate)
                                fmt_plate, raw, conf, ptype, addr = smart_plate_parser(
                                    one_plate["plate"], one_plate["color"], conf_threshold, one_plate["confidence"]
                                )
                                if fmt_plate:
                                    plates_info.append({
                                        "fmt_plate": fmt_plate, "raw": raw, "conf": conf,
                                        "ptype": ptype, "addr": addr,
                                        "color": one_plate["color"] if one_plate["color"] in ("blue", "green", "yellow") else "default",
                                        "box": one_plate["box"], "file_md5": file_md5, "idx": i,
                                    })
                            # 缓存带框图片
                            _, compressed = cv2.imencode('.jpg', cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                            # 创建历史记录 + 记录 hist_id
                            for p in plates_info:
                                if not any(h.get("file_md5") == file_md5 and h["原始号牌"] == p["raw"] for h in st.session_state.history):
                                    entry_id = str(uuid.uuid4())
                                    st.session_state.history.append({
                                        "id": entry_id,
                                        "file_md5": file_md5,
                                        "图片名称": img_name,
                                        "号牌": p["fmt_plate"],
                                        "原始号牌": p["raw"],
                                        "置信度": round(p["conf"], 2),
                                        "车辆类型": p["ptype"],
                                        "离线属地": p["addr"]
                                    })
                                    p["hist_id"] = entry_id
                            st.session_state.zip_cache.append({
                                "file_name": img_name,
                                "file_md5": file_md5,
                                "cached_img_bytes": compressed.tobytes(),
                                "plates": [{
                                    "box": pp["box"], "fmt_plate": pp["fmt_plate"],
                                    "raw": pp["raw"], "conf": pp["conf"],
                                    "ptype": pp["ptype"], "addr": pp["addr"],
                                    "color": pp["color"], "file_md5": pp["file_md5"], "idx": pp["idx"],
                                    "hist_id": pp.get("hist_id", ""),
                                } for pp in plates_info],
                            })
                status_text.text("✅ 全部处理完成！")
                st.rerun()

        # ---- 已处理 → 展示结果 ----
        if st.session_state.zip_cache:
            st.markdown("---")
            for result in st.session_state.zip_cache:
                with st.expander(f"📷 {result['file_name']}", expanded=False):
                    img_bgr = cv2.imdecode(np.frombuffer(result["cached_img_bytes"], np.uint8), cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                    for info in result["plates"]:
                        render_plate_card(info["fmt_plate"], info["color"], info["conf"], info["ptype"], info["addr"])
                        edit_key = f"edit_zip_{info['file_md5']}_{info['idx']}"
                        if st.button(f"✏️ 编辑修正", key=f"btn_{edit_key}", use_container_width=True):
                            st.session_state[edit_key] = True
                            st.rerun()
                        if st.session_state.get(edit_key, False):
                            new_plate = st.text_input("修正车牌号", value=info["fmt_plate"], key=f"input_{edit_key}")
                            col_ok, col_cancel = st.columns(2)
                            with col_ok:
                                if st.button("✅ 确认修改", key=f"ok_{edit_key}"):
                                    new_clean = new_plate.replace('·', '').strip()
                                    new_fmt = f"{new_clean[:2]}·{new_clean[2:]}" if len(new_clean) > 2 else new_plate
                                    target_id = info.get("hist_id", "")
                                    for h in st.session_state.history:
                                        if target_id and h.get("id") == target_id:
                                            h["号牌"] = new_fmt
                                            break
                                        elif not target_id and h.get("file_md5") == info["file_md5"] and h["原始号牌"] == info["raw"]:
                                            h["号牌"] = new_fmt
                                    for cache in st.session_state.zip_cache:
                                        if cache["file_md5"] == info["file_md5"]:
                                            for p in cache["plates"]:
                                                if (target_id and p.get("hist_id") == target_id) or \
                                                   (not target_id and p["fmt_plate"] == info["fmt_plate"]):
                                                    p["fmt_plate"] = new_fmt
                                    st.session_state[edit_key] = False
                                    st.success("已修正！")
                                    st.rerun()
                            with col_cancel:
                                if st.button("❌ 取消", key=f"cancel_{edit_key}"):
                                    st.session_state[edit_key] = False
                                    st.rerun()
                        # AI 分析
                        if st.session_state.api_key.strip():
                            loader2 = st.empty()
                            loader2.markdown("""
                            <div class="ai-loader">
                                <div class="ai-pulse-ring"></div>
                                <div class="ai-loader-content">
                                    <span class="ai-loader-text">🧠 AI 智能分析中</span>
                                    <div class="ai-loader-bar"><div class="ai-loader-fill"></div></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            ai_ret = deepseek_analyze(f"车牌号：{info['fmt_plate']}，类型：{info['ptype']}", st.session_state.api_key.strip())
                            loader2.empty()
                            st.markdown(f'<div class="ai-box"><strong>🧠 AI 分析</strong><br>{ai_ret}</div>', unsafe_allow_html=True)
            # 历史记录表
            if st.session_state.history:
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                col_title, col_export = st.columns([3, 1])
                with col_title:
                    st.markdown("### 📋 识别记录表")
                with col_export:
                    df_export = pd.DataFrame(st.session_state.history)
                    csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode()
                    st.download_button("📥 导出 CSV", csv_bytes, "车牌识别记录.csv", mime="text/csv", use_container_width=True, key="dl_csv_tab2")
                st.dataframe(st.session_state.history, use_container_width=True, height=280)

# ---------- Tab3: 统计筛选 ----------
with tab3:
    st.markdown("### 📊 统计分析与筛选")
    if not st.session_state.history:
        st.info("暂无识别记录，请先上传图片识别")
    else:
        df = pd.DataFrame(st.session_state.history)
        df_display = df.drop(columns=["id", "file_md5"], errors="ignore")
        if len(df_display) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("总识别数", len(df_display))
            with col2: st.metric("新能源占比", f"{df_display['车辆类型'].str.contains('新能源').mean():.1%}")
            with col3: st.metric("平均置信度", f"{df_display['置信度'].mean():.1%}")
            with col4: st.metric("涉及属地", df_display["离线属地"].nunique())
        st.subheader("🔍 筛选记录")
        col1, col2, col3 = st.columns(3)
        with col1: addr_filter = st.multiselect("属地筛选", sorted(df_display["离线属地"].unique()))
        with col2: type_filter = st.multiselect("类型筛选", sorted(df_display["车辆类型"].unique()))
        with col3: conf_min = st.slider("最小置信度", 0.0, 1.0, 0.0)
        mask = df_display["置信度"] >= conf_min
        if addr_filter: mask &= df_display["离线属地"].isin(addr_filter)
        if type_filter: mask &= df_display["车辆类型"].isin(type_filter)
        st.dataframe(df_display[mask], use_container_width=True, height=300)
        st.subheader("📈 属地分布")
        chart_data = df_display["离线属地"].value_counts().reset_index()
        chart_data.columns = ["属地", "数量"]
        chart_data = chart_data[chart_data["属地"] != ""]
        fig = px.bar(
            chart_data,
            x="属地",
            y="数量",
            color="数量",
            color_continuous_scale="tealgrn",
            text="数量",
            height=400,
        )
        fig.update_traces(
            marker_cornerradius=dict(topLeft=6, topRight=6),
            textposition="outside",
            textfont=dict(size=12, color="#cccccc"),
            hovertemplate="<b>%{x}</b><br>数量: %{y}<extra></extra>",
        )
        fig.update_layout(
            xaxis=dict(title="", tickangle=-25, color="#aaaaaa"),
            yaxis=dict(title="", color="#aaaaaa", gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(t=10, b=40, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="closest",
            dragmode=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
