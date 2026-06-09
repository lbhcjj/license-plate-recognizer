# -*- coding: utf-8 -*-
"""
AI车牌识别系统
技术栈：HyperLPR3 车牌检测识别 + DeepSeek 大模型智能分析
功能：图片压缩、多车牌标注、离线属地、CSV导出、接口重试
"""

import streamlit as st
import cv2
import numpy as np
from openai import OpenAI
import hyperlpr3 as lpr3
import re
import pandas as pd
import time
from functools import wraps

# ========== 模块常量 ==========
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

PROVINCE = r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]'
CITY = r'[A-Z]'
RULE_BLUE = re.compile(f'^{PROVINCE}{CITY}[A-Z0-9]{{5}}$')
RULE_GREEN = re.compile(f'^{PROVINCE}{CITY}[A-Z0-9]{{6}}$')
RULE_YELLOW = re.compile(f'^{PROVINCE}{CITY}[A-Z0-9]{{4,5}}$')


# API重试装饰器
def retry_api(max_retry=2, sleep_sec=1):
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
                        time.sleep(sleep_sec)
            return f"调用失败，重试{max_retry}次仍异常：{str(err)}"
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
            third_char = raw_plate[2] if len(raw_plate) >= 3 else ''
            if third_char in EV_PURE:
                plate_type = "新能源汽车（纯电动）"
            elif third_char in EV_HYBRID:
                plate_type = "新能源汽车（插电混动/增程）"
            elif third_char in EV_OTHER:
                plate_type = "新能源汽车（特殊号段）"
            else:
                plate_type = "新能源汽车"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}" if len(raw_plate) >= 2 else raw_plate
        else:
            plate_type = "新能源(格式异常)"
    elif plate_color == 'blue':
        if RULE_BLUE.match(raw_plate):
            plate_type = "燃油汽车"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}" if len(raw_plate) >= 2 else raw_plate
        else:
            plate_type = "燃油车(格式异常)"
    elif plate_color == 'yellow':
        if RULE_YELLOW.match(raw_plate):
            plate_type = "大型黄牌车辆"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}" if len(raw_plate) >= 2 else raw_plate
        else:
            plate_type = "黄牌车辆(格式异常)"
    else:
        if RULE_GREEN.match(raw_plate):
            plate_type = "新能源汽车（推断）"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}" if len(raw_plate) >= 2 else raw_plate
        elif RULE_BLUE.match(raw_plate):
            plate_type = "燃油汽车（推断）"
            formatted_plate = f"{raw_plate[:2]}·{raw_plate[2:]}" if len(raw_plate) >= 2 else raw_plate
        else:
            plate_type = "未知格式车牌"
    return formatted_plate, raw_plate, confidence, plate_type, addr


@st.cache_resource
def load_lpr():
    return lpr3.LicensePlateCatcher()


def parse_lpr_results(results_tuple):
    results_dict = []
    for item in results_tuple:
        try:
            if len(item) < 4:
                continue
            plate_str = str(item[0])
            conf = float(item[1])
            box = item[3] if len(item) > 3 else None
            color = 'unknown'
            if len(item) >= 5 and isinstance(item[4], str):
                color = item[4].lower()
            if box and isinstance(box, (list, tuple)) and len(box) == 4:
                results_dict.append({
                    "plate": plate_str,
                    "confidence": conf,
                    "color": color,
                    "box": box
                })
        except Exception:
            continue
    return results_dict


def draw_plate_box(img_bgr, plate_item):
    box = plate_item.get("box")
    if box is None or not isinstance(box, (list, tuple)) or len(box) != 4:
        return img_bgr
    try:
        x1, y1, x2, y2 = map(int, box)
        # 更醒目的标注框
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 120), 3)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 80), 1)
        # 标签背景
        label = plate_item["plate"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMLEX, 0.7, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 200, 80), -1)
        cv2.putText(img_bgr, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMLEX, 0.7, (0, 0, 0), 2)
    except Exception:
        pass
    return img_bgr


@retry_api(max_retry=2)
def deepseek_analyze(plate_number, context_info, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=15.0
    )
    system_msg = (
        "你是车牌分析专家。用户会提供系统已识别的车牌号和类型。"
        "请基于此分析归属地（根据车牌首汉字），并评价识别结果是否合理。"
        "新能源车牌第三位可为 D/A/B/C/E（纯电）或 F/G/H/J/K（插混），请认可此类格式。"
    )
    user_msg = f"{context_info}\n请给出：1.归属地 2.合理性评价。"
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}]
    )
    return resp.choices[0].message.content


# ====================================================================
# 页面初始化
# ====================================================================
st.set_page_config(page_title="AI车牌识别系统", layout="wide", page_icon="🚗")

if "history" not in st.session_state:
    st.session_state.history = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False
if "results_cache" not in st.session_state:
    st.session_state.results_cache = []

# ====================================================================
# 主题 CSS（根据暗色/亮色切换）
# ====================================================================
THEME = st.session_state.theme_mode

if THEME == "dark":
    BG_PRIMARY = "#0b0e1a"
    BG_CARD = "rgba(255, 255, 255, 0.04)"
    BG_CARD_HOVER = "rgba(255, 255, 255, 0.07)"
    BG_CARD_GLASS = "rgba(255, 255, 255, 0.05)"
    BORDER_CARD = "rgba(255, 255, 255, 0.06)"
    BORDER_CARD_HOVER = "rgba(255, 255, 255, 0.12)"
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#c8cdd8"
    TEXT_MUTED = "#8892b0"
    TEXT_HEADING = "#f0f4ff"
    SIDEBAR_BG = "rgba(11, 14, 26, 0.98)"
    TABLE_TH_BG = "rgba(255,255,255,0.06)"
    TABLE_TD_BORDER = "rgba(255,255,255,0.04)"
    TABLE_TH_BORDER = "rgba(255,255,255,0.08)"
    TABLE_HOVER = "rgba(255,255,255,0.03)"
    UPLOAD_BG = "rgba(255,255,255,0.03)"
    UPLOAD_BORDER = "rgba(255,255,255,0.12)"
    SCROLL_THUMB = "#2a2d40"
    SCROLL_THUMB_HOVER = "#3a3d55"
    LOADER_BG = "rgba(255, 255, 255, 0.04)"
    LOADER_BORDER = "rgba(255, 255, 255, 0.08)"
    AI_BOX_BG = "rgba(64, 196, 255, 0.06)"
    AI_BOX_BORDER = "rgba(64, 196, 255, 0.15)"
    DIVIDER_COLOR = "rgba(255,255,255,0.08)"
    INPUT_BG = "rgba(255,255,255,0.06)"
    BTN_BG = "rgba(255,255,255,0.06)"
    ALERT_BG = "rgba(255,255,255,0.04)"
    SIDEBAR_TEXT = "#c8cdd8"
    CAPTION_COLOR = "#8892b0"
    EXPANDER_BG = "rgba(255,255,255,0.04)"
else:
    BG_PRIMARY = "#f0f2f6"
    BG_CARD = "rgba(255, 255, 255, 0.7)"
    BG_CARD_HOVER = "rgba(255, 255, 255, 0.9)"
    BG_CARD_GLASS = "rgba(255, 255, 255, 0.6)"
    BORDER_CARD = "rgba(0, 0, 0, 0.08)"
    BORDER_CARD_HOVER = "rgba(0, 0, 0, 0.15)"
    TEXT_PRIMARY = "#1a1a2e"
    TEXT_SECONDARY = "#2d2d44"
    TEXT_MUTED = "#6b7280"
    TEXT_HEADING = "#1a1a2e"
    SIDEBAR_BG = "#f8f9fb"
    TABLE_TH_BG = "rgba(0,0,0,0.04)"
    TABLE_TD_BORDER = "rgba(0,0,0,0.06)"
    TABLE_TH_BORDER = "rgba(0,0,0,0.08)"
    TABLE_HOVER = "rgba(0,0,0,0.02)"
    UPLOAD_BG = "#ffffff"
    UPLOAD_BORDER = "rgba(0,0,0,0.15)"
    SCROLL_THUMB = "#c4c4c4"
    SCROLL_THUMB_HOVER = "#a0a0a0"
    LOADER_BG = "#ffffff"
    LOADER_BORDER = "rgba(0, 0, 0, 0.1)"
    AI_BOX_BG = "rgba(64, 196, 255, 0.08)"
    AI_BOX_BORDER = "rgba(64, 196, 255, 0.2)"
    DIVIDER_COLOR = "rgba(0,0,0,0.08)"
    INPUT_BG = "#ffffff"
    BTN_BG = "#ffffff"
    ALERT_BG = "rgba(255,255,255,0.85)"
    SIDEBAR_TEXT = "#2d2d44"
    CAPTION_COLOR = "#6b7280"
    EXPANDER_BG = "rgba(255,255,255,0.5)"

st.markdown(f"""
<style>
    /* ---------- 全局 ---------- */
    .stApp {{
        background: {BG_PRIMARY};
        color: {TEXT_PRIMARY};
    }}
    .block-container {{
        padding: 1.5rem 2rem !important;
        max-width: 1400px;
        margin: 0 auto;
    }}
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG_PRIMARY}; }}
    ::-webkit-scrollbar-thumb {{ background: {SCROLL_THUMB}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {SCROLL_THUMB_HOVER}; }}

    /* ---------- 标题文字 ---------- */
    h1, h2, h3, h4 {{
        color: {TEXT_HEADING} !important;
        letter-spacing: 0.3px;
    }}
    p, li, .stMarkdown {{
        color: {TEXT_SECONDARY};
    }}

    /* ---------- 标题横幅 ---------- */
    .app-header {{
        text-align: center;
        padding: 28px 0 12px 0;
        margin-bottom: 20px;
        position: relative;
    }}
    .app-header h1 {{
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #64ffda, #40c4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }}
    .app-header .subtitle {{
        color: {TEXT_MUTED};
        font-size: 14px;
        margin: 6px 0 0 0;
        letter-spacing: 1px;
    }}

    /* ---------- 玻璃卡片基类 ---------- */
    .glass-card {{
        background: {BG_CARD_GLASS};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {BORDER_CARD};
        border-radius: 14px;
        padding: 16px 14px;
        transition: all 0.25s ease;
    }}
    .glass-card:hover {{
        background: {BG_CARD_HOVER};
        border-color: {BORDER_CARD_HOVER};
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }}

    /* ---------- 结果卡片 ---------- */
    .plate-card {{
        background: {BG_CARD};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {BORDER_CARD};
        border-radius: 16px;
        padding: 18px 16px;
        margin: 8px 0;
        transition: all 0.25s ease;
    }}
    .plate-card:hover {{
        background: {BG_CARD_HOVER};
        border-color: {BORDER_CARD_HOVER};
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
    .plate-number.default {{ color: {TEXT_PRIMARY}; }}

    /* 指标标签-值对 */
    .metric-item {{
        text-align: center;
        padding: 4px 0;
    }}
    .metric-item .label {{
        font-size: 12px;
        color: {TEXT_MUTED};
        margin-bottom: 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .metric-item .value {{
        font-size: 16px;
        font-weight: 600;
        color: {TEXT_PRIMARY};
    }}
    .metric-item .value.highlight {{
        color: #64ffda;
    }}

    /* ---------- 置信度徽章 ---------- */
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

    /* ---------- 图片标题 ---------- */
    .image-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 4px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid {BORDER_CARD};
    }}
    .image-header .icon {{ font-size: 20px; }}
    .image-header .name {{
        font-size: 16px;
        font-weight: 600;
        color: {TEXT_PRIMARY};
    }}

    /* ---------- 加载动画 ---------- */
    @keyframes dot-bounce {{
        0%, 80%, 100% {{ transform: scale(0); }}
        40% {{ transform: scale(1); }}
    }}
    .scan-loader {{
        display: flex;
        align-items: center;
        gap: 14px;
        background: {LOADER_BG};
        border: 1px solid {LOADER_BORDER};
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
        color: {TEXT_MUTED};
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }}
    .scan-text.active {{ color: #64ffda; }}
    .scan-text.blue   {{ color: #40c4ff; }}

    /* ---------- 侧边栏 ---------- */
    [data-testid="stSidebar"] {{
        background: {SIDEBAR_BG};
        border-right: 1px solid {BORDER_CARD};
    }}
    [data-testid="stSidebar"] .stMarkdown {{
        color: {TEXT_SECONDARY};
    }}

    /* ---------- AI 分析框 ---------- */
    .ai-box {{
        background: {AI_BOX_BG};
        border: 1px solid {AI_BOX_BORDER};
        border-left: 3px solid #40c4ff;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: {TEXT_SECONDARY};
        font-size: 14px;
        line-height: 1.6;
    }}
    .ai-box strong {{ color: #40c4ff; }}

    /* ---------- 消息框 ---------- */
    .stAlert {{ border-radius: 10px !important; border: none !important; }}
    [data-testid="stAlert"] {{ padding: 12px 16px !important; }}
    div[data-baseweb="alert"] {{ border-radius: 10px !important; }}

    /* ---------- 按钮 ---------- */
    .stButton button {{
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: 1px solid {BORDER_CARD} !important;
    }}
    .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }}

    /* ---------- 上传组件 ---------- */
    [data-testid="stFileUploader"] {{
        background: {UPLOAD_BG};
        border: 1px dashed {UPLOAD_BORDER};
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
        background: linear-gradient(90deg, transparent, {DIVIDER_COLOR}, transparent);
        margin: 28px 0;
    }}

    /* ---------- 响应式 ---------- */
    @media (max-width: 768px) {{
        .block-container {{ padding: 1rem 1rem !important; }}
        .app-header h1 {{ font-size: 26px; }}
        .app-header {{ padding-top: 36px !important; }}
        .plate-number {{ font-size: 22px; letter-spacing: 2px; }}
        .plate-card {{ padding: 14px 10px; }}
    }}
    @media (max-width: 480px) {{
        .block-container {{ padding: 0.6rem 0.6rem !important; }}
        .plate-number {{ font-size: 18px; }}
        .metric-item .value {{ font-size: 14px; }}
    }}

    /* ---------- 主题切换徽章 ---------- */
    .theme-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        padding: 2px 10px;
        border-radius: 20px;
        background: {BG_CARD_GLASS};
        border: 1px solid {BORDER_CARD};
        color: {TEXT_MUTED};
    }}

    /* ================================================================
       以下覆盖 Streamlit 原生组件，确保亮色模式彻底无暗色残留
       使用双倍 class (e.g. .stTextInput.stTextInput) 提高特异性
       ================================================================ */

    /* ---------- 顶部黑杠（仅桌面端隐藏，手机端保留 ☰ 汉堡菜单） ---------- */
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
            background: rgba(11, 14, 26, 0.85) !important;
            backdrop-filter: blur(10px) !important;
            border: none !important;
            height: 40px !important;
            min-height: 40px !important;
        }}
        header[data-testid="stHeader"] button {{
            color: #e0e0e0 !important;
        }}
        header[data-testid="stHeader"] svg {{
            fill: #e0e0e0 !important;
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
        color: {SIDEBAR_TEXT} !important;
    }}
    section[data-testid="stSidebar"] .stCaption {{
        color: {CAPTION_COLOR} !important;
    }}

    /* ---------- 文本输入框：全覆盖 ---------- */
    .stTextInput.stTextInput {{
        background: transparent !important;
    }}
    .stTextInput.stTextInput > div {{
        background-color: {INPUT_BG} !important;
        border-radius: 10px !important;
        border: 1px solid {BORDER_CARD} !important;
    }}
    .stTextInput.stTextInput > div > div {{
        background-color: transparent !important;
    }}
    /* 输入文字颜色（双保险：color + -webkit-text-fill-color） */
    .stTextInput.stTextInput input {{
        background-color: transparent !important;
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
        caret-color: {TEXT_PRIMARY} !important;
        border: none !important;
        box-shadow: none !important;
    }}
    .stTextInput.stTextInput input::placeholder {{
        color: {TEXT_MUTED} !important;
        -webkit-text-fill-color: {TEXT_MUTED} !important;
        opacity: 0.7 !important;
    }}
    .stTextInput.stTextInput label {{
        color: {TEXT_SECONDARY} !important;
    }}
    /* password 眼睛图标（所有 SVG 层次全覆盖） */
    .stTextInput [data-testid="stTextInputVisibilityToggle"] {{
        color: {TEXT_SECONDARY} !important;
        opacity: 0.65 !important;
    }}
    .stTextInput [data-testid="stTextInputVisibilityToggle"] *,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg *,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg path,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg circle,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg line,
    .stTextInput [data-testid="stTextInputVisibilityToggle"] svg polygon {{
        fill: {TEXT_SECONDARY} !important;
        color: {TEXT_SECONDARY} !important;
        stroke: {TEXT_SECONDARY} !important;
    }}
    .stTextInput [data-testid="stTextInputVisibilityToggle"]:hover {{
        opacity: 1 !important;
    }}

    /* ---------- 滑块 ---------- */
    .stSlider.stSlider label {{
        color: {TEXT_SECONDARY} !important;
    }}
    .stSlider.stSlider div[data-testid="stTickBar"] {{
        color: {TEXT_MUTED} !important;
    }}

    /* ---------- 普通按钮 + 下载按钮 ---------- */
    .stButton.stButton button,
    .stDownloadButton.stDownloadButton button {{
        background: {BTN_BG} !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {BORDER_CARD} !important;
        border-radius: 10px !important;
    }}
    .stButton.stButton button:hover,
    .stDownloadButton.stDownloadButton button:hover {{
        background: {BG_CARD_HOVER} !important;
        border-color: {BORDER_CARD_HOVER} !important;
    }}

    /* ---------- Alert / info / error / success ---------- */
    .stAlert.stAlert {{
        background: {ALERT_BG} !important;
        backdrop-filter: blur(8px);
        color: {TEXT_SECONDARY} !important;
    }}

    /* ---------- 图片 caption ---------- */
    .stImage.stImage figcaption {{
        color: {CAPTION_COLOR} !important;
        font-size: 13px !important;
    }}

    /* ---------- 展开器 ---------- */
    .streamlit-expanderHeader {{
        background: {EXPANDER_BG} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    .streamlit-expanderHeader:hover {{
        background: {BG_CARD_HOVER} !important;
    }}
    .streamlit-expanderContent {{
        background: transparent !important;
    }}

    /* ---------- 选择框等 ---------- */
    .stSelectbox.stSelectbox div[data-baseweb="select"] > div {{
        background-color: {INPUT_BG} !important;
        border: 1px solid {BORDER_CARD} !important;
    }}

    /* ---------- 分隔线 ---------- */
    hr {{
        border-color: {DIVIDER_COLOR} !important;
    }}

    /* ---------- 文件上传器：背景+文字全覆盖 ---------- */
    [data-testid="stFileUploader"] {{
        background: {UPLOAD_BG} !important;
    }}
    [data-testid="stFileUploader"] section {{
        color: {TEXT_SECONDARY} !important;
    }}
    [data-testid="stFileUploader"] button {{
        background: {BTN_BG} !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {BORDER_CARD} !important;
    }}
    /* 上传器内所有 div 背景覆盖 */
    [data-testid="stFileUploader"] div {{
        background: transparent !important;
    }}
    /* 上传器拖拽区域专门处理 */
    [data-testid="stFileUploader"] [data-testid="stFileUploadDragzone"] {{
        background: {UPLOAD_BG} !important;
        border: 1px dashed {UPLOAD_BORDER} !important;
    }}

    /* ---------- spinner ---------- */
    .stSpinner.stSpinner > div {{
        border-color: {TEXT_MUTED} !important;
    }}

    /* ---------- tooltip ---------- */
    [data-baseweb="tooltip"] {{
        background: {BG_CARD_GLASS} !important;
        backdrop-filter: blur(12px);
        color: {TEXT_PRIMARY} !important;
    }}

    /* ---------- 表格：数据区容器 + 所有子层级 ---------- */
    /* 先把容器显式设置为主体背景色而非 transparent */
    [data-testid="stDataFrame"] {{
        background: {BG_PRIMARY} !important;
    }}
    [data-testid="stDataFrame"] > div:first-child {{
        background: {BG_PRIMARY} !important;
    }}
    [data-testid="stDataFrame"] table {{
        background: {BG_PRIMARY} !important;
    }}
    [data-testid="stDataFrame"] thead {{
        background: {BG_PRIMARY} !important;
    }}
    [data-testid="stDataFrame"] th {{
        background: {TABLE_TH_BG} !important;
        color: {TEXT_MUTED} !important;
        border-bottom: 1px solid {TABLE_TH_BORDER} !important;
    }}
    [data-testid="stDataFrame"] td {{
        background: {BG_PRIMARY} !important;
        color: {TEXT_SECONDARY} !important;
        border-bottom: 1px solid {TABLE_TD_BORDER} !important;
    }}
    [data-testid="stDataFrame"] tr {{
        background: {BG_PRIMARY} !important;
    }}
    [data-testid="stDataFrame"] tbody {{
        background: {BG_PRIMARY} !important;
    }}
    [data-testid="stDataFrame"] tr:hover td {{
        background: {TABLE_HOVER} !important;
    }}
    /* 虚拟滚动容器 */
    [data-testid="stDataFrame"] [data-testid="StyledVirtuosoItem"] {{
        background: {BG_PRIMARY} !important;
    }}
    [data-testid="stDataFrame"] [data-testid="StyledVirtuosoItem"] td {{
        background: {BG_PRIMARY} !important;
    }}
    [data-testid="stDataFrame"] div[data-testid="StyledVirtuoso"] {{
        background: {BG_PRIMARY} !important;
    }}
    [data-testid="stDataFrame"] [data-testid="StyledVirtuoso"] > div {{
        background: {BG_PRIMARY} !important;
    }}
    /* 表格内任何 emotion 容器 */
    [data-testid="stDataFrame"] [class] {{
        background: transparent !important;
    }}
    /* 表格区域所有 div 背景清除（确保无暗色残留） */
    [data-testid="stDataFrame"] div {{
        background: transparent !important;
    }}
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 顶部标题
# ====================================================================
st.markdown("""
<div class="app-header">
    <h1>🚗 AI 车牌识别系统</h1>
    <p class="subtitle">HyperLPR3 车牌检测 · DeepSeek 智能分析 · 多车牌标注 · 记录导出</p>
</div>
""", unsafe_allow_html=True)

# ====================================================================
# 侧边栏
# ====================================================================
with st.sidebar:
    st.markdown("### ⚙️ 设置")

    api_key_input = st.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="sk-... (空则跳过AI分析)",
        help="在 deepseek.com 申请，用于大模型分析"
    )

    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.6, 0.05)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 16px 0;'>", unsafe_allow_html=True)

    # ── 暗色/亮色切换 ──
    theme_icon = "🌙" if THEME == "dark" else "☀️"
    theme_label = "暗色主题" if THEME == "dark" else "亮色主题"
    if st.button(f"{theme_icon} 切换为{'亮色' if THEME == 'dark' else '暗色'}", use_container_width=True):
        st.session_state.theme_mode = "light" if THEME == "dark" else "dark"
        st.rerun()
    st.markdown(f"<div style='text-align:center;'><span class='theme-badge'>{theme_icon} 当前：{theme_label}</span></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 16px 0;'>", unsafe_allow_html=True)

    col_info, col_btn = st.columns([1, 1])
    with col_info:
        st.caption(f"📋 已识别 **{len(st.session_state.history)}** 条记录")
    with col_btn:
        if st.button("🗑️ 清空", use_container_width=True):
            st.session_state.history = []
            st.session_state.uploader_key += 1
            st.session_state.files_processed = False
            st.session_state.results_cache = []
            st.rerun()

    if st.session_state.history:
        df_export = pd.DataFrame(st.session_state.history)
        csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode()
        st.download_button(
            "📥 导出 CSV",
            csv_bytes,
            "车牌识别记录.csv",
            mime="text/csv",
            use_container_width=True
        )

# ====================================================================
# 主界面
# ====================================================================

# 上传组件（动态 key）
uploaded_files = st.file_uploader(
    "选择图片（支持批量 JPG / PNG）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_files and not st.session_state.files_processed:
    # ── 首次处理：识别 + 缓存 + 前端显示 ──
    st.session_state.results_cache = []
    lpr = load_lpr()

    for idx, file in enumerate(uploaded_files):
        file_bytes = file.read()
        img_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None:
            from PIL import Image
            from io import BytesIO
            try:
                img_pil = Image.open(BytesIO(file_bytes))
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception:
                st.error(f"⚠️ [{file.name}] 无法解析图片，请检查格式")
                continue

        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w = img_cv.shape[:2]
        if max(w, h) > 1920:
            scale = 1920 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 展示原图
        st.image(img_cv, caption="原图预览", use_container_width=True)
        draw_img = img_cv.copy()

        # 加载动画
        loader = st.empty()
        loader.markdown("""
        <div class="scan-loader">
            <div class="scan-dots"><span></span><span></span><span></span></div>
            <span class="scan-text active">🔍 扫描车牌中</span>
        </div>
        """, unsafe_allow_html=True)

        try:
            res_tuple = lpr(img_cv)
        except Exception as e:
            loader.empty()
            st.error(f"⚠️ [{file.name}] 识别异常：{e}")
            continue

        res_list = parse_lpr_results(res_tuple)
        if not res_list:
            loader.empty()
            st.error(f"⚠️ [{file.name}] 未检测到任何车牌")
            continue

        loader.empty()

        plates_cache = []
        for one_plate in res_list:
            draw_img = draw_plate_box(draw_img, one_plate)
            fmt_plate, raw, conf, ptype, addr = smart_plate_parser(
                one_plate["plate"], one_plate["color"], conf_threshold, one_plate["confidence"]
            )
            if not fmt_plate:
                st.error(f"❌ {ptype}｜{raw}")
                continue

            color_cls = one_plate.get("color", "unknown").lower()
            if color_cls not in ("blue", "green", "yellow"):
                color_cls = "default"

            if conf >= 0.85:
                badge_cls, badge_txt = "badge-high", "高"
            elif conf >= 0.60:
                badge_cls, badge_txt = "badge-medium", "中"
            else:
                badge_cls, badge_txt = "badge-low", "低"

            # 车牌卡片
            st.markdown(f"""
            <div class="plate-card">
                <div style="display:grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap:12px; align-items:center;">
                    <div class="metric-item" style="grid-column: 1 / 2;">
                        <div class="label">车牌号</div>
                        <div class="plate-number {color_cls}">{fmt_plate}</div>
                    </div>
                    <div class="metric-item">
                        <div class="label">置信度</div>
                        <div><span class="badge {badge_cls}">{conf:.0%} · {badge_txt}</span></div>
                    </div>
                    <div class="metric-item">
                        <div class="label">类型</div>
                        <div class="value" style="font-size:14px;">{ptype}</div>
                    </div>
                    <div class="metric-item">
                        <div class="label">属地</div>
                        <div class="value highlight">{addr}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # AI 分析
            if api_key_input.strip():
                loader2 = st.empty()
                loader2.markdown("""
                <div class="scan-loader">
                    <div class="scan-dots"><span></span><span></span><span></span></div>
                    <span class="scan-text blue">🧠 大模型分析中</span>
                </div>
                """, unsafe_allow_html=True)
                ctx = f"车牌号：{fmt_plate}，系统已识别为：{ptype}。"
                ai_ret = deepseek_analyze(raw, ctx, api_key_input.strip())
                loader2.empty()
                st.markdown(f"""
                <div class="ai-box">
                    <strong>🧠 AI 分析</strong><br>
                    {ai_ret}
                </div>
                """, unsafe_allow_html=True)

            # 写历史（去重）
            already_exists = any(
                h["图片名称"] == file.name and h["号牌"] == fmt_plate
                for h in st.session_state.history
            )
            if not already_exists:
                st.session_state.history.append({
                    "图片名称": file.name,
                    "号牌": fmt_plate,
                    "原始号牌": raw,
                    "置信度": round(conf, 2),
                    "车辆类型": ptype,
                    "离线属地": addr
                })

            # 写入缓存（供主题切换后重显示）
            plates_cache.append({
                "box": one_plate["box"],
                "plate": one_plate["plate"],
                "fmt_plate": fmt_plate,
                "raw": raw,
                "confidence": conf,
                "ptype": ptype,
                "addr": addr,
                "color": one_plate.get("color", "unknown"),
            })

        # 标注图
        st.image(draw_img, caption="🎯 车牌框标注效果图", use_container_width=True)

        # 整图缓存
        st.session_state.results_cache.append({
            "file_name": file.name,
            "file_bytes": file_bytes,
            "plates": plates_cache,
        })

    st.session_state.files_processed = True

# ── 主题切换等 rerun 后：从缓存重显示（不重复识别、不写历史）──
elif st.session_state.results_cache:
    for idx, result in enumerate(st.session_state.results_cache):
        with st.expander(f"📷 **{idx+1}. {result['file_name']}**", expanded=True):
            # 解码原图
            img_cv = cv2.imdecode(np.frombuffer(result["file_bytes"], np.uint8), cv2.IMREAD_COLOR)
            if img_cv is None:
                from PIL import Image
                from io import BytesIO
                img_pil = Image.open(BytesIO(result["file_bytes"]))
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            h, w = img_cv.shape[:2]
            if max(w, h) > 1920:
                scale = 1920 / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            st.image(img_cv, caption="原图预览", use_container_width=True)
            draw_img = img_cv.copy()

            for plate_data in result["plates"]:
                draw_img = draw_plate_box(draw_img, plate_data)
                color_cls = plate_data["color"].lower()
                if color_cls not in ("blue", "green", "yellow"):
                    color_cls = "default"

                conf = plate_data["confidence"]
                if conf >= 0.85:
                    badge_cls, badge_txt = "badge-high", "高"
                elif conf >= 0.60:
                    badge_cls, badge_txt = "badge-medium", "中"
                else:
                    badge_cls, badge_txt = "badge-low", "低"

                st.markdown(f"""
                <div class="plate-card">
                    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap:12px; align-items:center;">
                        <div class="metric-item" style="grid-column: 1 / 2;">
                            <div class="label">车牌号</div>
                            <div class="plate-number {color_cls}">{plate_data['fmt_plate']}</div>
                        </div>
                        <div class="metric-item">
                            <div class="label">置信度</div>
                            <div><span class="badge {badge_cls}">{conf:.0%} · {badge_txt}</span></div>
                        </div>
                        <div class="metric-item">
                            <div class="label">类型</div>
                            <div class="value" style="font-size:14px;">{plate_data['ptype']}</div>
                        </div>
                        <div class="metric-item">
                            <div class="label">属地</div>
                            <div class="value highlight">{plate_data['addr']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.image(draw_img, caption="🎯 车牌框标注效果图", use_container_width=True)

# ====================================================================
# 历史记录表
# ====================================================================
if st.session_state.history:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_title, col_export = st.columns([3, 1])
    with col_title:
        st.markdown("### 📋 识别记录表")
    with col_export:
        df_export = pd.DataFrame(st.session_state.history)
        csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode()
        st.download_button(
            "📥 导出 CSV",
            csv_bytes,
            "车牌识别记录.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.dataframe(st.session_state.history, use_container_width=True, height=280)
