# -*- coding: utf-8 -*-
"""
AI车牌识别系统
技术栈：HyperLPR3 车牌检测识别 + DeepSeek 大模型智能分析
新增：图片压缩防卡顿、多车牌标注、离线属地、CSV导出、接口重试
作者：刘博华
日期：2026-06-06
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
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

# 图片压缩
def compress_image(pil_img, max_long=1920):
    w, h = pil_img.size
    if max(w, h) <= max_long:
        return pil_img
    if w > h:
        new_w = max_long
        new_h = int(h * (new_w / w))
    else:
        new_h = max_long
        new_w = int(w * (new_h / h))
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

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
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, plate_item["plate"], (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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

# ========== 页面初始化 ==========
st.set_page_config(page_title="AI车牌识别系统", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# 优化后的深色主题样式（无外层嵌套背景，卡片占满）
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    /* 移除 Markdown 容器的背景和边距，避免嵌套框 */
    [data-testid="stMarkdownContainer"] {
        background: transparent;
        padding: 0;
        margin: 0;
    }
    h1, h2, h3, h4 {
        color: #f0f4ff !important;
    }
    .stAlert {
        border-left: 4px solid #00e676 !important;
        background: rgba(0, 230, 118, 0.08) !important;
    }
    .stAlert.info {
        border-left: 4px solid #40c4ff !important;
        background: rgba(64, 196, 255, 0.08) !important;
    }
    .stAlert.warning {
        border-left: 4px solid #ffd740 !important;
        background: rgba(255, 215, 64, 0.08) !important;
    }
    .stAlert.error {
        border-left: 4px solid #ff5252 !important;
        background: rgba(255, 82, 82, 0.08) !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
    }
    /* 卡片样式：占满整列，无外框嵌套 */
    .custom-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
        word-break: break-word;
        width: 100%;
        margin: 0;
        box-sizing: border-box;
    }
    .custom-card-label {
        font-size: 14px;
        color: #aaa;
        margin-bottom: 4px;
    }
    .custom-card-value {
        font-size: 18px;
        font-weight: 500;
        line-height: 1.3;
    }
    .image-title {
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .image-title span {
        background: rgba(255, 255, 255, 0.12);
        padding: 6px 16px;
        border-radius: 24px;
        font-size: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Banner
st.markdown("""
<div style="text-align:center; padding:24px 0 16px 0; margin-bottom:20px;">
    <h1 style="color:#fff; font-size:34px; margin:0;">🚗 AI 车牌识别系统</h1>
    <p style="color:#90caf9; font-size:14px; margin:6px 0 0 0;">
        HyperLPR3 车牌检测 · DeepSeek 智能分析 · 多车牌标注 · 记录导出
    </p>
</div>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    api_key_input = st.text_input(
        "DeepSeek API Key (空则跳过AI)",
        type="password",
        help="在 deepseek.com 申请，用于大模型分析"
    )
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.6, 0.05)
    st.caption(f"已识别：{len(st.session_state.history)} 条记录")

    col_btn, _ = st.columns([1, 0.2])
    with col_btn:
        if st.button("🗑️ 清空识别历史", use_container_width=True):
            st.session_state.history = []
            st.session_state.uploader_key += 1
            st.rerun()

    if st.session_state.history:
        df_export = pd.DataFrame(st.session_state.history)
        csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode()
        st.download_button(
            "📥 导出全部记录CSV",
            csv_bytes,
            "车牌识别记录.csv",
            mime="text/csv",
            use_container_width=True
        )

# 上传组件（动态 key）
uploaded_files = st.file_uploader(
    "上传图片(支持批量jpg/png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_files:
    lpr = load_lpr()
    for idx, file in enumerate(uploaded_files):
        st.markdown(f"""
        <div class="image-title">
            <span>📷 图片 {idx+1} · {file.name}</span>
        </div>
        """, unsafe_allow_html=True)

        img_pil_ori = Image.open(file)
        img_pil = compress_image(img_pil_ori, max_long=1920)
        st.image(img_pil, caption="原图预览", use_container_width=True)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        draw_img = img_cv.copy()

        with st.spinner("识别车牌中..."):
            try:
                res_tuple = lpr(img_cv)
            except Exception as e:
                st.error(f"识别异常：{e}")
                continue
        res_list = parse_lpr_results(res_tuple)
        if not res_list:
            st.error("未检测到任何车牌")
            continue

        for one_plate in res_list:
            draw_img = draw_plate_box(draw_img, one_plate)
            fmt_plate, raw, conf, ptype, addr = smart_plate_parser(
                one_plate["plate"], one_plate["color"], conf_threshold, one_plate["confidence"]
            )
            if fmt_plate:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="custom-card">
                        <div class="custom-card-label">🚗 车牌号</div>
                        <div class="custom-card-value">{fmt_plate}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="custom-card">
                        <div class="custom-card-label">🎯 置信度</div>
                        <div class="custom-card-value">{conf:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="custom-card">
                        <div class="custom-card-label">🏷️ 类型</div>
                        <div class="custom-card-value" style="font-size:16px;">{ptype}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="custom-card">
                        <div class="custom-card-label">📍 属地</div>
                        <div class="custom-card-value">{addr}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if api_key_input.strip():
                    ctx = f"车牌号：{fmt_plate}，系统已识别为：{ptype}。"
                    with st.spinner("大模型分析..."):
                        ai_ret = deepseek_analyze(raw, ctx, api_key_input.strip())
                    st.info(f"AI分析：\n{ai_ret}")

                st.session_state.history.append({
                    "图片名称": file.name,
                    "号牌": fmt_plate,
                    "原始号牌": raw,
                    "置信度": round(conf, 2),
                    "车辆类型": ptype,
                    "离线属地": addr
                })
            else:
                st.error(f"❌ {ptype}｜{raw}")

        out_pil = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
        st.image(out_pil, caption="车牌框标注效果图", use_container_width=True)

# 历史表格
if st.session_state.history:
    st.markdown("---")
    st.subheader("📋 识别记录表")
    st.dataframe(st.session_state.history, use_container_width=True)