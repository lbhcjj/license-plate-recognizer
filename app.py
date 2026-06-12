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

for key in ["history", "results_cache", "api_key"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "api_key" else ""
for key in ["uploader_key", "files_processed", "last_file_count", "zip_processed", "zip_name"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key not in ("zip_processed", "zip_name") else False if key == "zip_processed" else None

# ====================================================================
# CSS 主题 + 扫描动画 + AI 动效
# ====================================================================
st.markdown("""
<style>
    /* ---------- CSS 变量（亮/暗） ---------- */
    :root, html[data-theme="dark"] {
        --bg-primary: #0b0e1a;
        --bg-card: rgba(255,255,255,0.04);
        --bg-card-hover: rgba(255,255,255,0.07);
        --border-card: rgba(255,255,255,0.06);
        --text-primary: #e0e0e0;
        --text-secondary: #c8cdd8;
        --text-muted: #8892b0;
        --ai-box-bg: rgba(64,196,255,0.06);
        --ai-box-border: rgba(64,196,255,0.15);
        --divider-color: rgba(255,255,255,0.08);
        --input-bg: rgba(255,255,255,0.06);
        --btn-bg: rgba(255,255,255,0.06);
        --sidebar-bg: rgba(11,14,26,0.98);
        --upload-bg: rgba(255,255,255,0.03);
        --upload-border: rgba(255,255,255,0.12);
    }
    html[data-theme="light"] {
        --bg-primary: #ffffff;
        --bg-card: rgba(0,0,0,0.02);
        --bg-card-hover: rgba(0,0,0,0.05);
        --border-card: rgba(0,0,0,0.08);
        --text-primary: #1a1a1a;
        --text-secondary: #4a4a4a;
        --text-muted: #6b7280;
        --ai-box-bg: rgba(64,196,255,0.08);
        --ai-box-border: rgba(64,196,255,0.2);
        --divider-color: rgba(0,0,0,0.1);
        --input-bg: rgba(0,0,0,0.04);
        --btn-bg: rgba(0,0,0,0.04);
        --sidebar-bg: #f8fafc;
        --upload-bg: rgba(0,0,0,0.02);
        --upload-border: rgba(0,0,0,0.15);
    }
    .stApp { background: var(--bg-primary) !important; color: var(--text-primary); }
    .app-header { text-align: center; padding: 20px 0; }
    .app-header h1 { font-size: 36px; font-weight: 800; color: #7ec8e3; margin: 0; }
    .plate-card { background: var(--bg-card); border-radius: 16px; padding: 20px; margin: 10px 0; }
    .plate-number { font-size: 32px; font-weight: 700; letter-spacing: 3px; font-family: 'Courier New', monospace; text-align: center; }
    .plate-number.blue { color: #4fc3f7; } .plate-number.green { color: #81c784; } .plate-number.yellow { color: #ffd54f; }
    .result-metrics { display: flex; justify-content: space-around; text-align: center; gap: 8px; margin-top: 16px; }
    .badge { padding: 3px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
    .badge-high { background: rgba(16,185,129,0.15); color: #059669; }
    .badge-medium { background: rgba(245,158,11,0.15); color: #d97706; }
    .badge-low { background: rgba(239,68,68,0.15); color: #dc2626; }
    .ai-box { background: var(--ai-box-bg); border-left: 4px solid #40c4ff; border-radius: 12px; padding: 16px 20px; margin: 12px 0; }

    /* ---------- 扫描动画 ---------- */
    @keyframes dot-bounce { 0%,80%,100% { transform: scale(0); } 40% { transform: scale(1); } }
    .scan-loader { display: flex; align-items: center; gap: 14px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 20px; margin: 10px 0; }
    .scan-dots { display: flex; gap: 5px; }
    .scan-dots span { width: 8px; height: 8px; border-radius: 50%; display: inline-block; animation: dot-bounce 1.4s ease-in-out infinite both; }
    .scan-dots span:nth-child(1) { background: #64ffda; animation-delay: -0.32s; }
    .scan-dots span:nth-child(2) { background: #40c4ff; animation-delay: -0.16s; }
    .scan-dots span:nth-child(3) { background: #b388ff; animation-delay: 0s; }
    .scan-text { color: #8892b0; font-size: 14px; font-weight: 500; }
    .scan-text.active { color: #64ffda; }

    /* ---------- AI 分析动效 ---------- */
    .ai-loader { display: flex; align-items: center; gap: 16px; background: rgba(64,196,255,0.05); border: 1px solid rgba(64,196,255,0.15); border-radius: 12px; padding: 16px 20px; margin: 10px 0; }
    .ai-pulse-ring { width: 32px; height: 32px; border-radius: 50%; border: 2px solid #40c4ff; animation: ai-pulse 1.5s ease-in-out infinite; flex-shrink: 0; }
    @keyframes ai-pulse { 0% { transform: scale(0.8); opacity: 0.6; } 50% { transform: scale(1.2); opacity: 1; } 100% { transform: scale(0.8); opacity: 0.6; } }
    .ai-loader-content { display: flex; flex-direction: column; gap: 10px; flex: 1; }
    .ai-loader-text { color: #40c4ff; font-size: 15px; font-weight: 600; letter-spacing: 1px; }
    .ai-loader-bar { height: 3px; background: rgba(64,196,255,0.15); border-radius: 2px; overflow: hidden; width: 100%; }
    .ai-loader-fill { height: 100%; width: 30%; background: linear-gradient(90deg, transparent, #40c4ff, transparent); border-radius: 2px; animation: ai-loading-bar 1.8s ease-in-out infinite; }
    @keyframes ai-loading-bar { 0% { transform: translateX(-100%); } 100% { transform: translateX(400%); } }
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
                            for h in st.session_state.history:
                                if h.get("file_md5") == info["file_md5"] and h["原始号牌"] == info["raw"]:
                                    h["号牌"] = f"{new_plate[:2]}·{new_plate[2:]}"
                            for cache in st.session_state.results_cache:
                                if cache["file_md5"] == info["file_md5"]:
                                    for p in cache["plates"]:
                                        if p["fmt_plate"] == info["fmt_plate"]:
                                            p["fmt_plate"] = f"{new_plate[:2]}·{new_plate[2:]}"
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
                    st.session_state.history.append({
                        "id": str(uuid.uuid4()),
                        "file_md5": info["file_md5"],
                        "图片名称": file.name,
                        "号牌": info["fmt_plate"],
                        "原始号牌": info["raw"],
                        "置信度": round(info["conf"], 2),
                        "车辆类型": info["ptype"],
                        "离线属地": info["addr"]
                    })

            # 缓存已标注图
            _, compressed = cv2.imencode('.jpg', cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            st.session_state.results_cache.append({
                "file_name": file.name,
                "file_md5": file_md5,
                "cached_img_bytes": compressed.tobytes(),
                "plates": [{"box": p["box"], "fmt_plate": p["fmt_plate"]} for p in plates_info],
            })

        status_text.text("✅ 全部处理完成！")
        st.session_state.files_processed = True

    elif st.session_state.results_cache:
        for result in st.session_state.results_cache:
            with st.expander(f"📷 {result['file_name']}", expanded=True):
                img_bgr = cv2.imdecode(np.frombuffer(result["cached_img_bytes"], np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)

# ---------- Tab2: ZIP批量 ----------
with tab2:
    st.markdown("### 📦 ZIP批量处理")
    uploaded_zip = st.file_uploader("上传包含图片的ZIP压缩包", type="zip")
    if uploaded_zip:
        if uploaded_zip.name != st.session_state.zip_name:
            st.session_state.zip_processed = False
            st.session_state.zip_name = uploaded_zip.name
        with zipfile.ZipFile(uploaded_zip, 'r') as zf:
            image_files = [f for f in zf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            st.info(f"发现 {len(image_files)} 张图片")
            if st.button("🚀 开始批量识别", type="primary") and not st.session_state.zip_processed:
                st.session_state.zip_processed = True
                lpr = load_lpr()
                progress = st.progress(0)
                for idx, img_name in enumerate(image_files):
                    progress.progress((idx + 1) / len(image_files))
                    with zf.open(img_name) as f:
                        file_bytes = f.read()
                        img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                        if img_bgr is not None:
                            img_bgr = resize_image_if_needed(img_bgr)
                            res_list = parse_lpr_results(lpr(img_bgr))
                            file_md5 = get_file_hash(file_bytes)
                            for one_plate in res_list:
                                fmt_plate, raw, conf, ptype, addr = smart_plate_parser(
                                    one_plate["plate"], one_plate["color"], conf_threshold, one_plate["confidence"]
                                )
                                if fmt_plate and not any(h.get("file_md5") == file_md5 and h["原始号牌"] == raw for h in st.session_state.history):
                                    st.session_state.history.append({
                                        "id": str(uuid.uuid4()),
                                        "file_md5": file_md5,
                                        "图片名称": img_name,
                                        "号牌": fmt_plate,
                                        "原始号牌": raw,
                                        "置信度": round(conf, 2),
                                        "车辆类型": ptype,
                                        "离线属地": addr
                                    })
                                    st.success(f"{img_name}: {fmt_plate}")
                st.success("✅ 批量处理完成！")
                st.session_state.zip_processed = False

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
        st.bar_chart(df_display["离线属地"].value_counts())
