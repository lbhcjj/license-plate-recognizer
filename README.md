# 🚗 AI车牌识别系统（HyperLPR3 + DeepSeek大模型）

🌐 **在线体验**：[https://license-plate-recognizer-xu4n4rczfazyg2nf9emhzy.streamlit.app/](https://license-plate-recognizer-xu4n4rczfazyg2nf9emhzy.streamlit.app/) （手机/电脑直接使用，无需安装）

## 项目简介
本项目基于深度学习的 HyperLPR3 车牌识别框架和 DeepSeek 大语言模型，实现从上传图片中自动识别车牌号码、车辆类型（燃油/新能源/黄牌等），并调用大模型分析归属地与合理性。系统**支持新能源绿牌扩展号段**（第三位可为 D/A/B/C/E 纯电或 F/G/H/J/K 混动），避免旧知识误判。

## 主要功能
- 📸 上传单张/多张图片，自动识别所有车牌
- 🖼️ 在一张图片上同时标注多个车牌位置（绿色矩形框）
- 🚘 区分燃油汽车、新能源汽车（纯电/混动）、大型黄牌车辆
- 🟢 新能源绿牌支持 D/A/B/C/E（纯电）和 F/G/H/J/K（非纯电）
- 🤖 调用 DeepSeek 大模型分析车牌归属地、合理性及动力类型
- 📊 识别历史自动保存，支持一键导出 CSV 表格
- 📱 深色主题，自适应手机屏幕，触摸操作友好
- ⏳ 扫描车牌和大模型分析时显示动态光条动画
- 🖥️ 简洁的 Web 界面（Streamlit）

## 环境要求
- Python 3.8～3.11
- Windows / macOS / Linux

## 快速开始

### 1. 安装 Python
如果尚未安装，请从 [python.org](https://python.org) 下载并安装，**安装时务必勾选 “Add Python to PATH”**。

### 2. 解压代码包
将本压缩包解压到任意文件夹。

### 3. 运行程序
- **Windows**：双击 `run.bat`
- **Mac / Linux**：在终端中执行 `streamlit run app.py`

首次运行时，系统会自动安装依赖库（需要网络），并下载 HyperLPR3 模型文件（约10MB），请耐心等待。

### 4. 使用大模型功能（可选）
程序支持 DeepSeek 大模型智能分析。如需使用：
- 访问 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册账号，创建 API Key。
- 在程序界面的密码框中粘贴 API Key，即可获得分析结果。

若不提供 Key，程序仍可正常识别车牌，只是不会显示大模型分析结果。

## 文件说明
- `app.py`：主程序（含车牌解析、大模型调用、多车牌标注）
- `run.bat`：一键启动脚本（Windows）
- `requirements.txt`：依赖清单
- `README.md`：说明文档
- `data/`：（可选）测试图片样例

## 常见问题

### Q1: 双击 run.bat 提示 `'python' 不是内部或外部命令`
**解决**：Python 未正确安装或未添加到 PATH，请重新安装 Python 并勾选 “Add Python to PATH”。也可以使用 `py -3 -m streamlit run app.py` 启动。

### Q2: 提示 `ModuleNotFoundError: No module named 'streamlit'`
**解决**：依赖安装失败，请手动打开命令行，进入本目录，执行 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`。

### Q3: 识别不准确
**解决**：请使用清晰、正视角、光线充足的车牌图片。HyperLPR3 对模糊或倾斜车牌识别率较低。建议测试蓝牌或绿牌正面照片。

### Q4: 新能源车牌被大模型判为“不合理”
**解决**：本程序已优化提示词，支持新能源绿牌扩展字母（A/B/C/E/G/H/J/K 等）。如果仍出现误判，请确认你的 API Key 是否正确，或检查图片中的车牌第三位是否为合法的字母（非数字）。若实际车牌是 D/F 但被误识别为 B，可能是 HyperLPR3 识别误差，可换一张更清晰的图片重试。

### Q5: 大模型返回错误或超时
**解决**：检查 API Key 是否正确，网络是否通畅。可尝试重新生成 Key，或检查账户剩余额度。

## 技术特点
- 基于 HyperLPR3 轻量级模型（`model='M'`），CPU 即可实时识别。
- 正则表达式严格校验车牌格式，区分蓝/绿/黄牌。
- 新能源车牌纯电/混动字母集合扩展，兼容各地实际投放号段（如粤AB、粤AD等）。
- 大模型提示词注入最新规则，明确说明允许的第三位字母范围，避免因知识陈旧而误判。
- 置信度过滤（阈值可调），识别无效时自动提示，减少错误输出。
- 双重图像解码（OpenCV + PIL 兜底），提高兼容性。

## 作者信息
- 专  业：建筑环境与能源应用工程
- 班  级：建环2501
- 姓  名：刘博华
- 学  号：202506120132

## 致谢
- [HyperLPR3](https://github.com/szad670401/HyperLPR) 开源车牌识别框架
- [DeepSeek](https://deepseek.com) 提供大模型 API
- [Streamlit](https://streamlit.io) 快速构建 Web 界面

## 许可证
本项目仅供学习交流，不作商业用途。