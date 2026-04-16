# -*- coding: utf-8 -*-
import streamlit as st
import json
import os
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
import re
import io
import sys
import traceback
import warnings

warnings.filterwarnings("ignore", message=".*use_container_width.*")

# ==========================================
# 0. 核心模块与工具类
# ==========================================
try:
    # 假设这些模块存在于您的项目路径中
    import llm_select_topk
    import evaluate_f1
    import onlyselect
except ImportError as e:
    st.error(f"核心后端模块导入失败，请确保 llm_select_topk.py, evaluate_f1.py, onlyselect.py 文件存在: {e}")
    st.stop()


class TeeStdout(object):
    def __init__(self, buffer):
        self.buffer = buffer
        self.terminal = sys.stdout

    def write(self, message):
        if "use_container_width" in message or "width='stretch'" in message:
            return
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()
        self.buffer.flush()


def apply_ngram_patch(n_value):
    # 此函数保持不变
    def patched_calculate_similarity(text1, text2):
        if not text1 and not text2: return 1.0
        if not text1 or not text2: return 0.0
        ngrams1 = llm_select_topk.get_char_ngrams(text1, n=n_value)
        ngrams2 = llm_select_topk.get_char_ngrams(text2, n=n_value)
        if not ngrams1 and not ngrams2: return 1.0
        if not ngrams1 or not ngrams2: return 0.0
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        return len(intersection) / len(union)

    llm_select_topk.calculate_similarity = patched_calculate_similarity
    # 假设 onlyselect 也有这个函数
    if hasattr(onlyselect, 'calculate_similarity'):
        onlyselect.calculate_similarity = patched_calculate_similarity


# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="大模型驱动的情感细粒度分析处理平台", layout="wide")

UPLOAD_DIR = "data_repository"
CHECKPOINT_DIR = "checkpoints"
HISTORY_FILE = os.path.join(UPLOAD_DIR, "eval_history.csv")

for d in [UPLOAD_DIR, CHECKPOINT_DIR]:
    if not os.path.exists(d): os.makedirs(d)


# ==========================================
# 2. 核心计算与统计函数
# ==========================================
def parse_train_txt(file_content):
    examples = []
    lines = file_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if '\t' in line:
            parts = line.split('\t')
            input_text = parts[0].strip()

            quads = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue

                if '\t' in next_line or not next_line.startswith('['):
                    break

                if next_line.startswith('[[') and next_line.endswith(']]'):
                    content = next_line[1:-1]
                    segments = content.split(';')
                    if len(segments) == 5:
                        parsed_quad = []
                        for k, seg in enumerate(segments):
                            seg = seg.strip()
                            if seg.startswith('[') and seg.endswith(']'):
                                inner = seg[1:-1].strip()
                                if k < 4:
                                    if inner:
                                        if '&' in inner:
                                            tokens = re.findall(r'&(\S+)', inner)
                                            parsed_quad.append("".join(tokens))
                                        else:
                                            parsed_quad.append(inner.replace(" ", ""))
                                    else:
                                        parsed_quad.append("")
                                else:
                                    parsed_quad.append(inner)
                            else:
                                parsed_quad.append("")

                        if any(parsed_quad):
                            quads.append(parsed_quad)
                j += 1

            examples.append({'input': input_text, 'target': quads})
            i = j
        else:
            i += 1
    return examples


def calc_stats(examples):
    total = len(examples)
    s_cnt = {"Pos (1)": 0, "Neg (-1)": 0, "Neu (0)": 0, "None (2)": 0}
    a_cnt = {}
    for ex in examples:
        targets = ex.get('target', [])
        for t in targets:
            if len(t) == 5:
                s = str(t[4]).strip()
                if s == '1':
                    s_cnt["Pos (1)"] += 1
                elif s == '-1':
                    s_cnt["Neg (-1)"] += 1
                elif s == '0':
                    s_cnt["Neu (0)"] += 1
                else:
                    s_cnt["None (2)"] += 1

                asp = t[2].strip()
                if asp: a_cnt[asp] = a_cnt.get(asp, 0) + 1
    return {
        "total": total,
        "s_cnt": s_cnt,
        "top_a": dict(sorted(a_cnt.items(), key=lambda x: x[1], reverse=True)[:10])
    }


def estimate_deepseek_cost(json_data):
    if not json_data or 'examples' not in json_data:
        return 0.0, 0, 0

    in_chars = 0
    out_chars = 0
    for ex in json_data['examples']:
        in_chars += 300
        in_chars += len(ex.get('input', ''))
        cands = ex.get('model_output_top3', []) or ex.get('model_output_top7', []) or ex.get('model_output_top9', [])
        for c in cands:
            in_chars += len(c)
        out_chars += 50

    in_tokens = int(in_chars * 1.3)
    out_tokens = int(out_chars * 1.3)

    cost = (in_tokens / 1000000.0) * 1.0 + (out_tokens / 1000000.0) * 2.0
    return cost, in_tokens, out_tokens


def render_backend_logs_visuals(log_txt):
    st.divider()
    sim_matches = re.findall(r'^\s*(\d+\.\d+):\s+(.*)$', log_txt, re.MULTILINE)

    # --- 图表渲染部分 ---
    if sim_matches:
        df_viz = pd.DataFrame(sim_matches, columns=['Score', 'FullText'])
        df_viz['Score'] = df_viz['Score'].astype(float)


        df_viz['Display'] = df_viz['FullText'].apply(lambda x: (x[:35] + '...') if len(x) > 35 else x)


        df_viz['ScoreText'] = (df_viz['Score'] * 100).map('{:.1f}%'.format)

        st.markdown("**相似样本匹配度 (N-gram)**")

        # 使用 Plotly Express 创建更精美的图表
        fig = px.bar(
            df_viz,
            x='Score',
            y='Display',
            orientation='h',
            text='ScoreText',
            labels={'Score': '相似度得分', 'Display': '相似样本'},

            hover_data={'Score': ':.3f', 'FullText': True, 'Display': False, 'ScoreText': False}
        )

        fig.update_layout(

            xaxis=dict(
                range=[0, 1.05],
                showgrid=True,
                gridcolor='LightGray'
            ),

            yaxis=dict(
                categoryorder='total ascending',
                showticklabels=True
            ),

            height=max(250, len(df_viz) * 45),
            margin=dict(l=10, r=10, t=40, b=20),

            uniformtext_minsize=8,
            uniformtext_mode='hide',
            title={
                'text': "Top K 相似样本匹配度分析",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )

        # 更新条形图样式和文本位置
        fig.update_traces(
            marker_color='#1f77b4',  # 设置一个专业蓝色
            textposition='outside',  # 将文本放在条形外部
            textfont_size=12
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- 其他日志解析部分 (保持不变) ---
    gen_matches = re.findall(r'生成新候选:\s*([\s\S]*?)(?=择优 API|原句|\[第|$)', log_txt)
    valid_gens = [g.strip() for g in gen_matches if g.strip()]
    if valid_gens:
        st.markdown("**大模型生成的新候选**")
        for g in valid_gens:
            st.info(g)

    api_matches = re.findall(r'择优 API 结果:\s*(.*)', log_txt)
    if api_matches:
        st.markdown("**择优 API 决策轨迹**")
        trace = " -> ".join([f"` {m.strip()} `" for m in api_matches])
        st.write(trace)


def show_analysis_expander(stats, title):
    if not stats or 's_cnt' not in stats:
        st.info("等待加载数据后生成分析报表...")
        return

    with st.expander(f"{title} 数据分析 (下拉查看)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            if sum(stats['s_cnt'].values()) > 0:
                df = pd.DataFrame(list(stats['s_cnt'].items()), columns=['Label', 'Count'])
                fig = px.pie(df, values='Count', names='Label', title="情感分布", height=250)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("无情感数据")
        with c2:
            if stats['top_a']:
                df = pd.DataFrame(list(stats['top_a'].items()), columns=['Aspect', 'Count'])
                fig = px.bar(df, x='Aspect', y='Count', title="Top 10 属性", height=250)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("无属性数据")


# ==========================================
# 3. 文件与历史记录管理
# ==========================================

def save_eval_result(dataset, filename, method, n_val, p, r, f1):
    new_row = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Dataset": dataset, "Filename": filename,
        "Method": method, "N-Gram": n_val,
        "Precision": f"{p:.2f}%", "Recall": f"{r:.2f}%", "F1": f"{f1:.2f}%"
    }
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(HISTORY_FILE, index=False)


def save_file_with_progress(uploaded_file):
    clean_name = re.sub(r'[\\/*?:"<>|]', "", uploaded_file.name)
    path = os.path.join(UPLOAD_DIR, clean_name)
    if os.path.exists(path):
        timestamp = datetime.now().strftime("%H%M%S")
        parts = os.path.splitext(clean_name)
        clean_name = f"{parts[0]}_{timestamp}{parts[1]}"
        path = os.path.join(UPLOAD_DIR, clean_name)

    progress_bar = st.progress(0, text="上传中...")
    with open(path, "wb") as f:
        bytes_data = uploaded_file.getvalue()
        total_size = len(bytes_data)
        chunk_size = max(total_size // 50, 1024)
        for i in range(0, total_size, chunk_size):
            f.write(bytes_data[i:i + chunk_size])
            progress_bar.progress(min(100, int((i + chunk_size) / total_size * 100)))
    progress_bar.empty()
    return clean_name


def get_files_df(ext):
    data = []
    if not os.path.exists(UPLOAD_DIR): return pd.DataFrame()
    for f in os.listdir(UPLOAD_DIR):
        if f.endswith(ext) and "history" not in f:
            path = os.path.join(UPLOAD_DIR, f)
            stat = os.stat(path)
            data.append({
                "文件名": f,
                "上传时间": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "大小 (KB)": round(stat.st_size / 1024, 2),
                "完整路径": path
            })
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    return df.sort_values(by="文件名", ascending=True, key=lambda col: col.str.lower())


# ==========================================
# 4. 主界面逻辑
# ==========================================
def main():
    st.sidebar.header("全局资源监控与配置")

    if "json_data" in st.session_state:
        cost, in_tokens, out_tokens = estimate_deepseek_cost(st.session_state.json_data)
        st.sidebar.markdown("### API 消耗预估")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("预估输入", f"{in_tokens / 1000:.1f}k T")
        c2.metric("预估输出", f"{out_tokens / 1000:.1f}k T")
        st.sidebar.markdown(
            f"<div style='padding:10px; border-radius:5px; border:1px solid #d3d3d3; text-align:center;'>"
            f"<span style='font-size:12px; color:gray;'>本次批处理预计成本</span><br>"
            f"<span style='font-size:20px; font-weight:bold; color:#d9534f;'>¥ {cost:.4f} 元</span>"
            f"</div>", unsafe_allow_html=True
        )
    else:
        st.sidebar.info("请先加载 JSON 文件以查看消耗预估")

    st.sidebar.divider()

    st.sidebar.markdown("### 核心算法参数")
    ngram_n = st.sidebar.slider(
        "N-Gram 相似度窗口",
        min_value=1, max_value=20, value=4,
        help="用于在知识库中匹配相似样本的 N-gram 字符窗口大小。值越大，匹配越严格。"
    )
    apply_ngram_patch(ngram_n)

    # <<< FIX 1: 添加动态示例数量控制滑块 >>>
    dyn_ex_count = st.sidebar.slider(
        "动态示例数量",
        min_value=1, max_value=10, value=6,
        help="控制 LLM 生成新候选时，从知识库中挑选的最相似示例的数量。"
    )

    st.sidebar.divider()

    # 模型配置
    DEFAULT_MODEL_CONFIGS = {
        "deepseek-v3": {"key": "239cee64-362b-4277-aa6a-1fba3626cb6b", "url": "https://ai.shiep.edu.cn/api/share"},
        "gpt-5-nano": {"key": "sk-hNVUK8iCqiqwDtJFeW9qZJXSeMlYiBfvb0hGdG3kPqMo4GvA",
                       "url": "https://91vip.futureppo.top"},
        "doubao-seed-2.0-pro": {"key": "sk-KZ1lbFFiaKiqFA5PdkkqqMGrXgoM9eMGM08GGxUOlZCSAyer",
                                "url": "https://91vip.futureppo.top"},
        "qwen-2.5-7b-instruct": {"key": "sk-338lVRL0R0jV4vl6NZ6a1r8EUslWer6QoOzRFB5Wj6nmjfxq",
                                 "url": "https://91vip.futureppo.top"},
        "glm-4.5-air": {"key": "sk-sLajqHW3AkTwnE8yEX5t63QiJy6l9zQCDZ8UYEydbF0hfVJk",
                        "url": "https://91vip.futureppo.top"},
    }

    if 'MODEL_CONFIGS' not in st.session_state:
        st.session_state.MODEL_CONFIGS = DEFAULT_MODEL_CONFIGS.copy()

    model_options = list(st.session_state.MODEL_CONFIGS.keys())
    new_model_index = st.session_state.get('new_model_index', 0)
    model = st.sidebar.selectbox(
        "推理模型选择",
        model_options,
        index=new_model_index,
        key="model_selector"
    )
    st.session_state['new_model_index'] = 0

    with st.sidebar.expander("添加自定义模型"):
        with st.form("new_model_form"):
            new_model_name = st.text_input("模型名称 (例如 my-new-model)")
            new_api_key = st.text_input("API Key", type="password")
            new_api_base = st.text_input("Base URL (例如 https://api.example.com)")
            submitted = st.form_submit_button("添加并应用")
            if submitted:
                if new_model_name and new_api_key and new_api_base:
                    st.session_state.MODEL_CONFIGS[new_model_name] = {"key": new_api_key, "url": new_api_base}
                    st.session_state['new_model_index'] = len(st.session_state.MODEL_CONFIGS) - 1
                    st.success(f"模型 '{new_model_name}' 已添加并选中!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("请填写所有字段！")

    selected_cfg = st.session_state.MODEL_CONFIGS[model]
    api_key = selected_cfg["key"]
    api_base = selected_cfg["url"]

    st.sidebar.caption(f"当前 URL: {api_base}")
    st.sidebar.caption(f"当前 Key: {api_key[:8]}...{api_key[-4:]}")

    llm_select_topk.API_URL = api_base
    llm_select_topk.YOUR_DEEPSEEK_API_KEY = api_key
    llm_select_topk.MODEL_NAME = model

    if hasattr(onlyselect, 'API_URL'):
        onlyselect.API_URL = api_base
        onlyselect.YOUR_DEEPSEEK_API_KEY = api_key
        onlyselect.MODEL_NAME = model

    st.title("大模型驱动的情感细粒度分析处理平台")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. 数据仓库", "2. 单条调试", "3. 自动化批处理",
        "4. 评测历史榜单", "5. Bad Case 诊断"
    ])

    # --- TAB 1: 数据仓库 ---
    with tab1:
        c_j, c_t = st.columns(2)

        with c_j:
            st.subheader("待处理 JSON 队列")
            up_j = st.file_uploader("上传候选集 JSON", type=['json'], key="up_j")
            if up_j and st.button("保存文件", key="s_j"):
                saved_name = save_file_with_progress(up_j)
                st.success(f"已保存: {saved_name}")
                time.sleep(0.5);
                st.rerun()

            df_j = get_files_df(".json")
            if not df_j.empty:
                st.dataframe(df_j[["文件名", "上传时间", "大小 (KB)"]], hide_index=True, use_container_width=True)
                sel_j = st.selectbox("工作区载入 JSON", df_j["文件名"].tolist(), key="sel_j")
                path_j = df_j[df_j["文件名"] == sel_j]["完整路径"].values[0]

                col1, col2, col3, col4 = st.columns([1.5, 4.5, 1.5, 1.5])
                with col1:
                    if st.button("加载分析", key="l_j", type="primary", use_container_width=True):
                        try:
                            with open(path_j, 'r', encoding='utf-8') as f:
                                tmp = json.load(f)
                            st.session_state.json_data = tmp
                            st.session_state.json_name = sel_j
                            st.session_state.json_full_path = path_j
                            st.session_state.current_dataset = tmp.get('dataset', 'unknown')
                            st.session_state.json_stats = calc_stats(tmp.get('examples', []))
                            st.toast("JSON加载成功")
                        except Exception as e:
                            st.error(str(e))
                with col2:
                    new_name_j = st.text_input("重命名为", value=sel_j, label_visibility="collapsed",
                                               key=f"rn_in_j_{sel_j}")
                with col3:
                    if st.button("重命名", key="rn_btn_j", use_container_width=True):
                        if new_name_j and new_name_j != sel_j:
                            new_path = os.path.join(UPLOAD_DIR, new_name_j)
                            if not os.path.exists(new_path):
                                os.rename(path_j, new_path);
                                st.success("重命名成功");
                                time.sleep(0.5);
                                st.rerun()
                            else:
                                st.error("文件名已存在")
                with col4:
                    if st.button("删除", key="d_j", use_container_width=True):
                        os.remove(path_j);
                        st.rerun()

                if st.session_state.get("json_data") and st.session_state.get("json_name") == sel_j:
                    st.divider();
                    st.success(f"当前数据流: {sel_j}")
                    show_analysis_expander(st.session_state.get("json_stats"), "JSON 统计特征")

        with c_t:
            st.subheader("领域知识库 (TXT)")
            up_t = st.file_uploader("上传 train.txt", type=['txt'], key="up_t")
            if up_t and st.button("保存文件", key="s_t"):
                saved_name = save_file_with_progress(up_t)
                st.success(f"已保存: {saved_name}")
                time.sleep(0.5);
                st.rerun()

            df_t = get_files_df(".txt")
            if not df_t.empty:
                st.dataframe(df_t[["文件名", "上传时间", "大小 (KB)"]], hide_index=True, use_container_width=True)
                sel_t = st.selectbox("工作区载入知识库", df_t["文件名"].tolist(), key="sel_t")
                path_t = df_t[df_t["文件名"] == sel_t]["完整路径"].values[0]

                col1, col2, col3, col4 = st.columns([1.5, 4.5, 1.5, 1.5])
                with col1:
                    if st.button("加载解析", key="l_t", type="primary", use_container_width=True):
                        try:
                            with open(path_t, 'r', encoding='utf-8') as f:
                                content = f.read()
                            exs = parse_train_txt(content)
                            st.session_state.train_data = exs
                            st.session_state.train_name = sel_t
                            st.session_state.train_stats = calc_stats(exs)
                            llm_select_topk.TRAIN_EXAMPLES = exs
                            if hasattr(onlyselect, 'TRAIN_EXAMPLES'):
                                onlyselect.TRAIN_EXAMPLES = exs
                            st.toast(f"成功注入 {len(exs)} 条领域先验知识")
                        except Exception as e:
                            st.error(str(e))
                with col2:
                    new_name_t = st.text_input("重命名为", value=sel_t, label_visibility="collapsed",
                                               key=f"rn_in_t_{sel_t}")
                with col3:
                    if st.button("重命名", key="rn_btn_t", use_container_width=True):
                        if new_name_t and new_name_t != sel_t:
                            new_path = os.path.join(UPLOAD_DIR, new_name_t)
                            if not os.path.exists(new_path):
                                os.rename(path_t, new_path);
                                st.success("重命名成功");
                                time.sleep(0.5);
                                st.rerun()
                            else:
                                st.error("文件名已存在")
                with col4:
                    if st.button("删除", key="d_t", use_container_width=True):
                        os.remove(path_t);
                        st.rerun()

                if st.session_state.get("train_data") and st.session_state.get("train_name") == sel_t:
                    st.divider();
                    st.success(f"当前知识库: {sel_t}")
                    show_analysis_expander(st.session_state.get("train_stats"), "先验数据特征")

    # --- TAB 2: 单条调试 ---
    with tab2:
        if "json_data" not in st.session_state:
            st.warning("拦截：请先在数据仓库中加载待处理的 JSON 文件。")
        else:
            exs = st.session_state.json_data['examples']
            if "c_idx" not in st.session_state: st.session_state.c_idx = 0

            c_p, c_i, c_n = st.columns([1, 4, 1])
            with c_p:
                if st.button("上一条记录"): st.session_state.c_idx = max(0, st.session_state.c_idx - 1)
            with c_n:
                if st.button("下一条记录"): st.session_state.c_idx = min(len(exs) - 1, st.session_state.c_idx + 1)
            with c_i:
                st.markdown(f"<div style='text-align:center'>记录索引: {st.session_state.c_idx + 1} / {len(exs)}</div>",
                            unsafe_allow_html=True)

            curr = exs[st.session_state.c_idx]
            c1, c2 = st.columns(2)
            with c1:
                st.text_area("自然语言输入 (Input)", curr['input'], height=100)
                cands = curr.get('model_output_top3', []) or curr.get('model_output_top7', []) or curr.get(
                    'model_output_top9', [])
                st.info("\n".join([f"[{i}] {c}" for i, c in enumerate(cands)]))
            with c2:
                st.write("标注基准 (Ground Truth):")
                st.dataframe(pd.DataFrame(curr['target'], columns=["Sub", "Obj", "Asp", "Opi", "Sent"]),
                             hide_index=True)

                method = st.radio("推理策略", ["混合模式 (Select+Gen)", "纯选择 (OnlySelect)"], horizontal=True)
                target_mod = llm_select_topk if "混合" in method else onlyselect

                col_btn_run, col_btn_stop = st.columns(2)
                with col_btn_run:
                    run_single = st.button("启动单步推理探针", type="primary", use_container_width=True)
                with col_btn_stop:
                    st.button("强制中断调试", use_container_width=True)

                if run_single:
                    if "train_data" not in st.session_state:
                        st.warning("警告：尚未注入领域知识库，相似度匹配引擎将降级。")
                        if hasattr(llm_select_topk, 'TRAIN_EXAMPLES'):
                            llm_select_topk.TRAIN_EXAMPLES = []
                        if hasattr(onlyselect, 'TRAIN_EXAMPLES'):
                            onlyselect.TRAIN_EXAMPLES = []
                    else:
                        if hasattr(llm_select_topk, 'TRAIN_EXAMPLES'):
                            llm_select_topk.TRAIN_EXAMPLES = st.session_state.train_data
                        if hasattr(onlyselect, 'TRAIN_EXAMPLES'):
                            onlyselect.TRAIN_EXAMPLES = st.session_state.train_data

                    log_capture = io.StringIO()
                    original_stdout = sys.stdout
                    res = ""

                    try:
                        sys.stdout = TeeStdout(log_capture)
                        st.info("指令已下发至推理集群，请耐心等待...")

                        with st.spinner("推理引擎正在执行链路择优..."):
                            if "混合" in method:
                                current_cands = cands.copy()
                                for round_num in range(5):
                                    print(f"\n>>> [第 {round_num + 1} 轮循环深度寻优]")
                                    res = target_mod.select_best_candidate_from_model_output(curr['input'],
                                                                                             current_cands)
                                    if res is not None: break

                                    # <<< FIX 1 (CONTINUED): 将滑块值传递给后端函数 >>>
                                    new_cand = target_mod.generate_new_model_output(curr['input'],
                                                                                    dyn_ex_count=dyn_ex_count)

                                    if new_cand:
                                        current_cands.append(new_cand)
                                    else:
                                        break
                            else:
                                res = target_mod.select_best_candidate_from_model_output(curr['input'], cands)

                            if not res:
                                res = cands[0] if cands else ""
                                st.warning("提示：大模型未能返回有效格式，已回退采用第一条候选。")

                        sys.stdout = original_stdout
                        log_txt = log_capture.getvalue()
                        if log_txt:
                            render_backend_logs_visuals(log_txt)
                            with st.expander("查看底层网络与处理日志"):
                                st.code(log_txt)

                        st.success("抽取完成，标准化输出 (Result):")
                        st.code(res)
                        spans = target_mod.extract_spans_para(res)
                        if spans:
                            try:
                                st.dataframe(pd.DataFrame(spans, columns=["主体", "客体", "属性", "观点", "情感"]),
                                             hide_index=True)
                            except:
                                st.warning("模型输出格式不规则：");
                                st.write(spans)
                        else:
                            st.info("未能从中提取到符合规范的五元组。")
                    except Exception as e:
                        sys.stdout = original_stdout
                        st.error("执行时发生严重崩溃：")
                        st.code(traceback.format_exc(), language="python")

    # --- TAB 3: 自动化批处理 ---
    with tab3:
        if "json_data" not in st.session_state:
            st.warning("拦截：��先加载待处理 JSON 队列")
        else:
            json_path = st.session_state.json_full_path
            base_name = os.path.basename(json_path)
            exs = st.session_state.json_data['examples']
            total = len(exs)

            st.header(f"批处理调度池: {base_name}")

            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                method = st.radio("系统运行模式", ["混合模式", "纯选择模式"])
            with col_cfg2:
                dataset_options = ["car", "ele", "car-ele", "ele-car", "自定义..."]
                default_tag = st.session_state.get('current_dataset', 'unknown')

                try:
                    default_index = dataset_options.index(default_tag)
                except ValueError:
                    default_index = len(dataset_options) - 1

                selected_tag = st.selectbox(
                    "数据集溯源标签 (Dataset)",
                    dataset_options,
                    index=default_index
                )

                if selected_tag == "自定义...":
                    custom_value = default_tag if default_tag not in dataset_options[:-1] else ""
                    task_tag = st.text_input("输入自定义标签", value=custom_value)
                else:
                    task_tag = selected_tag

            # <<< FIX 2: 多断点恢复机制 >>>
            st.divider()
            st.subheader("断点与会话管理")

            # 动态生成模式标签
            mode_tag = "mix" if "混合" in method else "only"

            # 扫描现有的 checkpoint 文件
            try:
                # +++ 修正后的代码 +++
                base_name_no_ext = os.path.splitext(base_name)[0]
                ckpt_prefix = f"ckpt_{task_tag}_{mode_tag}_{base_name_no_ext}"
                ckpt_files = [f for f in os.listdir(CHECKPOINT_DIR) if
                              f.startswith(ckpt_prefix) and f.endswith('.json')]
            except FileNotFoundError:
                ckpt_files = []

            session_options = ["** 启动一个全新会话**"] + sorted(ckpt_files, reverse=True)

            selected_session = st.selectbox("选择一个会话以继续或启动新会话:", session_options)

            proc_exs = [];
            start = 0
            # 定义 ckpt_path 变量
            ckpt_path = None
            if selected_session != session_options[0]:
                # 加载选中的断点
                ckpt_path = os.path.join(CHECKPOINT_DIR, selected_session)
                try:
                    with open(ckpt_path, 'r', encoding='utf-8') as f:
                        saved = json.load(f)
                        proc_exs = saved.get('examples', []);
                        start = len(proc_exs)
                        st.success(f"已加载断点: {selected_session}。将从第 {start + 1} 条记录开始。")
                except Exception as e:
                    st.error(f"加载断点失败: {e}")
                    ckpt_path = None  # 加载失败，阻止执行
            else:
                # 启动新会话，生成新的 checkpoint 文件名
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                ckpt_name = f"ckpt_{task_tag}_{mode_tag}_{base_name.split('.')[0]}_{timestamp}.json"
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
                st.info("将启动一个全新的处理会话。")

            if "stop_flag" not in st.session_state: st.session_state.stop_flag = False

            c_s, c_e = st.columns([1, 1])
            with c_s:
                start_btn = st.button("投递集群运算", type="primary", disabled=(ckpt_path is None))
            with c_e:
                if st.button("强制终止进程"): st.session_state.stop_flag = True

            if start_btn:
                st.session_state.stop_flag = False
                if "train_data" in st.session_state:
                    if hasattr(llm_select_topk, 'TRAIN_EXAMPLES'):
                        llm_select_topk.TRAIN_EXAMPLES = st.session_state.train_data
                    if hasattr(onlyselect, 'TRAIN_EXAMPLES'):
                        onlyselect.TRAIN_EXAMPLES = st.session_state.train_data
                else:
                    st.warning("警告：由于缺乏领域知识库，相似度计算将受限。")

                target_mod = llm_select_topk if "混合" in method else onlyselect
                res_obj = {
                    'dataset': task_tag, 'top_k': st.session_state.json_data.get('top_k', 0),
                    'f1': st.session_state.json_data.get('f1', 0), 'method': mode_tag,
                    'ngram': ngram_n, 'examples': proc_exs
                }

                bar = st.progress(start / total if total > 0 else 0, text="握手中...")
                log_box = st.empty();
                start_time = time.time()

                for i in range(start, total):
                    if st.session_state.stop_flag:
                        st.warning("进程已挂起，当前计算状态已写入持久层。");
                        break

                    ex = exs[i];
                    inp = ex['input']
                    cands = ex.get('model_output_top3', []) or ex.get('model_output_top7', []) or ex.get(
                        'model_output_top9', [])

                    log_capture = io.StringIO();
                    original_stdout = sys.stdout
                    try:
                        sys.stdout = TeeStdout(log_capture)
                        if "混合" in method:
                            current_cands = cands.copy();
                            final = None
                            for round_num in range(5):
                                print(f"\n>>> [第 {round_num + 1} 轮循环深度寻优]")
                                final = target_mod.select_best_candidate_from_model_output(inp, current_cands)
                                if final is not None: break
                                new_cand = target_mod.generate_new_model_output(inp, dyn_ex_count=dyn_ex_count)
                                if new_cand:
                                    current_cands.append(new_cand)
                                else:
                                    break
                        else:
                            final = target_mod.select_best_candidate_from_model_output(inp, cands)

                        if not final: final = cands[0] if cands else ""
                        pred = target_mod.extract_spans_para(final)
                    except Exception as e:
                        print(f"Error: {e}");
                        final = "";
                        pred = []
                    finally:
                        sys.stdout = original_stdout

                    log_box.code(log_capture.getvalue()[-2000:], language='log')
                    res_obj['examples'].append(
                        {"input": inp, "target": ex['target'], "prediction": pred, "model_output": final})

                    with open(ckpt_path, 'w', encoding='utf-8') as f:
                        json.dump(res_obj, f, indent=2, ensure_ascii=False)

                    elapsed = time.time() - start_time;
                    processed = i - start + 1
                    if processed > 0:
                        avg_time = elapsed / processed;
                        remain = (total - i - 1) * avg_time
                        bar.progress((i + 1) / total, text=f"系统进度: {i + 1}/{total} | TTA: {remain / 60:.1f} 分钟")

                if not st.session_state.stop_flag:
                    final_name = f"processed_{task_tag}_{mode_tag}_{base_name}"
                    final_path = os.path.join(UPLOAD_DIR, final_name)
                    if os.path.exists(final_path):
                        timestamp = datetime.now().strftime("%H%M%S")
                        final_name = f"processed_{task_tag}_{mode_tag}_{os.path.splitext(base_name)[0]}_{timestamp}.json"
                        final_path = os.path.join(UPLOAD_DIR, final_name)

                    os.rename(ckpt_path, final_path)
                    st.success(f"批处理完成，产出文件已进入数据池: {final_name}")

    # --- TAB 4: F1 历史榜单 ---
    with tab4:
        st.header("系统评测大盘")
        st.subheader("创建新评测任务")
        df_res = get_files_df(".json")
        proc_files = df_res[df_res['文件名'].str.contains("processed")]

        if not proc_files.empty:
            sel_res = st.selectbox("选择参与评测的产出文件", proc_files["文件名"].tolist())
            path_res = proc_files[proc_files["文件名"] == sel_res]["完整路径"].values[0]

            if st.button("启动算法对齐与打分"):
                try:
                    with open(path_res, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    record_dataset = data.get('dataset', 'unknown')
                    lbls = [x.get('target', []) for x in data['examples']]
                    prds = [x.get('prediction', []) for x in data['examples']]
                    s = evaluate_f1.compute_f1_scores(prds, lbls)
                    p, r, f1 = s['precision'] * 100, s['recall'] * 100, s['f1'] * 100

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Precision", f"{p:.2f}%");
                    c2.metric("Recall", f"{r:.2f}%");
                    c3.metric("F1 Score", f"{f1:.2f}%")

                    meta_method = data.get('method', 'Unknown');
                    meta_n = data.get('ngram', 'Unknown')
                    save_eval_result(record_dataset, sel_res, meta_method, meta_n, p, r, f1)
                    st.success(f"指标计算完毕，已持久化至 [{record_dataset}] 的业务集。")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"打分组件运行异常: {str(e)}")

        st.divider()
        st.subheader("历代模型效果排行")
        if os.path.exists(HISTORY_FILE):
            try:
                df_hist = pd.read_csv(HISTORY_FILE)
            except pd.errors.EmptyDataError:
                df_hist = pd.DataFrame()

            if df_hist.empty:
                st.info("暂无评测记录档案，请先创建评测任务。")
            else:
                # <<< FIX: 移除 Top 5 排行榜，只保留趋势图 >>>

                # 预处理数据
                for col in ['Precision', 'Recall', 'F1']:
                    df_hist[f'{col}_val'] = pd.to_numeric(df_hist[col].astype(str).str.rstrip('%'), errors='coerce')

                all_datasets = df_hist['Dataset'].unique()
                for ds in all_datasets:
                    st.markdown(f"### {str(ds).upper()} 业务视图")
                    df_ds = df_hist[df_hist['Dataset'] == ds].copy()

                    # --- 添加交互式筛选器 ---
                    c_filter1, c_filter2 = st.columns(2)
                    with c_filter1:
                        methods = df_ds['Method'].unique()
                        sel_methods = st.multiselect(
                            f"筛选运行模式 ({ds})",
                            options=methods,
                            default=methods,
                            key=f"method_{ds}"
                        )
                    with c_filter2:
                        ngrams = sorted(df_ds['N-Gram'].unique())
                        sel_ngrams = st.multiselect(
                            f"筛选N-Gram值 ({ds})",
                            options=ngrams,
                            default=ngrams,
                            key=f"ngram_{ds}"
                        )

                    filtered_df = df_ds[
                        (df_ds['Method'].isin(sel_methods)) &
                        (df_ds['N-Gram'].isin(sel_ngrams))
                        ]

                    if filtered_df.empty:
                        st.warning("当前筛选条件下无数据。")
                        continue

                    # --- 渲染图表 (只保留趋势图) ---
                    st.markdown("#####  F1分数趋势")
                    trend_df = filtered_df.sort_values(by="Time")
                    # 使用不同的颜色来区分不同的运行模式
                    fig_line = px.line(
                        trend_df,
                        x="Time", y="F1_val",
                        color="Method",  # 新增：按方法区分颜色
                        markers=True,
                        hover_data=["Filename", "N-Gram"],
                        labels={"F1_val": "F1 Score (%)", "Method": "运行模式"},
                        height=400  # 增加图表高度
                    )
                    fig_line.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_line, use_container_width=True)

                    # --- 显示筛选后的数据表 ---
                    st.dataframe(
                        filtered_df.sort_values(by="F1_val", ascending=False)[
                            ['Time', 'Filename', 'Method', 'N-Gram', 'Precision', 'Recall', 'F1']],
                        hide_index=True,
                        use_container_width=True
                    )
                    st.write("")

        else:
            st.info("暂无评测记录档案，请先创建评测任务。")

        st.divider()
        st.subheader("底层数据治理")
        if os.path.exists(HISTORY_FILE) and 'df_hist' in locals() and not df_hist.empty:
            with st.expander("数据修改", expanded=False):
                record_list = df_hist.apply(lambda row: f"[{row['Dataset']}] {row['Filename']} ({row['Time']})",
                                            axis=1).tolist()
                selected_record_str = st.selectbox("选中要干预的异常记录", record_list)
                if selected_record_str:
                    sel_idx = record_list.index(selected_record_str)
                    target_row = df_hist.iloc[sel_idx]
                    new_dataset_tag = st.text_input("重置数据归属标签", value=target_row['Dataset'])
                    col_u, col_d = st.columns([1, 4])
                    with col_u:
                        if st.button("下发更新指令", type="primary"):
                            if new_dataset_tag and new_dataset_tag != target_row['Dataset']:
                                df_hist.at[sel_idx, 'Dataset'] = new_dataset_tag
                                df_hist.to_csv(HISTORY_FILE, index=False)
                                target_json_path = os.path.join(UPLOAD_DIR, target_row['Filename'])
                                if os.path.exists(target_json_path):
                                    try:
                                        with open(target_json_path, 'r', encoding='utf-8') as f:
                                            j_data = json.load(f)
                                        j_data['dataset'] = new_dataset_tag
                                        with open(target_json_path, 'w', encoding='utf-8') as f:
                                            json.dump(j_data, f, indent=2, ensure_ascii=False)
                                    except:
                                        pass
                                st.success(f"记录已更新至域 '{new_dataset_tag}'。");
                                time.sleep(1.5);
                                st.rerun()
                    with col_d:
                        if st.button("逻辑删除记录"):
                            df_hist = df_hist.drop(index=sel_idx);
                            df_hist.to_csv(HISTORY_FILE, index=False)
                            st.success("操作成功");
                            time.sleep(1);
                            st.rerun()

    # --- TAB 5: Bad Case 深度诊断室 ---
    with tab5:
        st.header("Bad Case 深度诊断室")

        # <<< FIX 1: 简化说明文字，移除 e.g. 示例 >>>
        st.markdown("""
        **配色与错误类型规范：**
        - **<span style="color:#28a745">绿色 (Ground Truth)</span>**: 在左侧“标准答案”列，所有项均为标准答案，统一用绿色表示。
        - **<span style="color:green">绿色 (命中)</span>**: 在右侧“模型预测”列，与标准答案完全一致的项。
        - **<span style="color:purple">紫色 (纯幻觉)</span>**: 在右侧“模型预测”列，无中生有的项。
        - **<span style="color:#fd7e14">橙色 (字段错误)</span>**: 在右侧“模型预测”列，与标准答案相比，某个或多个字段存在差异。具体错误类型会以标签标出：
            - `主体错误`: 主体字段不匹配。
            - `客体错误`: 客体字段不匹配。
            - `属性错误`: 属性字段不匹配。
            - `观点错误`: 观点字段不匹配。
            - `情感错误`: 情感标签不匹配。
            - `混合错误`: 两种或以上字段同时错误。
        """, unsafe_allow_html=True)

        df_res_bad = get_files_df(".json")
        proc_files_bad = df_res_bad[df_res_bad['文件名'].str.contains("processed")]

        if not proc_files_bad.empty:
            sel_res_bad = st.selectbox("导入预测结果库进行切片分析", proc_files_bad["文件名"].tolist(), key="bc_select")
            path_res_bad = proc_files_bad[proc_files_bad["文件名"] == sel_res_bad]["完整路径"].values[0]

            try:
                with open(path_res_bad, 'r', encoding='utf-8') as f:
                    bad_data = json.load(f)

                all_examples = bad_data.get('examples', [])

                pure_bad_cases = []
                partial_correct_cases = []
                total_correct_samples = 0

                for ex in all_examples:
                    def clean_quads(quads):
                        cleaned = []
                        if quads is None: return set()
                        for q in quads:
                            if q and any(q):
                                cleaned.append(tuple(str(item).strip() for item in q))
                        return set(cleaned)

                    t_set = clean_quads(ex.get('target', []))
                    p_set = clean_quads(ex.get('prediction', []))

                    if not t_set and not p_set:
                        total_correct_samples += 1
                        continue

                    tp = t_set & p_set
                    fn = t_set - p_set
                    fp = p_set - t_set

                    case_data = {"input": ex.get('input', ''), "tp": tp, "fn": fn, "fp": fp}
                    if t_set == p_set:
                        total_correct_samples += 1
                    elif len(tp) > 0:
                        partial_correct_cases.append(case_data)
                    else:
                        pure_bad_cases.append(case_data)

                # <<< FIX 2: 简化统计指标，移除 FN/FP 总数 >>>
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("总样本数", len(all_examples))
                m2.metric("完全正确样本", total_correct_samples)
                m3.metric("部分正确样本", len(partial_correct_cases))
                m4.metric("纯粹错误样本", len(pure_bad_cases))

                def render_quad_html(quad, color, label=""):
                    q_str = " | ".join([x if x else "空" for x in quad])
                    label_html = f"<span style='font-size:10px; color:white; background:{color}; padding:1px 6px; border-radius:3px; margin-right:6px;'>{label}</span>" if label else ""
                    return f"<div style='border-left: 4px solid {color}; background-color:#fafafa; padding:8px; margin-bottom:6px; font-family:monospace; border-radius:3px;'>{label_html}<span style='color:{color};'>[{q_str}]</span></div>"

                def render_case(case_data):
                    tp = case_data['tp'];
                    fn = case_data['fn'];
                    fp = case_data['fp']

                    paired_matches = {}
                    remaining_fn = list(fn)

                    for fp_item in fp:
                        best_match_fn = None
                        highest_match_count = 2

                        temp_remaining_fn = list(remaining_fn)
                        for fn_item in temp_remaining_fn:
                            match_count = sum(1 for a, b in zip(fp_item, fn_item) if a == b)
                            if match_count > highest_match_count:
                                highest_match_count = match_count
                                best_match_fn = fn_item

                        if best_match_fn:
                            paired_matches[fp_item] = best_match_fn
                            remaining_fn.remove(best_match_fn)

                    pure_hallucination = fp - set(paired_matches.keys())

                    col_t, col_p = st.columns(2)
                    with col_t:
                        st.markdown("**Ground Truth (标准答案)**")
                        ground_truth_color = "#28a745"
                        for q in tp: st.markdown(render_quad_html(q, ground_truth_color), unsafe_allow_html=True)
                        for q in fn: st.markdown(render_quad_html(q, ground_truth_color), unsafe_allow_html=True)

                    with col_p:
                        st.markdown("**Model Prediction (模型预测)**")

                        for q in tp: st.markdown(render_quad_html(q, "#28a745", "命中"), unsafe_allow_html=True)

                        for fp_item, fn_item in paired_matches.items():
                            errors = []
                            fields = ["主体", "客体", "属性", "观点", "情感"]
                            for i in range(5):
                                if fp_item[i] != fn_item[i]:
                                    errors.append(fields[i])

                            if len(errors) == 1:
                                label = f"{errors[0]}错误"
                            else:
                                label = "混合错误"
                            st.markdown(render_quad_html(fp_item, "#fd7e14", label), unsafe_allow_html=True)

                        for q in pure_hallucination:
                            st.markdown(render_quad_html(q, "#6f42c1", "纯幻觉"), unsafe_allow_html=True)

                        if not tp and not paired_matches and not pure_hallucination:
                            st.warning("模型无任何输出")

                st.divider()

                if pure_bad_cases:
                    st.error(f"**第一诊断层：纯粹错误样本 ({len(pure_bad_cases)} 条)**")
                    st.write("在这些案例中，模型未能正确命中任何一个标准答案。")
                    for idx, bc in enumerate(pure_bad_cases):
                        st.markdown(f"**Case {idx + 1} (纯粹错误):**  `{bc['input']}`")
                        render_case(bc)
                        st.divider()

                if partial_correct_cases:
                    st.warning(f"**第二诊断层：部分正确样本 ({len(partial_correct_cases)} 条)**")
                    st.write("在这些案例中，模型至少命中了一个标准答案，但仍存在漏抽或幻觉。")
                    for idx, bc in enumerate(partial_correct_cases):
                        st.markdown(f"**Case {idx + 1} (部分正确):**  `{bc['input']}`")
                        render_case(bc)
                        st.divider()

                if not pure_bad_cases and not partial_correct_cases:
                    st.success("🎉 恭喜！在此次分析中没有发现任何差异样本。")

            except Exception as e:
                st.error(f"解析失败: {str(e)}")
                st.code(traceback.format_exc())
        else:
            st.info("尚未生成预测文件，请先前往批处理模块执行任务。")

if __name__ == "__main__":
    main()