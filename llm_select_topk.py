# -*- coding: utf-8 -*-
"""
COQE 五元组后处理：从 model_output_top5 中选择完整候选（支持 [SSEP] 多五元组）
输入：T5-Top5 输出的 JSON（含 model_output_top7）
输出：符合要求的 JSON
"""

import json
import re
import time
import openai
import os

# ==============================
# 配置区
# ==============================
API_URL = "https://ai.shiep.edu.cn/api/share"
MODEL_NAME = "deepseek-v3"
YOUR_DEEPSEEK_API_KEY = "239cee64-362b-4277-aa6a-1fba3626cb6b"
MAX_RETRIES = 3
API_TIMEOUT_PER_CHUNK = 10

# 全局缓存 train 数据
TRAIN_EXAMPLES = None


def load_train_examples(train_file="data/car/train.txt"):
    """加载 train.txt 数据，格式：sentence\tlabel_count\n[quad1]\n[quad2]..."""
    global TRAIN_EXAMPLES
    if TRAIN_EXAMPLES is not None:
        return TRAIN_EXAMPLES

    examples = []
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # 第一行：句子 + 标签数量
            if '\t' in line:
                parts = line.split('\t')
                input_text = parts[0].strip()
                label_count = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 0

                quads = []
                # 读取接下来的 label_count 行（或直到空行/下一句）
                j = i + 1
                read_count = 0
                # 增加对label_count为0或不存在时的容错
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line or '\t' in next_line:
                        break  # 遇到空行或下一句的开始，则停止读取

                    # 解析五元组 [[...];[...];[...];[...];[...]]
                    if next_line.startswith('[[') and next_line.endswith(']]'):
                        quad_str = next_line[2:-2]  # 移除外层 [[ ]]
                        quad_parts = quad_str.split(';')
                        if len(quad_parts) == 5:
                            processed_quad = []
                            for part in quad_parts:
                                part = part.strip()
                                if part.startswith('[') and part.endswith(']'):
                                    content = part[1:-1]  # 移除 [ ]
                                    if content:
                                        # 提取 & 后面的字符（忽略位置信息）
                                        tokens = [token.split('&', 1)[1] for token in content.split() if '&' in token]
                                        field_text = ''.join(tokens)
                                        processed_quad.append(field_text)
                                    else:
                                        processed_quad.append("")
                                else:
                                    processed_quad.append("")

                            if len(processed_quad) == 5:
                                quads.append(processed_quad)
                            read_count += 1

                    if label_count > 0 and read_count >= label_count:
                        break
                    j += 1

                examples.append({
                    'input': input_text,
                    'target': quads
                })
                i = j
            else:
                i += 1

    except FileNotFoundError:
        print(f"⚠️ 警告：未找到训练文件 {train_file}，将使用静态示例")
        examples = []

    TRAIN_EXAMPLES = examples
    return examples


# === 新增：字符级 n-gram 相似度 ===
def get_char_ngrams(text, n=4):
    """获取字符 n-gram（去除标点和空格）"""
    # 只保留中文、字母、数字
    clean_text = re.sub(r'[^\w\u4e00-\u9fff]', '', text.lower())
    if len(clean_text) < n:
        return set(clean_text) if clean_text else set()
    return set(clean_text[i:i + n] for i in range(len(clean_text) - n + 1))


def calculate_similarity(text1, text2):
    """计算字符 n-gram 相似度（Jaccard 系数）"""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    ngrams1 = get_char_ngrams(text1, n=4)
    ngrams2 = get_char_ngrams(text2, n=4)

    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union)


def format_examples_for_prompt(examples):
    """将示例列表转换为标准 prompt 格式"""
    formatted = ""
    for inp, out in examples:
        formatted += f"""原句：{inp}
候选：
--- 【候选 0】 ---
{out}
输出:0

"""
    return formatted.strip()


def build_dynamic_examples(original_sentence, top_k=6):
    """
    从 train 数据中选择最相似的示例（使用字符 n-gram 相似度）
    修改：将 top_k 的默认值从 7 改为 6，以匹配 UI 滑块的默认值。
    """
    train_examples = load_train_examples()

    if not train_examples:
        # 回退到静态示例
        static_examples = [
            ("相机比手机好，因为续航长，但重量重。", "相机相比手机是1因为续航是长 [SSEP] 相机相比手机是-1因为重量是重"),
            ("屏幕比平板电脑清晰很多。", "相比平板电脑是1因为屏幕是清晰很多"),
            ("音质一般，不如老款。", "相比老款是-1因为音质是一般"),
            ("整部机都显得比较小巧，和我女友的8250差不多", "相比8250是0因为是差不多")
        ]
        return format_examples_for_prompt(static_examples)

    # 计算相似度并排序
    similarities = sorted(
        [(calculate_similarity(original_sentence, ex['input']), ex) for ex in train_examples],
        key=lambda x: x[0],
        reverse=True
    )

    # 调试：打印最高分的样本
    print(f"\n🔍 原句: {original_sentence}")
    print(f"📊 相似度最高的样本 (展示前6, 使用 top_k={top_k}):")
    for i, (sim, ex) in enumerate(similarities[:6]):
        print(f"  {sim:.4f}: {ex['input'][:60]}...")

    # 构建示例对
    examples = []
    # 这里使用传入的 top_k 参数
    for _, ex in similarities[:top_k]:
        input_text = ex['input']
        target_quads = ex['target']

        # 将 target 转换为 model_output 格式
        output_parts = []
        for quad in target_quads:
            sub, ob, ap, op, pr = quad
            part = f"{sub + ' ' if sub else ''}相比{ob}是{pr}因为{ap}是{op}"
            output_parts.append(part.strip())

        output_text = " [SSEP] ".join(output_parts) if output_parts else "这不是对比句"
        examples.append((input_text, output_text))

    return format_examples_for_prompt(examples)


def call_deepseek_api_stream(prompt, max_retries=MAX_RETRIES):
    """调用 DeepSeek API，支持流式响应"""
    for attempt in range(1, max_retries + 1):
        try:
            client = openai.OpenAI(api_key=YOUR_DEEPSEEK_API_KEY, base_url=API_URL)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": (
                        "你是一个专业的数据集生成器。\n"
                        "输出必须是纯文本，不含任何解释、markdown、代码块。\n"
                        "严格按用户要求的格式生成。"
                    )},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )

            full_content = ""
            last_time = time.time()
            for chunk in response:
                now = time.time()
                if now - last_time > API_TIMEOUT_PER_CHUNK:
                    raise TimeoutError(f"流式响应中断：{API_TIMEOUT_PER_CHUNK}秒无新数据")
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_content += delta.content
                    last_time = now
            return full_content.strip()

        except Exception as e:
            print(f"⚠️ API 调用失败（第 {attempt}/{max_retries} 次）: {type(e).__name__}: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"⏳ {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print("❌ 所有重试失败")
                return None

    return None


def extract_spans_para(seq):
    """从自然语言句子中提取五元组（支持 [SSEP]）"""
    quads = []
    if seq is None or not isinstance(seq, str) or seq == '这不是对比句':
        return quads

    try:
        sents = [s.strip() for s in seq.split('[SSEP]') if s.strip()]
        for s in sents:
            if not s or '因为' not in s or '相比' not in s:
                continue

            parts = s.split('因为', 1)
            sub_ob_pr = parts[0].strip()
            ap_op = parts[1].strip()

            sub_parts = sub_ob_pr.split('相比', 1)
            if len(sub_parts) != 2:
                continue
            sub = sub_parts[0].strip()
            ob_pr = sub_parts[1].strip()

            # 分离客体和偏好
            if '是' in ob_pr:
                ob_parts = ob_pr.rsplit('是', 1)
                ob = ob_parts[0].strip()
                pr = ob_parts[1].strip()
            else:
                ob = ob_pr
                pr = ""

            # 分离属性和观点
            if '是' in ap_op:
                ap_parts = ap_op.rsplit('是', 1)
                ap = ap_parts[0].strip()
                op = ap_parts[1].strip()
            else:
                ap = ap_op
                op = ""

            # 清理空格
            sub = sub.replace(' ', '')
            ob = ob.replace(' ', '')
            ap = ap.replace(' ', '')
            op = op.replace(' ', '')
            pr = pr.replace(' ', '')

            if any([sub, ob, ap, op, pr]):
                quads.append([sub, ob, ap, op, pr])
    except Exception as e:
        print(f"⚠️ 解析失败: {seq} -> {e}")

    return quads


def select_best_candidate_from_model_output(original_sentence, model_outputs):
    if not model_outputs:
        return None

    train_examples = load_train_examples()
    if train_examples:
        similarities = sorted(
            [(calculate_similarity(original_sentence, ex['input']), ex) for ex in train_examples],
            key=lambda x: x[0],
            reverse=True
        )
        print(f"\n🔍 原句: {original_sentence}")
        for sim, ex in similarities[:6]:
            print(f"{sim:.4f}: {ex['input']}")

    candidate_display = ""
    for idx, output in enumerate(model_outputs):
        candidate_display += f"--- 【候选 {idx}】 ---\n{output}\n\n"

    selection_prompt = f"""### 📌 任务说明：COQE 候选审核
你正在审核 T5 模型生成的完整候选。
- **目标**：选择一个最符合以下标准的完整候选：
  1. 所有五元组都符合规范（如果含多个五元组）
  2. 句式为："主体相比客体是偏好因为属性是观点"。
  3. 主体为比较的主方，若未提及,可忽略不写；
     客体为被比较的对象；
     属性必须是原文中的单一词（如"屏幕""音质"）；
     观点为原文中的比较描述（如"耐用"）；
     偏好必须是数字："2"(无关)、"1"（正向）、"-1"（负向）、"0"（无差异）
  4. 所有内容必须直接源自原句，且连续出现。这很重要.

### 📌 示例
原句：相机比手机好，因为续航长，但重量重。
候选：
--- 【候选 0】 ---
相机相比手机是1因为续航是长 [SSEP] 相机相比手机是-1因为重量是重
输出:0

原句：屏幕比平板电脑清晰很多。
候选：
--- 【候选 0】 ---
相比平板电脑是1因为屏幕是清晰很多
输出:0

原句：音质一般，不如老款。
候选：
--- 【候选 0】 ---
相比老款是-1因为音质是一般
输出:0

原句：整部机都显得比较小巧，和我女友的8250差不多
候选：
--- 【候选 0】 ---
相比8250是0因为是差不多
输出:0

### 📌 原句
{original_sentence}

### 📌 候选列表
{candidate_display}

### 📌 输出要求
- 如果存在合规候选，**仅输出其序号**（如：`0`）
- 如果所有候选都不合规，**仅输出：`NONE`**
- **严禁输出任何其他文字、解释或标点**
""".strip()

    res = call_deepseek_api_stream(selection_prompt)
    print(f"🎯 择优 API 结果: {res}")

    if res and "NONE" not in res and res.strip().isdigit():
        idx = int(res.strip())
        if 0 <= idx < len(model_outputs):
            return model_outputs[idx]

    return None


def generate_new_model_output(original_sentence, dyn_ex_count=6):
    """
    生成新的完整候选（可能含 [SSEP]）
    ### 关键修改 ###
    添加 dyn_ex_count 参数，并将其传递给 build_dynamic_examples。
    """
    # 获取格式化后的动态示例，并传入从UI获取的数量
    formatted_examples = build_dynamic_examples(original_sentence, top_k=dyn_ex_count)

    generation_prompt = f"""### 📌 任务说明：COQE 完整候选生成
请寻找对比关系，生成一个完整的对比观点候选。

### 📌 规范
- 如果有多个比较，用 " [SSEP] " 分隔
- 句式："主体相比客体是偏好因为属性是观点"
- 偏好必须是数字："2" (无关)/"1"（正向）/"-1"（负向）/"0"（无差异）
- 属性必须是单一词
- 内容必须源自原句
- 无比较时输出："这不是对比句"
- 句式为："主体相比客体是偏好因为属性是观点"。
- 其中主体为比较的主方，若未提及,可忽略不写；客体为被比较的对象；属性必须是原文中的单一词（如“屏幕”）；观点为原文中的比较描述（如“耐用”）；偏好必须是数字："1"（正向）、"-1"（负向）、"0"（无差异）

### 📌 强制约束
1. **必须基于原句**：不能添加原句中不存在的信息
2. **属性必须是单一词**：如"电池"、"屏幕"，不能是"电池续航"
3. **偏好必须是数字**："2"、"1"、"-1"、"0"
4. **主体可省略**：比较的主方。如果原句无主语，不要强行添加
5. **观点**：必须是原文中的比较描述
6. **宁可少，不要错**：如果不确定，输出"这不是对比句"

### 📌你该如何完成 
**首先**，确定句子是否有对比关系
**然后找到句子主体**：即比较的对象
**再然后是客体**：即被比较的对象，两者都只需要筛选出关键的词语即可。
**之后确定偏好**：必须是数字:"2"(无关)/"1"（正向）/"-1"（负向）/"0"（无差异）
**再确定属性**：即在哪一方面，主体与客体是在哪一方面的比较，从文中找出
**观点**：即从文中的哪些描述可以看出主体相对于客体的比较关系
**注意以上元素若从文中无法找出可忽略**

### 📌 句式模板
"主体相比客体是偏好因为属性是观点"

### 📌 示例
{formatted_examples}

### 📌 原句
{original_sentence}

### 📌 输出（仅一行，无任何其他内容）
""".strip()

    res = call_deepseek_api_stream(generation_prompt)
    print(f"🆕 生成新候选: {res}")
    return res.strip() if res else None


def main():
    input_json_path = "llmxunhuan/car_ele_top7_oracle.json"
    output_json_path = "llmxunhuan/car_ele-COQE_top7_llm_selected.json"

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {
        'dataset': data['dataset'],
        'top_k': data['top_k'],
        'f1': data['f1'],
        'examples': []
    }

    for example in data['examples']:
        original_input = example['input']
        target_quads = example['target']
        top7_candidates = example.get('model_output_top7', [])  # 使用 .get 增加鲁棒性

        selected_candidate = None
        current_candidates = top7_candidates.copy()

        # 最多循环 5 次
        for round_num in range(5):
            selected_candidate = select_best_candidate_from_model_output(original_input, current_candidates)
            if selected_candidate is not None:
                break

            # 生成新候选
            # 注意：当从命令行直接运行此文件时，这里会使用 dyn_ex_count 的默认值 6
            new_candidate = generate_new_model_output(original_input)
            if new_candidate:
                current_candidates.append(new_candidate)
            else:
                break

        # 构建最终结果
        if selected_candidate is not None:
            final_pred = extract_spans_para(selected_candidate)
            selected_text = selected_candidate
        else:
            # 回退到 T5 第一条
            selected_text = top7_candidates[0] if top7_candidates else ""
            final_pred = extract_spans_para(selected_text)

        new_example = {
            "input": original_input,
            "target": target_quads,
            "prediction": final_pred,
            "model_output": selected_text
        }
        results['examples'].append(new_example)

    # 保存
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ LLM 后处理完成！结果保存至: {output_json_path}")


if __name__ == "__main__":
    main()