# -*- coding: utf-8 -*-
"""
COQE 五元组后处理：从 model_output_top5 中选择完整候选（仅挑选，不生成）
输入：T5-Top5 输出的 JSON（含 model_output_top5）
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
        if '[SSEP]' in seq:
            sents = [s.strip() for s in seq.split('[SSEP]') if s.strip()]
        else:
            sents = [seq.strip()] if seq.strip() else []

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
                ob_parts = ob_pr.split('是', 1)
                ob = ob_parts[0].strip()
                pr = ob_parts[1].strip()
            else:
                ob = ob_pr
                pr = ""

            # 分离属性和观点
            if '是' in ap_op:
                ap_parts = ap_op.split('是', 1)
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
    """
    从 model_output_top7 中选择一个完整候选（可能含 [SSEP]）
    返回: 最佳候选字符串 或 None
    """
    if not model_outputs:
        return None

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
     属性必须是单一词（如“屏幕”）；
     观点为原文中的比较描述（如“耐用”）；
     偏好必须是数字："2"(无关)、"1"（正向）、"-1"（负向）、"0"（无差异）
  4. 所有内容必须直接源自原句，且连续出现

  
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

    if res and "NONE" not in res:
        match = re.search(r'\d+', res)
        if match:
            idx = int(match.group())
            if 0 <= idx < len(model_outputs):
                return model_outputs[idx]
    
    return None


def main():
    input_json_path = "llmxunhuan/ele_top9_oracle.json"   
    output_json_path = "llmxunhuan/ele-COQE_selected.json"

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
        top9_candidates = example['model_output_top9']

        # 只做一次挑选（不生成）
        selected_candidate = select_best_candidate_from_model_output(original_input, top9_candidates)

        # 构建最终结果
        if selected_candidate is not None:
            final_pred = extract_spans_para(selected_candidate)
            selected_text = selected_candidate
        else:
            # 回退到 T5 第一条
            selected_text = top9_candidates[0] if top9_candidates else ""
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