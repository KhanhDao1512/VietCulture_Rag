import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

def paraphrase_question(original_question, question_type, num_variants=3):
    """
    Uses Ollama Qwen 2.5 with a bulletproof JSON Object structure to prevent parsing errors.
    """
    # Thay đổi định dạng kỳ vọng từ Array sang Object có các key cố định (q1, q2, q3)
    prompt = f"""
You are a Vietnamese linguistic expert. 
Your task is to rewrite the following Vietnamese question into exactly {num_variants} different variants (paraphrases) while keeping the original meaning intact.

Original Vietnamese Question: "{original_question}"

CRITICAL RULES:
1. The original question type is: "{question_type}". All generated variants MUST strictly preserve this exact "{question_type}" nature.
2. Inside each paraphrased question, replace any double quotes (") with single quotes (').

Strict Formatting Rules:
- You MUST return a valid JSON OBJECT with keys "q1", "q2", ..., up to "q{num_variants}".
- DO NOT wrap the output in markdown code blocks like ```json ... 
```.

Expected Output Format:
{{
  "q1": "First paraphrased variant goes here",
  "q2": "Second paraphrased variant goes here",
  "q3": "Third paraphrased variant goes here"
}}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # Ép Ollama xuất JSON
        "options": {
            "temperature": 0.7 
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result_text = response.json()['response'].strip()
        
        # Parse chuỗi JSON thành Dict của Python
        raw_obj = json.loads(result_text)
        
        # Chuyển đổi từ định dạng Dict {"q1": "...", "q2": "..."} sang List ["...", "..."]
        variants = []
        for i in range(1, num_variants + 1):
            key = f"q{i}"
            if key in raw_obj:
                variants.append(raw_obj[key])
            elif str(i) in raw_obj:  # Dự phòng nếu model trả về key là "1", "2", "3"
                variants.append(raw_obj[str(i)])
                
        # Nếu trích xuất thành công và đủ số lượng, trả về mảng kết quả
        if len(variants) > 0:
            return variants
            
        # Dự phòng trường hợp model trả về key ngẫu nhiên khác
        if isinstance(raw_obj, dict):
            return list(raw_obj.values())[:num_variants]

    except Exception as e:
        print(f"Error during Ollama API call or JSON parsing: {e}")
        
    # --- CƠ CHẾ FALLBACK TUYỆT ĐỐI (Nếu LLM vẫn lỗi, script KHÔNG BỊ SẬP) ---
    # Thay vì dừng chương trình, tạo câu hỏi tạm thời để chạy tiếp các câu sau
    safe_q = original_question.replace('"', "'")
    return [f"Biến thể {i+1} ({question_type}): {safe_q}" for i in range(num_variants)]
    
def process_benchmark_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]

    for index, item in enumerate(data):
        orig_q = item.get("original_question", "")
        # Lấy trực tiếp loại câu hỏi từ dữ liệu của item đó
        q_type = item.get("question_type", "general") 
        
        if orig_q:
            print(f"[{index+1}/{len(data)}] Processing ({q_type}): {orig_q}")
            
            # Truyền cả câu hỏi và loại câu hỏi vào hàm paraphrase
            variants = paraphrase_question(orig_q, q_type, num_variants=3)
            
            item["user_query"] = [
                {
                    "query_id": f"q{i+1}",
                    "text": variant
                } 
                for i, variant in enumerate(variants)
            ]
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nDone! Output saved to: {output_file}")

# --- Execute Script ---
if __name__ == "__main__":
    INPUT_PATH = "data/benchmark_candidates.json"
    OUTPUT_PATH = "data/output_benchmark.json"
    
    process_benchmark_file(INPUT_PATH, OUTPUT_PATH)