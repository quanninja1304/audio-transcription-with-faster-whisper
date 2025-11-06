# services.py

import json
import time

def extract_products_with_gemini(transcript_text, model, config, max_retries=3):
    """
    Gửi văn bản đến Gemini, sử dụng model và config được truyền vào.
    Bao gồm cơ chế tự động thử lại (retry) khi gặp lỗi 429.
    """
    if not transcript_text:
        print("Input text is empty, skipping API call.")
        return None
        
    if not model or not config:
        print("LỖI: Model hoặc Config của Gemini không hợp lệ.")
        return None

    wait_time = 5 # Thời gian chờ ban đầu
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                transcript_text,
                generation_config=config
            )
            parsed_data = json.loads(response.text)
            return parsed_data # Thành công

        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"WARNING: Rate Limit (429). Chờ {wait_time}s... (Lần {attempt + 1})")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                print(f"ERROR while calling Gemini API (không thể phục hồi): {e}")
                if "response" in locals():
                    print(f"Returned content: {response.text}")
                return None
    return None