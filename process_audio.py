import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import subprocess  # <-- THAY THẾ cho '!'
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from faster_whisper import WhisperModel

# -----------------------------------------------
# CÀI ĐẶT SEED VÀ CÁC BIẾN TOÀN CỤC
# -----------------------------------------------

SEED_VALUE = 42
torch.manual_seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_VALUE)

# -----------------------------------------------
# KHỞI TẠO GEMINI
# -----------------------------------------------

# PROMPT
SYSTEM_PROMPT = """
Bạn là một trợ lý AI chuyên nghiệp, nhiệm vụ của bạn là trích xuất thông tin đơn hàng
từ một đoạn văn bản phiên âm (transcript).

Yêu cầu:
1. Phân tích kỹ văn bản được cung cấp.
2. Xác định TẤT CẢ các sản phẩm được đề cập.
3. Với mỗi sản phẩm, trích xuất chính xác: Tên sản phẩm, Số lượng (phải là SỐ), và Đơn vị tính.
4. Trả lời CHÍNH XÁC dưới dạng một mảng JSON (JSON array).
5. KHÔNG thêm bất kỳ văn bản nào khác ngoài mảng JSON.

Ví dụ định dạng JSON mong muốn:
[
  {"ten_san_pham": "Tên A", "so_luong": 10, "don_vi": "thùng"},
  {"ten_san_pham": "Sản phẩm B", "so_luong": 5, "don_vi": "cái"}
]
"""

# Define JSON (schema)
output_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "ten_san_pham": {"type": "STRING"},
            "so_luong": {"type": "NUMBER"},
            "don_vi": {"type": "STRING"}
        },
        "required": ["ten_san_pham", "so_luong", "don_vi"]
    }
}

# Biến toàn cục cho model
gemini_model_global = None
gemini_config_global = None

# -----------------------------------------------
# UTILS
# -----------------------------------------------

def initialize_gemini():
    """
    Đọc API key từ biến môi trường và khởi tạo model Gemini.
    """
    global gemini_model_global, gemini_config_global
    try:
        # Đọc từ Biến môi trường
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("LỖI: Biến môi trường 'GOOGLE_API_KEY' không được tìm thấy.")
            print("Hãy chạy 'export GOOGLE_API_KEY=your_key_here' trước khi chạy script.")
            return False
            
        genai.configure(api_key=api_key)

        gemini_model_global = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT
        )
        gemini_config_global = GenerationConfig(
            response_mime_type="application/json",
            response_schema=output_schema
        )
        print("Gemini model và config đã được khởi tạo.")
        return True
        
    except Exception as e:
        print(f"LỖI: Không thể khởi tạo Gemini: {e}")
        return False


def extract_products_with_gemini(transcript_text, max_retries=3):
    """
    Gửi văn bản đến Gemini, sử dụng model và config toàn cục.
    Bao gồm cơ chế tự động thử lại (retry) khi gặp lỗi 429.
    """
    if not transcript_text:
        print("Input text is empty, skipping API call.")
        return None
        
    if not gemini_model_global or not gemini_config_global:
        print("LỖI: Model Gemini chưa được khởi tạo.")
        return None

    wait_time = 5 # Thời gian chờ ban đầu
    
    for attempt in range(max_retries):
        try:
            response = gemini_model_global.generate_content(
                transcript_text,
                generation_config=gemini_config_global
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


def save_to_excel(data, excel_path):
    """
    Lưu dữ liệu DataFrame ra file Excel.
    """
    if not data:
        print("Không có dữ liệu để lưu ra Excel.")
        return

    try:
        df = pd.DataFrame(data)
        df = df[["ten_san_pham", "so_luong", "don_vi"]] # Đảm bảo đúng thứ tự cột
        df.to_excel(excel_path, index=False)
        print(f"Excel file saved successfully!\nFilepath: {excel_path}")
    except Exception as e:
        print(f"ERROR while saving Excel file: {e}")


def load_whisper_model(model_size):
    """
    Tải model Whisper (đã tối ưu).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "default"
    
    print(f"Đang tải Whisper model '{model_size}' (device: {device}, compute: {compute_type})...")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Whisper model loaded.")
        return model
    except Exception as e:
        print(f"LỖI khi tải model Whisper: {e}")
        return None

def get_audio_duration(audio_path):
    """
    THAY THẾ cho !ffprobe: Dùng subprocess để lấy thời lượng audio.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        return float(probe_data['format']['duration'])
    except FileNotFoundError:
        print("LỖI: Lệnh 'ffprobe' không tìm thấy. Hãy đảm bảo FFmpeg đã được cài đặt.")
        return None
    except Exception as e:
        print(f"WARNING: Không thể lấy thời lượng audio: {e}")
        return None

# -----------------------------------------------
# MAIN PROCESS
# -----------------------------------------------

def process_single_file(whisper_model, audio_path, excel_output_path):
    """
    Chạy toàn bộ pipeline cho 1 file audio.
    """
    print("-" * 50)
    print(f"Bắt đầu xử lý file: {audio_path}")
    total_start_time = time.time()

    # Tạo thư mục output nếu chưa có
    try:
        TRANSCRIPT_PATH = os.path.dirname(excel_output_path)
        os.makedirs(TRANSCRIPT_PATH, exist_ok=True)
    except Exception as e:
        print(f"Lỗi khi tạo thư mục: {e}")
        return

    # Lấy thời lượng (tùy chọn)
    # audio_duration = get_audio_duration(audio_path)
    # if audio_duration:
    #     print(f"Audio duration: {audio_duration:.2f}s")

    # ----- PHASE 1: WHISPER (từ model "small") -----
    phase1_start = time.time()
    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            language="vi",
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
            beam_size=3,
            temperature=0,
        )
        segments_list = list(segments)
        phase1_end = time.time()
        print(f"phase 1 (Whisper): {(phase1_end - phase1_start):.3f} sec")
    except Exception as e:
        print(f"LỖI trong Phase 1 (Whisper): {e}")
        return

    # ----- BỎ HOÀN TOÀN PHASE 1B (GHI JSON) -----

    # ----- PHASE 2: GEMINI (từ bộ nhớ) -----
    phase2_start = time.time()

    # TẠO TEXT TRỰC TIẾP TỪ BỘ NHỚ
    transcript_text = " ".join([segment.text.strip() for segment in segments_list])

    if transcript_text:
        extracted_data = extract_products_with_gemini(transcript_text)
        if extracted_data:
            save_to_excel(extracted_data, excel_output_path)
        else:
            print("Gemini API call failed or returned no data.")
    else:
        print("Whisper returned an empty transcript.")

    phase2_end = time.time()
    print(f"phase 2 (Gemini + Excel): {(phase2_end - phase2_start):.3f} sec")

    total_end_time = time.time()
    print(f"** Total processed time: {(total_end_time - total_start_time):.3f} sec **")


# -----------------------------------------------
# CHECKPOINT
# -----------------------------------------------

if __name__ == "__main__":
    # --- CONFIG ---
    AUDIO_FILE = "sample_audio/2.wav" 
    EXCEL_OUTPUT = "transcripts/2_gemini_order.xlsx"

    # 1. Khởi tạo model
    whisper_model = load_whisper_model(model_size="medium")
    gemini_ready = initialize_gemini()

    # 2. main process
    if whisper_model and gemini_ready:
        process_single_file(whisper_model, AUDIO_FILE, EXCEL_OUTPUT)
    else:
        print("LỖI: Không thể khởi tạo model, script bị dừng.") 