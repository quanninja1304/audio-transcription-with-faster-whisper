# models.py
# Chứa logic khởi tạo và tải model

import os
import torch
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from faster_whisper import WhisperModel

# Import cấu hình từ config.py
import config

def initialize_gemini():
    """
    Đọc API key từ biến môi trường và khởi tạo, trả về model & config Gemini.
    """
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("LỖI: Biến môi trường 'GOOGLE_API_KEY' không được tìm thấy.")
            return None, None
            
        genai.configure(api_key=api_key)

        gemini_model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL_NAME,
            system_instruction=config.SYSTEM_PROMPT
        )
        gemini_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=config.OUTPUT_SCHEMA
        )
        print("Gemini model và config đã được khởi tạo.")
        return gemini_model, gemini_config
        
    except Exception as e:
        print(f"LỖI: Không thể khởi tạo Gemini: {e}")
        return None, None

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