# config.py

# --- Cài đặt chung ---
SEED_VALUE = 42

# --- Cài đặt Model ---
WHISPER_MODEL_SIZE = "medium"
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# --- Path ---
DEFAULT_AUDIO_FILE = "sample_audio/2.wav"
DEFAULT_EXCEL_OUTPUT = "transcripts/2_wav_gemini_order.xlsx"
TRANSCRIPT_OUTPUT_DIR = "transcripts" # Thư mục output chung


# --- Gemini Prompt & Schema ---

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

OUTPUT_SCHEMA = {
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