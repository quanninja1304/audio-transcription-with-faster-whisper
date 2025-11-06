# main.py
import os
import time

# Import các module
import config
import utils
import models
import services

def process_single_file(whisper_model, gemini_model, gemini_config, audio_path, excel_output_path):
    """
    Chạy toàn bộ pipeline cho 1 file audio.
    Hàm này nhận các model đã được khởi tạo làm tham số.
    """
    print("-" * 50)
    print(f"Bắt đầu xử lý file: {audio_path}")
    total_start_time = time.time()

    # ----- PHASE 1: WHISPER -----
    phase1_start = time.time()
    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            language="vi",
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
            beam_size=5,
            temperature=0,
        )
        segments_list = list(segments)
        phase1_end = time.time()
        print(f"phase 1 (Whisper): {(phase1_end - phase1_start):.3f} sec")
    except Exception as e:
        print(f"LỖI trong Phase 1 (Whisper): {e}")
        return

    # ----- PHASE 2: GEMINI (từ bộ nhớ) -----
    phase2_start = time.time()

    # TẠO TEXT TRỰC TIẾP
    transcript_text = " ".join([segment.text.strip() for segment in segments_list])

    if transcript_text:
        # Gọi services
        extracted_data = services.extract_products_with_gemini(
            transcript_text, gemini_model, gemini_config
        )
        
        if extracted_data:
            # Gọi utils
            utils.save_to_excel(extracted_data, excel_output_path)
        else:
            print("Gemini API call failed or returned no data.")
    else:
        print("Whisper returned an empty transcript.")

    phase2_end = time.time()
    print(f"phase 2 (Gemini + Excel): {(phase2_end - phase2_start):.3f} sec")

    total_end_time = time.time()
    print(f"** Total processed time: {(total_end_time - total_start_time):.3f} sec **")


if __name__ == "__main__":
    # 1. Cài đặt seed
    utils.set_seed(config.SEED_VALUE)

    # 2. Khởi tạo whisper + gemini
    print("--- Khởi tạo Model ---")
    whisper_model = models.load_whisper_model(model_size=config.WHISPER_MODEL_SIZE)
    gemini_model, gemini_config = models.initialize_gemini()

    # 3. main process
    if whisper_model and gemini_model and gemini_config:
        process_single_file(
            whisper_model=whisper_model,
            gemini_model=gemini_model,
            gemini_config=gemini_config,
            audio_path=config.DEFAULT_AUDIO_FILE,
            excel_output_path=config.DEFAULT_EXCEL_OUTPUT
        )
    else:
        print("LỖI: Không thể khởi tạo model, script bị dừng.")