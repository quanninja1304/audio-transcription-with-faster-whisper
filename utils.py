# utils.py
# Chứa các hàm tiện ích

import os
import json
import random
import subprocess
import numpy as np
import pandas as pd
import torch

def set_seed(seed_value):
    """
    Cài đặt seed cho tất cả các thư viện để đảm bảo tính tái lập.
    """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Seed ({seed_value}) đã được cài đặt.")

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


def get_audio_duration(audio_path):
    """
    Dùng subprocess để lấy thời lượng audio.
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
