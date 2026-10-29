"""Project configuration · 路徑與常數."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
UCI_DIR = RAW_DIR / "UCI HAR Dataset"
FITBIT_DIR = RAW_DIR / "fitbit"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# UCI HAR：6 種活動（官方代碼 → 名稱）
ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# 取樣率（Hz）與視窗長度（UCI HAR 慣例）
SAMPLING_RATE = 50
WINDOW_SIZE = 128

# Fitbit 心率最大值估計（220 - 年齡；實機版可由個人設定覆寫）
DEFAULT_MAX_HR = 190

# 心率強度分區（%HRmax；Karvonen 式分區簡化版）
HR_ZONES = [
    ("rest", 0, 50),
    ("light", 50, 60),
    ("moderate", 60, 70),
    ("vigorous", 70, 85),
    ("near_max", 85, 101),
]
