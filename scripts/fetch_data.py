"""fetch_data · 下載資料集.

    python scripts/fetch_data.py --all          # UCI + Fitbit
    python scripts/fetch_data.py --uci          # 只下載 UCI HAR
    python scripts/fetch_data.py --fitbit       # 只下載 Fitbit（kagglehub）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.isa.config import RAW_DIR  # noqa: E402
from src.isa.data.download import (download_fitbit, download_uci)  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="下載專案資料集")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="UCI + Fitbit")
    group.add_argument("--uci", action="store_true", help="只下載 UCI HAR")
    group.add_argument("--fitbit", action="store_true", help="只下載 Fitbit")
    args = parser.parse_args()

    if args.all or args.uci or not any(vars(args).values()):
        download_uci(RAW_DIR)
    if args.all or args.fitbit:
        download_fitbit(RAW_DIR)


if __name__ == "__main__":
    main()
