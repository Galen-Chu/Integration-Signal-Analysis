"""Dataset download · UCI HAR 直接下載；Fitbit 走 kagglehub（公開集免帳號）."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = ("https://archive.ics.uci.edu/static/public/240/"
           "human+activity+recognition+using+smartphones.zip")

# Kaggle: FitBit Fitness Tracker Data（33 位 Fitbit 使用者 03-04.2016 分鐘級資料）
FITBIT_DATASET = "mobius/fitbit-fitness-tracker-data"


def _progress(block_num: int, block_size: int, total: int):
    done = block_num * block_size
    if total > 0 and block_num % 200 == 0:
        pct = min(done / total * 100, 100)
        sys.stdout.write(f"\r  {pct:5.1f}%")
        sys.stdout.flush()


def download_uci(raw_dir: Path) -> Path:
    """下載並解壓 UCI HAR（外層 zip 內還有一層 UCI HAR Dataset.zip）."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    uci_root = raw_dir / "UCI HAR Dataset"
    if (uci_root / "train").is_dir():
        print("UCI HAR already present, skip.")
        return uci_root

    print(f"Downloading UCI HAR from {UCI_URL} …")
    data = urllib.request.urlopen(UCI_URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as outer:
        inner_name = next((n for n in outer.namelist()
                           if n.endswith("UCI HAR Dataset.zip")), None)
        if inner_name is None:
            # 有些版本外層就是資料夾
            outer.extractall(raw_dir)
        else:
            inner_bytes = outer.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                inner.extractall(raw_dir)
    assert (uci_root / "train").is_dir(), "unexpected archive layout"
    print(f"\nUCI HAR extracted to {uci_root}")
    return uci_root


def download_fitbit(raw_dir: Path) -> Path:
    """Fitbit 資料集（kagglehub）。公開資料集通常免帳號；失敗時給出手動指引."""
    try:
        import kagglehub
    except ImportError:
        print("請先 pip install kagglehub，或見 README 手動下載指引。")
        raise SystemExit(2)

    print(f"Fetching {FITBIT_DATASET} via kagglehub …")
    path = Path(kagglehub.dataset_download(FITBIT_DATASET))
    dest = raw_dir / "fitbit"
    if not dest.is_symlink() and not dest.is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(path, target_is_directory=True) if hasattr(Path, "symlink_to") else None
    if not dest.is_dir():  # symlink 建立失敗（Windows 權限）→ 直接記錄 kagglehub 路徑
        dest = path
    print(f"Fitbit data at {path}")
    return dest


def download_all(raw_dir: Path, include_fitbit: bool = True):
    download_uci(raw_dir)
    if include_fitbit:
        try:
            download_fitbit(raw_dir)
        except SystemExit:
            raise
        except Exception as e:  # kagglehub 失敗不阻斷 UCI 管線
            print(f"⚠ Fitbit 下載失敗（{e}）——UCI 管線不受影響；"
                  "手動下載說明見 README。")
    return True
