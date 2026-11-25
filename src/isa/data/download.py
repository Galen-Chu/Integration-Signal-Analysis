"""Dataset download · UCI HAR/PAMAP2 直接下載（斷點續傳＋重試）；Fitbit 走 kagglehub.

UCI archive 用 chunked 傳輸且實測會中途斷線（680MB 的 PAMAP2 在 ~281MB 處
IncompleteRead）——urlopen().read() 一把抓的寫法會整趟作廢。所有下載走
_download_with_resume：逐塊寫盤、斷線後以 Range 續傳、伺服器不支援 Range
（回 200）則整檔重下，全部失敗才 raise.
"""

import http.client
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
)

# PAMAP2：心率＋三 IMU 同步量測（9 人×18 活動×100Hz）——整合分析主力
PAMAP2_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"

# Kaggle: FitBit Fitness Tracker Data（33 位 Fitbit 使用者 03-04.2016 分鐘級資料）
FITBIT_DATASET = "mobius/fitbit-fitness-tracker-data"

CHUNK_BYTES = 1 << 20  # 1MB 逐塊寫盤
DOWNLOAD_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 10  # 測試以 monkeypatch 調小，避免重試測試拖秒


def _download_with_resume(url: str, dest: Path, attempts: int = DOWNLOAD_ATTEMPTS) -> None:
    """串流下載 url → dest（檔案），中斷則 Range 續傳，最多 attempts 次.

    timeout=300 是單次 socket 讀取上限（UCI 伺服器慢，~400KB/s 是常態），
    不是總時長上限——進度每塊 flush，未來 CI 日誌看得到活性.
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        pos = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url)
        if pos:
            req.add_header("Range", f"bytes={pos}-")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resume = pos > 0 and resp.status == 206
                with dest.open("ab" if resume else "wb") as fh:
                    total = None
                    length = resp.headers.get("Content-Length")
                    if length:
                        # 期望總量：206＝Range 續傳（剩餘長度＋既有位元組）；
                        # 200＝整檔。Content-Length 回應的早 EOF 不會 raise
                        # IncompleteRead（只有 chunked 會）——必須自己比對短讀
                        expected = pos if resume else 0
                        total = expected + int(length)
                    while True:
                        chunk = resp.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        fh.write(chunk)
                        if total:
                            done = fh.tell()
                            print(
                                f"\r  {done / total * 100:5.1f}% ({done >> 20} MB)",
                                end="",
                                flush=True,
                            )
                    if total is not None and fh.tell() < total:
                        raise http.client.IncompleteRead(
                            b"", fh.tell() - pos if resume else fh.tell()
                        )
            print(flush=True)
            return
        except urllib.error.HTTPError:
            raise  # 4xx/5xx 是確定性錯誤，重試無意義
        except http.client.IncompleteRead as e:
            # 關鍵：e.partial 是「最後一次 read() 已收到但未回傳給呼叫端」的
            # bytes——不補寫就續傳會跳過這段、檔案從斷點起整段錯位
            partial = e.partial or b""
            if partial:
                with dest.open("ab") as fh:
                    fh.write(partial)
            size = dest.stat().st_size if dest.exists() else 0
            last_err = e
            print(
                f"\n  attempt {attempt}/{attempts} interrupted at {size >> 20} MB "
                f"(partial {len(partial)} bytes salvaged): {e!r}",
                flush=True,
            )
            time.sleep(min(RETRY_BACKOFF_SECONDS * attempt, 30))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            size = dest.stat().st_size if dest.exists() else 0
            last_err = e
            print(
                f"\n  attempt {attempt}/{attempts} interrupted at {size >> 20} MB: {e!r}",
                flush=True,
            )
            time.sleep(min(RETRY_BACKOFF_SECONDS * attempt, 30))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_err


def download_uci(raw_dir: Path) -> Path:
    """下載並解壓 UCI HAR（外層 zip 內還有一層 UCI HAR Dataset.zip）."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    uci_root = raw_dir / "UCI HAR Dataset"
    if (uci_root / "train").is_dir():
        print("UCI HAR already present, skip.")
        return uci_root

    outer_zip = raw_dir / "_uci_har_outer.zip"
    print(f"Downloading UCI HAR from {UCI_URL} …")
    _download_with_resume(UCI_URL, outer_zip)
    with zipfile.ZipFile(outer_zip) as outer:
        inner_name = next((n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")), None)
        if inner_name is None:
            # 有些版本外層就是資料夾
            outer.extractall(raw_dir)
        else:
            inner_zip = raw_dir / "_uci_har_inner.zip"
            outer.extract(inner_name, raw_dir)
            extracted = raw_dir / inner_name
            extracted.rename(inner_zip)
            with zipfile.ZipFile(inner_zip) as inner:
                inner.extractall(raw_dir)
            inner_zip.unlink()
    outer_zip.unlink()
    assert (uci_root / "train").is_dir(), "unexpected archive layout"
    print(f"UCI HAR extracted to {uci_root}")
    return uci_root


def download_fitbit(raw_dir: Path) -> Path:
    """Fitbit 資料集（kagglehub）。公開資料集通常免帳號；失敗時給出手動指引."""
    try:
        import kagglehub
    except ImportError:
        print("請先 pip install kagglehub，或見 README 手動下載指引。")
        raise SystemExit(2) from None  # 已給使用者友善訊息，不必鏈結堆疊

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
            print(f"⚠ Fitbit 下載失敗（{e}）——UCI 管線不受影響；手動下載說明見 README。")
    return True


def download_pamap2(raw_dir: Path) -> Path:
    """下載 PAMAP2（HR＋IMU 同步量測，~680MB）；回傳 subject*.dat 所在目錄."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    hits = list(raw_dir.glob("**/subject101.dat"))
    if hits:
        print(f"PAMAP2 already present at {hits[0].parent}")
        return hits[0].parent

    inner_zip = raw_dir / "PAMAP2_Dataset.zip"
    if not inner_zip.is_file():
        outer_zip = raw_dir / "_pamap2_outer.zip"
        print(f"Downloading PAMAP2 from {PAMAP2_URL} …")
        _download_with_resume(PAMAP2_URL, outer_zip)
        with zipfile.ZipFile(outer_zip) as zf:
            zf.extractall(raw_dir)  # UCI 外層：內含 PAMAP2_Dataset.zip＋readme
        outer_zip.unlink()
    # 解內層 zip（subject*.dat 在這層）
    with zipfile.ZipFile(inner_zip) as inner:
        inner.extractall(raw_dir)
    protocol = raw_dir / "PAMAP2_Dataset" / "Protocol"
    assert protocol.is_dir(), "unexpected PAMAP2 archive layout"
    print(f"PAMAP2 extracted → {protocol}")
    return protocol
