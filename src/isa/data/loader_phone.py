"""Phone-recording loader · v2 自採 CSV → UCI 相容視窗（N1 資料落地入口）.

資料契約（app/README.md）：錄製頁匯出
`<subject>_<activity>_<position>_<stamp>.csv` ＋同名 `.meta.json`。
CSV 欄位 `t_rel_s,seg,bx,by,bz,tx,ty,tz`（50Hz 精確格線、m/s²）：
bx-bz＝body_acc（已除重力）、tx-tz＝total_acc（含重力）——直接對應 UCI
布局的 0-2/6-8 軸；`seg` 為 gap 切段編號（錄製頁保證不跨 gap 內插），
視窗只從段內切、段尾不足 window 樣本丟棄——與 loader_pamap2 同一慣例.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FS_PHONE = 50
WINDOW = 128  # 2.56s @50Hz——與 UCI/PAMAP2 特徵棧相同
STRIDE = 64  # 預設 50% 重疊（訓練增量）；合約測試用 stride=WINDOW 對齊錄製頁計數

CSV_COLUMNS = ["t_rel_s", "seg", "bx", "by", "bz", "tx", "ty", "tz"]


def load_phone_csv(csv_path) -> tuple[np.ndarray, np.ndarray]:
    """單一 session CSV → (rows, 6) float32 ＋ seg id 陣列.

    驗證：欄位順序、有限值、段內時間步 ≈ 0.02s（±1ms 容差）.
    """
    df = pd.read_csv(csv_path)
    assert list(df.columns) == CSV_COLUMNS, f"unexpected columns: {list(df.columns)}"
    vals = df[["bx", "by", "bz", "tx", "ty", "tz"]].to_numpy(dtype=np.float32)
    segs = df["seg"].to_numpy()
    assert np.isfinite(vals).all(), "NaN/inf in recording CSV"
    dt = np.diff(df["t_rel_s"].to_numpy())
    within = dt[segs[1:] == segs[:-1]]
    assert within.size == 0 or np.abs(within - 1 / FS_PHONE).max() < 0.001, (
        f"50Hz grid violated inside a segment: max|Δt-0.02| = "
        f"{np.abs(within - 1 / FS_PHONE).max():.4f}s"
    )
    return vals, segs


def windows_from_rows(
    vals: np.ndarray, segs: np.ndarray, window: int = WINDOW, stride: int = STRIDE
) -> np.ndarray:
    """逐段切 (9, window) 視窗：0-2 body、3-5 gyro 佔位 0、6-8 total."""
    out = []
    for seg in np.unique(segs):
        m = vals[segs == seg]
        for s in range(0, len(m) - window + 1, stride):
            w = np.zeros((9, window), dtype=np.float32)
            w[0:3, :] = m[s : s + window, 0:3].T
            w[6:9, :] = m[s : s + window, 3:6].T
            out.append(w)
    if not out:
        return np.zeros((0, 9, window), dtype=np.float32)
    return np.stack(out)


def load_phone_session(
    csv_path, meta_path=None, window: int = WINDOW, stride: int = STRIDE
) -> tuple:
    """單一 session → (windows, label, subject, meta).

    meta 缺席不擋載入（label/subject 回 "unknown"）——但正式資料集一律要有.
    """
    csv_path = Path(csv_path)
    meta_path = Path(meta_path) if meta_path else csv_path.with_name(f"{csv_path.stem}.meta.json")
    vals, segs = load_phone_csv(csv_path)
    windows = windows_from_rows(vals, segs, window=window, stride=stride)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta and meta.get("nominal_rate_hz") not in (None, FS_PHONE):
        raise ValueError(f"{csv_path.name}: meta rate {meta['nominal_rate_hz']} ≠ {FS_PHONE}")
    return windows, meta.get("activity", "unknown"), meta.get("subject", "unknown"), meta


def load_phone_dataset(phone_dir, window: int = WINDOW, stride: int = STRIDE) -> tuple:
    """整個 phone_v2 目錄 → (X (N, 9, window), y 活動名, groups 受試者).

    掃描 `*.csv`＋同名 `.meta.json`；回傳 label 為活動名稱字串、groups 為
    受試者編號字串（GroupKFold 直接可用，np.unique 排序後映射整數即可）.
    """
    records = sorted(Path(phone_dir).glob("*.csv"))
    if not records:
        raise FileNotFoundError(f"no *.csv under {phone_dir} — 錄製頁匯出後回存此處")
    all_w, labels, groups = [], [], []
    for csv_path in records:
        windows, label, subject, _ = load_phone_session(csv_path, window=window, stride=stride)
        if len(windows) == 0:
            print(f"  {csv_path.name}: 0 windows（段皆不足 {window} 樣本），skip")
            continue
        all_w.append(windows)
        labels += [label] * len(windows)
        groups += [subject] * len(windows)
        print(f"  {csv_path.name}: {len(windows)} windows ({label}, {subject})")
    return np.concatenate(all_w), np.asarray(labels), np.asarray(groups)
