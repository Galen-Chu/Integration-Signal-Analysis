# 📡 Integration-Signal-Analysis

> **Human Activity Recognition × Physiological Signals · 動作識別×生理訊號整合分析**

[![CI](https://github.com/Galen-Chu/Integration-Signal-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Galen-Chu/Integration-Signal-Analysis/actions/workflows/ci.yml)
[![Data source health check](https://github.com/Galen-Chu/Integration-Signal-Analysis/actions/workflows/health-check.yml/badge.svg)](https://github.com/Galen-Chu/Integration-Signal-Analysis/actions/workflows/health-check.yml)
![tests](https://img.shields.io/badge/tests-72%20passing-2a78d6)

Using wearable sensor data (3-axis accelerometer + gyroscope) to recognize
body movements, then layering heart-rate physiology to quantify each
movement's **load** — how hard the body works, and how well it recovers.

穿戴式裝置感測器資料辨識身體動作，疊加心率生理訊號量化每個動作的「身體負荷」。

---

## 🧭 Design Principles · 設計原則

1. **Subject-wise validation is non-negotiable.** Random splits leak
   subjects between train/test and inflate accuracy — every experiment
   here uses `GroupKFold(groups=subject)`. 受試者不重疊的交叉驗證，結果才可信。
2. **Two feature tracks.** Hand-engineered time+frequency features (ours)
   vs. the dataset's official 561-feature set (control) — the comparison
   is itself a finding. 自製特徵與官方特徵對照。
3. **Integration with real simultaneous signals.** Kaggle Fitbit now
   requires login (403), so the integration dataset pivoted to **PAMAP2**
   — heart rate and IMUs recorded on the *same subjects at the same
   time*, enabling a true per-subject activity×load ladder (the original
   two-dataset idea remains documented as methodology-level fallback).
   整合用同步訊號：同一群人的動作與心率。
4. **Reproducible pipeline, narrative notebooks.** `src/` package +
   pytest for engineering rigor; `notebooks/` + `reports/findings.md`
   tell the story. 工程與敘事並重。

## 🏗️ Architecture · 架構

```
data/raw/UCI HAR Dataset   30 subjects × 6 activities × 50Hz (9 signals)
data/raw/PAMAP2_Dataset    9 subjects × 18 activities × 100Hz — **HR＋IMU 同步**
                           （整合分析主力：真正的「每人動作×心率負荷」）
        │
src/isa/data/        download（斷點續傳＋重試）+ loader（測試用迷你 fixture）
src/isa/features/    time_domain（mean/std/MAD/SMA/energy/軸相關）
                     freq_domain（主頻/頻譜質心/1-3Hz 步頻帶/頻譜熵）
                     posture（重力向量＋軸夾角——43→49 維躍進的 6 維）
                     transient（姿態轉換暫態 5 維——預註冊負結果）
src/isa/models/      RF / SVM / XGBoost × GroupKFold 5 折（防受試者洩漏）
src/isa/physio/      HR 強度分區、心率恢復 HRR、每日負荷加權
src/isa/viz/         混淆矩陣、頻譜、心率時間軸、IMU 位置對照圖
        │
reports/findings.md  最終敘事（面試故事）; cheatsheet.md 一頁速查; figures/
app/recorder.html    v2 手機錄製頁（Pages 已部署；資料契約見 app/README.md）
```

## 🚀 Quick Start · 快速開始

```bash
pip install -e .                      # 可編輯安裝（含依賴）；或 pip install -r requirements.txt
python scripts/fetch_data.py --uci    # UCI HAR（~60MB，斷點續傳）；Fitbit 需手動下載（見下）
python -c "from src.isa.config import RAW_DIR; from src.isa.data.download import download_pamap2; download_pamap2(RAW_DIR)"   # PAMAP2（~680MB，續傳＋重試）
python scripts/run_pipeline.py        # 自製特徵 × RF/SVM × subject-wise CV
python scripts/run_pipeline.py --features both   # 與官方 561 特徵對照
python -m pytest                      # 72 項測試（合成資料，免下載）
```

> Fitbit 下載若失敗（kagglehub 需要 cookie 或被限流）：到
> [Kaggle: FitBit Fitness Tracker Data](https://www.kaggle.com/datasets/mobius/fitbit-fitness-tracker-data)
> 手動下載，解壓到 `data/raw/fitbit/`（含 `heartrate_seconds_merged.csv`）即可。

## 🗺️ Roadmap · 里程碑（雙資料集同步版）

- [x] **I1 資料層**：UCI 下載/載入（迷你 fixture 測試）、Fitbit 心率載入
- [x] **I2 特徵工程**：時域統計＋FFT 頻域（主頻/質心/步頻帶/熵）＋合成訊號精確值測試
- [x] **I3 模型基準**：RF/SVM×雙特徵軌×雙評估協議——自製 43 維 80-83%、官方 561 維 92-95%、官方 split SVM 95.5% 重現原始論文 ~96%（管線驗證）；詳見 [findings](reports/findings.md)
- [x] **I4 生理整合**：PAMAP2 動作×%HRmax 負荷梯（跳繙 85% > 跑步 82%；家務 55% 逼近走路）——`scripts/run_integration.py`
- [x] **I5 發表**：[reports/findings.md](reports/findings.md)——誠實驗證 vs 洩漏虛高、瓶頸的生理解釋、負荷梯、限制明列
- [x] **v2 前哨實驗**（2026-11-06）：姿態幾何 6 特徵——49 維官方 split SVM **90.4%**、SITTING 召回 54→80%（假說閉環）；PAMAP2 12 類分類複現 72-74%，靜態混淆跨資料集重現（手部感測器歸因）
- [x] **v2 前哨 2·IMU 位置對照**（2026-11-25）：PAMAP2 手/胸/踝三 IMU 同受試者受控對照（±6g 顆粒，±16g 有已知故障通道）——總體準確率 69-75% 對位置不敏感，**靜態召回結構對位置極敏感**：躺姿踝部 93% 最優、坐/站腕部 74/74% 最優（踝部物理上不可分 47/47%）——「位置>特徵工程」假說受控證實，v2 錄製頁位置指引定為腰/胸主、腕次；見 [findings](reports/findings.md)
- [x] **特徵負結果 2·暫態**（2026-11-25）：+5 維姿態轉換暫態特徵未達預設門檻（SVM SITTING 召回 79.6→78.4%）——「改善方向在轉態特徵」的舊推測正式否證，殘餘 5pp 再次收窄至靜態幾何工程廣度；49 維維持最適；見 [findings](reports/findings.md)
- [x] **工程強化批次**（2026-11-25）：ruff format 全庫套用＋CI 加 `format --check`；**端到端煙霧測試**（合成資料走 特徵→模型→評估→全視覺化 函式，69 項測試）；`pip install -e .` 可安裝（pyproject `[project]` 元資料＋套件宣告）
- [x] **工程強化 2**（2026-11-25）：GitHub Pages 部署錄製頁（`app/**` 有變更自動重部署，真機 https 就緒）；**每月資料源健康檢查**（實際下載 UCI/PAMAP2 並斷言可載入、形狀正確——合成資料測試無法察覺上游改版）；Dependabot（pip＋Actions 週期檢查）；[面試速查卡](reports/cheatsheet.md)（五個數字故事＋反問彈藥）

**下次開發 · Next（2026-11-06 收錄，追蹤於 Issue #2）**

- [ ] **N1 手機自採錄製器**（優先序 1·v2 主線）：**工程完成＋已上線（2026-11-25）**——`app/recorder.html`：DeviceMotion 錄製頁（iOS 權限手勢流程、50Hz 精確重採樣、gap 切段不內插、128 視窗計數、腰/胸/腕位置指引），核心邏輯有 node 測試進 CI；資料格式契約見 [app/README](app/README.md)。**正式網址**：<https://galen-chu.github.io/Integration-Signal-Analysis/recorder.html>——**待 Galen 真機採集**（10 秒試拍 → 檢查 CSV/meta → 正式錄，每活動 ≥60 秒）
- [x] **N2 選擇性特徵移植**（負結果·2026-11-10）：ECDF/頻帶能量/jerk 單一家族皆買不到剩餘 5pp（jerk +0.4pp 代價 43 維）——49 維＝精簡集最適解，剩餘差距屬官方 17 家族工程廣度；見 findings N2 節
- [x] **N3 Notebooks 01/02/04 填實**（2026-11-25）：EDA（訊號長相/步頻指紋/受試者變異）、特徵與模型戰績（49 維故事的真實數字重現）、生理整合（分區/HRR/每日負荷，Fitbit 缺席時自動以 PAMAP2 Protocol 場次示範）——已執行存出輸出；03 的 Protocol 路徑地雷（glob 誤抓 Optional 場次）一併修正
- [x] **N4 小項批次**（2026-11-10）：CI 加 ruff、每受試者準確率分布圖、XGBoost 第三模型對照（561 維 94.2% 仍不敵 SVM 95.5%——瓶頸在特徵不在模型）
- [ ] **N5 HRV（RR-interval）**（遠程）：接智慧手錶原始資料，疲勞監測升級

## 📄 Data Sources · 資料來源

- [UCI HAR Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
  （Anguita et al., 2013 — 30 subjects, 6 activities, 50Hz）
- [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
  （Reiss & Stricker, 2012 — 9 subjects, 18 activities, HR＋3 IMU @100Hz）
- [Kaggle FitBit Fitness Tracker Data](https://www.kaggle.com/datasets/mobius/fitbit-fitness-tracker-data)
  （kagglehub 已需登入同意——保留手動下載路徑，physio 模組可直接使用）

## 📄 License

MIT — see [LICENSE](LICENSE).
