# 📡 Integration-Signal-Analysis

> **Human Activity Recognition × Physiological Signals · 動作識別×生理訊號整合分析**

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
src/isa/data/        download + loader（可測試：測試用迷你 fixture）
src/isa/features/    time_domain（mean/std/MAD/SMA/energy/軸相關）
                     freq_domain（主頻/頻譜質心/1-3Hz 步頻帶/頻譜熵）
src/isa/models/      RF / SVM（scaled pipeline）× GroupKFold 5 折
src/isa/physio/      HR 強度分區、心率恢復 HRR、每日負荷加權
src/isa/viz/         混淆矩陣、頻譜、心率時間軸
        │
reports/findings.md  最終敘事（面試故事）; figures/
```

## 🚀 Quick Start · 快速開始

```bash
pip install -r requirements.txt
python scripts/fetch_data.py --all    # UCI HAR (~60MB) + Fitbit（kagglehub 公開集免帳號）
python scripts/run_pipeline.py        # 自製特徵 × RF/SVM × subject-wise CV
python scripts/run_pipeline.py --features both   # 與官方 561 特徵對照
python -m pytest                      # 測試（合成資料，免下載）
```

> Fitbit 下載若失敗（kagglehub 需要 cookie 或被限流）：到
> [Kaggle: FitBit Fitness Tracker Data](https://www.kaggle.com/datasets/mobius/fitbit-fitness-tracker-data)
> 手動下載，解壓到 `data/raw/fitbit/`（含 `heartrate_seconds_merged.csv`）即可。

## 🗺️ Roadmap · 里程碑（雙資料集同步版）

- [x] **I1 資料層**：UCI 下載/載入（迷你 fixture 測試）、Fitbit 心率載入
- [x] **I2 特徵工程**：時域統計＋FFT 頻域（主頻/質心/步頻帶/熵）＋合成訊號精確值測試
- [x] **I3 模型基準**：RF/SVM＋subject-wise GroupKFold＋混淆矩陣
- [x] **I4 生理整合**：PAMAP2 動作×%HRmax 負荷梯（跳繙 85% > 跑步 82%；家務 55% 逼近走路）——`scripts/run_integration.py`
- [x] **I5 發表**：[reports/findings.md](reports/findings.md)——誠實驗證 vs 洩漏虛高、瓶頸的生理解釋、負荷梯、限制明列
- [ ] **v2 客製動作**：手機感測器自採舞蹈/瑜珈/HIIT 動作（優勢：舞蹈背景）
- [ ] **v2 實機端**：HRV（RR-interval）分析，接智慧手錶

## 📄 Data Sources · 資料來源

- [UCI HAR Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
  （Anguita et al., 2013 — 30 subjects, 6 activities, 50Hz）
- [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
  （Reiss & Stricker, 2012 — 9 subjects, 18 activities, HR＋3 IMU @100Hz）
- [Kaggle FitBit Fitness Tracker Data](https://www.kaggle.com/datasets/mobius/fitbit-fitness-tracker-data)
  （kagglehub 已需登入同意——保留手動下載路徑，physio 模組可直接使用）

## 📄 License

MIT — see [LICENSE](LICENSE).
