# Findings · 動作識別×生理負荷——分析發現（I5）

> Integration-Signal-Analysis · 2026-10-29
> 問題：穿戴裝置能不能「看懂」你在做什麼動作（HAR），並量化這個動作讓身體付出多少（心率負荷）？

## TL;DR

1. **動作可以誠實地被辨識到 ~80%**：RF 80.4%±3.6%、SVM 80.5%±3.3%（macro-F1 ~0.81）
   ——前提是 subject-wise 驗證。隨機切分會把同一人的資料漏進測試集，文獻常見的
   90%+ 是被受試者洩漏撐高的數字。
2. **辨識的瓶頸與生理的瓶頸是同一個地方**：SITTING 召回率僅 57%，主因是與
   STANDING 互混——PAMAP2 的同步心率顯示，坐（42%HRmax）與站（47%HRmax）在
   **動作量與心率上都幾乎無差別**，模型混淆其實是「訊號本身無區別力」的忠實反映。
3. **日常動作的隱形負荷**：吸地板（55%HRmax）已接近走路（59%）、北歐式健走與
   樓梯（65-68%）直逼中強度區——日常累積負荷容易被低估。
4. **跳繙是量測中最重的動作**（85%HRmax，高於跑步的 82%）——高強度間歇（HIIT）
   與舞蹈體能訓練的負荷分層有了實證排序。

## 一、HAR 分類（UCI HAR，30 人 × 6 活動）

**方法**：自製特徵（時域統計 27 維＋FFT 頻域 16 維）→ RF / scaled-RBF-SVM →
**GroupKFold 5 折（受試者零重疊，每折內建斷言）**。官方 561 特徵對照組見
`--features both`。

| 模型 | Accuracy（subject-wise） | Macro-F1 | 最強 / 最弱召回 |
|---|---|---|---|
| Random Forest | 80.4% ± 3.6% | 0.809 | DOWNSTAIRS 90% / **SITTING 57%** |
| SVM (RBF, scaled) | 80.5% ± 3.3% | 0.813 | WALKING 91% / **SITTING 58%** |

**為什麼誠實的數字比較低**：隨機切分下，同一受試者的相似動作段會同時出現在
訓練與測試——模型學到的是「這個人的走路風格」而不是「走路的通用模式」。
部署到新使用者時，subject-wise 數字才是可預期的表現。**這是穿戴裝置產品
的實際情境：模型永遠要面對沒見過的使用者。**

## 二、動作 × 心率負荷（PAMAP2，9 人 × 12 活動 × HR/IMU 同步）

每活動平均 %HRmax（HRmax 以 190bpm 估計，受試者 24-32 歲）：

| 活動 | %HRmax | 心率 bpm | 分區 |
|---|---|---|---|
| lying | 39.8 | 76 | rest |
| sitting | 42.1 | 80 | rest |
| standing | 46.6 | 89 | rest |
| ironing | 47.4 | 90 | rest |
| vacuum_cleaning | 54.8 | 104 | light |
| walking | 59.4 | 113 | light 邊界 |
| nordic_walking | 65.2 | 124 | moderate |
| cycling | 65.7 | 125 | moderate |
| descending_stairs | 67.9 | 129 | moderate |
| ascending_stairs | 68.2 | 130 | moderate |
| running | 82.3 | 157 | vigorous |
| **rope_jumping** | **85.2** | **162** | **vigorous/near-max** |

（完整數據：`data/processed/activity_hr_load.json`；圖：
`reports/figures/activity_hr_load_ladder.png`）

### 生醫×舞蹈觀點

- **靜態三兄弟（躺/坐/站）在 40-47% 緊密聚集**——低動作量×低心率差，同時解釋了
  HAR 的 SITTING↔STANDING 混淆。改善方向不在分類器，而在**姿態轉換事件**
  （sit→stand 的暫態）特徵。
- **家務被低估**：吸地 55% 已達 light 上緣、逼近 walking——對久坐族群的
  「非運動性熱量消耗」（NEAT）量化，穿戴裝置比自陳問卷誠實。
- **跳繙 > 跑步**（85% vs 82%）：舞蹈/格鬥體能常用的跳繩是每單位時間負荷最高
  的入門動作——HIIT 課表設計的實證排序基礎。
- **樓梯上下差異極小**（68.2 vs 67.9）但加速度劇烈（暫態）——離心/向心肌群
  差異心率看不出來，需要肌電（EMG）才能分——本分析的天花板誠實標記。

## 三、限制（明說，不藏）

1. PAMAP2 僅 9 人（subject109 僅 65 秒可用）；%HRmax 用年齡上限公式估計，
   未用個人實測 HRmax。
2. UCI HAR（分類）與 PAMAP2（負荷）是不同群受試者——分類模型未跨資料集遷移
   驗證；負荷梯是 PAMAP2 內部結論。兩者的整合是**框架層級**：
   「HAR 分類結果 × 本負荷梯」即可對任何新資料做負荷估算。
3. 心率為秒級平均（PAMAP2 原始 ~9Hz 缺值已前向填充）；HRV（RR 間期）分析
   留給 v2（需原始 ECG/PPG）。
4. 手部單點加速度（±6g，m/s²）；跳舞等大幅度全身動作的腰部/踝部訊號表現
   可能不同——v2 自採資料驗證。

## 四、可重現

```bash
python scripts/fetch_data.py --uci          # UCI HAR
python -c "from src.isa.data.download import download_pamap2 as f; f()"  # PAMAP2
python scripts/run_pipeline.py              # 分類基準（subject-wise CV）
python scripts/run_integration.py           # 動作×心率負荷梯
python -m pytest                            # 33 項測試
```

## 五、v2 藍圖

手機感測器自採舞蹈核心動作/瑜珈姿勢（舞蹈背景優勢）→ 以本管線遷移學習；
HRV（RR-interval）疲勞監測接智慧手錶——「動作識別＋疲勞預警」的完整閉環。
