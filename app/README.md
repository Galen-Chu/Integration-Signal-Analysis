# ISA 手機訊號錄製器 · v2 自採工程

瀏覽器 DeviceMotion 錄製頁——v2「舞蹈/瑜珈/HIIT 自採＋遷移學習」的採集
工具（N1 的工程部分；資料採集需 Galen 親自執行）。

## 快速開始（真機）

1. **用 HTTPS 開啟本頁**——感測器 API 需要 secure context（`http://` 的
   區網 IP 在 iOS/Android 上拿不到動作事件）：
   - **正式網址（Pages 已部署）**：
     `https://galen-chu.github.io/Integration-Signal-Analysis/recorder.html`
     （Pages workflow 在 push 到 main、`app/**` 有變更時自動重新部署；
       artifact 根目錄即 `app/`，所以 URL 不帶 `/app`）
   - 本機驗證：`python -m http.server` + `http://localhost:8000/app/recorder.html`
     （`localhost` 視為 secure context，桌機可驗 UI、真機感測器仍需 https）
2. 按「啟用感測器」——**iOS 13+ 會彈權限詢問**，必須在這次點擊內完成
   （`DeviceMotionEvent.requestPermission()` 的瀏覽器限制，本頁已照做）
3. 選位置（腰/胸為主）、填受試者編號與活動名，「開始錄製」
4. 單一活動連續錄 **≥60 秒**（≈23 個 128 視窗）；換活動前按「停止並匯出」
5. 匯出兩個檔：`S01_<活動>_<位置>_<時間>.csv` ＋同名 `.meta.json`

**收集日流程**：每按一次「停止並匯出」，該 session 會加入頁面下方的
清單（活動/位置/視窗數一覽）並自動下載——換活動只要改名再按開始。
清單支援逐列「重新匯出」與「全部重新匯出」（0.9 秒間隔避開瀏覽器的
多檔下載限制），離開頁面前若清單非空會攔截提醒；「清除清單」只清
本頁記錄，已下載的檔案不受影響。

## 設計鐵律（與管線相容的根據）

| 鐵律 | 來源 |
|---|---|
| 位置：腰/胸優先、腕為次選 | UCI HAR（腰側）+ PAMAP2 位置對照實驗（findings 第一節：坐/站腕部 74%、踝部物理不可分） |
| 50Hz 精確重採樣 | UCI HAR 取樣率；手機 devicemotion 事件率**不保證** 50Hz（Android 50-100Hz 不等、iOS ~60Hz）——頁面錄原始時間戳，匯出時線性內插到精確 50Hz 格線 |
| 128 樣本視窗（2.56s） | 現有 49 維特徵棧的視窗長度（`src/isa/config.WINDOW_SIZE`） |

## 資料格式（`isa` 管線的 v2 輸入契約）

```
t_rel_s,seg,bx,by,bz,tx,ty,tz      # CSV，單位 m/s²
```

- `bx-bz`＝body_acc（`event.acceleration`，已除重力）；`tx-tz`＝total_acc
  （`event.accelerationIncludingGravity`，含重力）——對應 UCI 布局的
  0-2/6-8 軸，`src/isa/features/` 全部特徵直接可用
- `seg`＝切段編號：事件流失 >200ms 切段、**不跨 gap 內插**（內插會偽造
  訊號）；視窗只從每段內切，段尾不足 128 樣本拋棄
- `.meta.json`：受試者/活動/位置/裝置 UA/樣本數/視窗數/丟棄計數——
  未來 v2 loader（`load_phone_csv`）據此建立 labels 與 groups

## 可行性 spike 結論（2026-11-25 調研＋實作）

1. **iOS 權限**：`DeviceMotionEvent.requestPermission()` 存在且必須由
   使用者手勢觸發——本頁的「啟用感測器」按鈕即為此設計；拒絕後要到
   設定 → Safari 重新給權。
2. **Secure context**：Chrome 系要求 https；GitHub Pages 是零成本解。
   iOS Safari 對 devicemotion 在 http 上的行為歷來寬鬆，但以 https 為準
   最穩。
3. **事件率不穩**：裝置/瀏覽器/省電模式都會影響——頁面即時顯示事件率，
   錄製時應看到 ~50-60Hz；低於 40Hz 的段落會被 gap 邏輯保護。
4. **鎖屏即斷流**：開始錄製時申請 Screen Wake Lock（同樣要 https），
   不可用時 UI 提醒保持螢幕開啟；切到背景會觸發警告訊息。
5. **資料只在本機**：頁面無任何上傳——匯出靠瀏覽器下載，再以 AirDrop/
   傳輸線/雲端硬碟回存 `data/raw/phone_v2/`（路徑約定）。

## 工程測試

```bash
node app/test/recorder_core.test.mjs   # 50Hz 格線/內插/gap 切段/視窗/CSV
```

CI 已含本測試（`recorder-logic` job）。UI 本體需真機驗證——首次採集
流程：先錄 10 秒試拍 → 匯出檢查 CSV 列數與 `.meta.json` → 正式採集。

## 資料入口（已就緒，2026-11-26）

- [x] `src/isa/data/loader_phone.py`：讀 phone_v2 CSV＋meta → (N,9,128) 視窗
      （body/total 直取、seg 內切窗、49 維特徵全相容）——
      `load_phone_dataset("data/raw/phone_v2")` 即得 X/y/groups；
      **跨語言合約測試**進 CI（node recorder_core 產生真錄製格式 fixture →
      python loader 消費，視窗數逐檔相等）
- [ ] 與 UCI/PAMAP2 的遷移學習基準（原 49 維 vs fine-tune）——跨資料集
      零樣本已實測為 chance 級（18.9%，findings 第一節），warm-start 是否
      加速適配等真機資料到位後測試
