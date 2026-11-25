/* ISA mobile recorder core · 純邏輯（無 DOM）——v2 手機自採的資料品質根基.
 *
 * 設計鐵律（README N1 / findings 位置對照實驗）：
 *   - 50Hz 精確重採樣（手機 devicemotion 事件率不保證 50Hz——錄原始時間戳，
 *     匯出時線性內插到精確 50Hz 格線，與 UCI/PAMAP2 特徵棧相容）
 *   - 128 樣本視窗（2.56s）——與現有 49 維特徵棧相容
 *   - gap 切段：事件流失 > gapMs 時切段、不跨 gap 內插（內插會偽造訊號）
 *
 * UMD-lite：<script src> 與 node require 皆可用（node 測試見 test/）.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ISARecorderCore = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const RATE_HZ = 50;
  const WINDOW = 128;     // 2.56s @50Hz
  const GAP_MS = 200;     // 10×步長：超過視為事件流失，切段

  class SessionRecorder {
    constructor({ rateHz = RATE_HZ, window = WINDOW, gapMs = GAP_MS } = {}) {
      this.rateHz = rateHz;
      this.window = window;
      this.gapMs = gapMs;
      this.raw = [];            // {t, bx,by,bz, tx,ty,tz}——原始時間戳（ms）
      this.nullDropped = 0;     // 感測器中途給 null（iOS 常見）→ 丟棄計數
      this.rejected = 0;        // 時序不單調 → 丟棄計數
      this.startedAt = null;
    }

    /** 回傳 true 表示已收錄。body/total 為 {x,y,z}（m/s²）. */
    addSample(tMs, body, total) {
      const ok = (v) => v && Number.isFinite(v.x) && Number.isFinite(v.y)
        && Number.isFinite(v.z);
      if (!ok(body) || !ok(total)) {
        this.nullDropped += 1;
        return false;
      }
      if (this.raw.length > 0 && tMs <= this.raw[this.raw.length - 1].t) {
        this.rejected += 1;
        return false;
      }
      if (this.startedAt === null) this.startedAt = tMs;
      this.raw.push({ t: tMs, bx: body.x, by: body.y, bz: body.z,
        tx: total.x, ty: total.y, tz: total.z });
      return true;
    }

    /** 依 gap 切段 → 原始陣列的 [start, end) 索引段列表（單一樣本的段拋棄）. */
    segments() {
      const segs = [];
      let s = 0;
      for (let i = 1; i <= this.raw.length; i++) {
        const isGap = i < this.raw.length
          && this.raw[i].t - this.raw[i - 1].t > this.gapMs;
        if (i === this.raw.length || isGap) {
          if (i - s > 1) segs.push([s, i]);
          s = i;
        }
      }
      return segs;
    }

    /** 線性內插重採樣到精確 rateHz 格線；回傳 {t,seg,bx,by,bz,tx,ty,tz}[]. */
    resample() {
      const step = 1000 / this.rateHz;
      const out = [];
      let segIdx = 0;
      for (const [s, e] of this.segments()) {
        segIdx += 1;
        const t0 = this.raw[s].t;
        const tEnd = this.raw[e - 1].t;
        const n = Math.floor((tEnd - t0) / step) + 1;
        let j = s;
        for (let k = 0; k < n; k++) {
          const g = t0 + k * step;
          while (j < e - 1 && this.raw[j + 1].t <= g) j++;
          const a = this.raw[j];
          const b = this.raw[Math.min(j + 1, e - 1)];
          const f = (g - a.t) / ((b.t - a.t) || 1);
          const lerp = (u, v) => u + (v - u) * f;
          out.push({ t: g, seg: segIdx,
            bx: lerp(a.bx, b.bx), by: lerp(a.by, b.by), bz: lerp(a.bz, b.bz),
            tx: lerp(a.tx, b.tx), ty: lerp(a.ty, b.ty), tz: lerp(a.tz, b.tz) });
        }
      }
      return out;
    }

    /** 128 樣本視窗總數（各段獨立取整——段尾不足 128 的拋棄）. */
    windowCount(resampled) {
      let count = 0;
      let seg = null;
      let n = 0;
      for (const r of resampled) {
        if (r.seg !== seg) {
          count += Math.floor(n / this.window);
          seg = r.seg;
          n = 0;
        }
        n += 1;
      }
      count += Math.floor(n / this.window);
      return count;
    }
  }

  /** resampled → CSV 字串（t_rel_s 相對首列；seg 欄供 pandas groupby）. */
  function toCsv(resampled) {
    const t0 = resampled.length ? resampled[0].t : 0;
    const lines = ["t_rel_s,seg,bx,by,bz,tx,ty,tz"];
    for (const r of resampled) {
      const rel = ((r.t - t0) / 1000).toFixed(3);
      const c = (v) => Number(v.toFixed(5));
      lines.push(`${rel},${r.seg},${c(r.bx)},${c(r.by)},${c(r.bz)},` +
        `${c(r.tx)},${c(r.ty)},${c(r.tz)}`);
    }
    return lines.join("\n") + "\n";
  }

  return { SessionRecorder, toCsv, RATE_HZ, WINDOW, GAP_MS };
});
