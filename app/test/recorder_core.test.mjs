/* recorder_core 邏輯測試（node app/test/recorder_core.test.mjs）.
 * 50Hz 格線、內插正確性、gap 切段、視窗計數、CSV 形狀——v2 資料品質的根基.
 */
import { createRequire } from "module";
import assert from "assert";

const require = createRequire(import.meta.url);
const { SessionRecorder, toCsv, RATE_HZ, WINDOW } = require("../recorder_core.js");

const B = (v) => ({ x: v, y: 0, z: 0 });

function feed(rec, timesMs, bxOf) {
  let n = 0;
  for (const t of timesMs) {
    if (rec.addSample(t, B(bxOf(t)), B(0))) n += 1;
  }
  return n;
}

// 1. 精確 50Hz 格線：20ms 均勻取樣 → 重採樣後長度一致、時間戳等距
{
  const rec = new SessionRecorder();
  const n = feed(rec, Array.from({ length: 100 }, (_, i) => i * 20), (t) => t);
  const out = rec.resample();
  assert.strictEqual(n, 100);
  assert.strictEqual(out.length, 100);
  const steps = out.slice(1).map((r, i) => r.t - out[i].t);
  assert.ok(steps.every((s) => s === 20), "grid steps are exactly 20ms");
  console.log("ok 1 · exact 50Hz grid");
}

// 2. 抖動時間戳的線性內插：bx 是時間的線性函數 → 格線上仍取到該函數值
{
  const rec = new SessionRecorder();
  const times = Array.from({ length: 60 }, (_, i) => i * 20 + (i % 2 ? 4 : -3));
  feed(rec, times, (t) => t / 10);
  const out = rec.resample();
  assert.strictEqual(out.length, 60);
  for (const r of out) {
    assert.ok(Math.abs(r.bx - r.t / 10) < 0.01,
      `interpolation at ${r.t}: ${r.bx} vs ${r.t / 10}`);
  }
  console.log("ok 2 · linear interpolation under jitter");
}

// 3. gap 切段：中間掉 1 秒 → 兩段、不跨 gap 內插
{
  const rec = new SessionRecorder();
  const t1 = Array.from({ length: 50 }, (_, i) => i * 20);           // 0-980ms
  const t2 = Array.from({ length: 50 }, (_, i) => 1420 + i * 20);    // gap 440ms > GAP_MS
  feed(rec, [...t1, ...t2], (t) => t);
  const segs = rec.segments();
  assert.strictEqual(segs.length, 2, "gap splits into two segments");
  const out = rec.resample();
  const segIds = [...new Set(out.map((r) => r.seg))];
  assert.deepStrictEqual(segIds, [1, 2]);
  // seg2 首格線 t ≥ 1420（從 seg2 原始起點重新起格，不內插越過 gap）
  const seg2First = out.find((r) => r.seg === 2);
  assert.ok(seg2First.t >= 1420 && seg2First.t < 1421, "seg2 grid restarts clean");
  console.log("ok 3 · gap segmentation, no interpolation across gaps");
}

// 4. null 樣本與時序亂序：計數丟棄、不進緩衝
{
  const rec = new SessionRecorder();
  assert.strictEqual(rec.addSample(0, B(1), B(0)), true);
  assert.strictEqual(rec.addSample(20, { x: NaN, y: 0, z: 0 }, B(0)), false);
  assert.strictEqual(rec.addSample(40, B(2), B(0)), true);
  assert.strictEqual(rec.addSample(10, B(1), B(0)), false);   // 亂序（< 40）
  assert.strictEqual(rec.nullDropped, 1);
  assert.strictEqual(rec.rejected, 1);
  assert.strictEqual(rec.raw.length, 2);
  console.log("ok 4 · null/out-of-order rejection");
}

// 5. 視窗計數：一個 300 樣本段 → 2 個 128 視窗；不足 128 的段 → 0
{
  const rec = new SessionRecorder({ window: WINDOW });
  feed(rec, Array.from({ length: 300 }, (_, i) => i * 20), (t) => t);
  const out = rec.resample();
  assert.strictEqual(rec.windowCount(out), 2);
  const rec2 = new SessionRecorder();
  feed(rec2, Array.from({ length: 100 }, (_, i) => i * 20), (t) => t);
  assert.strictEqual(rec2.windowCount(rec2.resample()), 0);
  console.log("ok 5 · window counting per segment");
}

// 6. CSV 形狀：標頭、t_rel_s 從 0 起、小數位穩定
{
  const rec = new SessionRecorder();
  feed(rec, Array.from({ length: 40 }, (_, i) => 1000 + i * 20), (t) => t);
  const csv = toCsv(rec.resample());
  const lines = csv.trim().split("\n");
  assert.strictEqual(lines[0], "t_rel_s,seg,bx,by,bz,tx,ty,tz");
  assert.strictEqual(lines[1].split(",")[0], "0.000");
  assert.strictEqual(lines.length, 41);
  assert.ok(Number.isFinite(parseFloat(lines[3].split(",")[2])));
  console.log("ok 6 · csv shape");
}

console.log(`\nall recorder_core tests passed (rate=${RATE_HZ}Hz window=${WINDOW})`);
