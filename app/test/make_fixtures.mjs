/* 產生 v2 錄製契約的 fixture（CSV＋meta.json）——python loader 的跨語言測試輸入.
 *
 * 用真實的 recorder_core（抖動時間戳＋gap 切段＋50Hz 重採樣）產生，
 * 所以這份 fixture 就是錄製頁會吐出的格式本身。
 * 用法：node app/test/make_fixtures.mjs <outdir>   （stdout 印 JSON 摘要）
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { SessionRecorder, toCsv, RATE_HZ, WINDOW } = require(
  path.join(here, "..", "recorder_core.js"),
);

const outdir = process.argv[2];
if (!outdir) {
  console.error("usage: node make_fixtures.mjs <outdir>");
  process.exit(2);
}
fs.mkdirSync(outdir, { recursive: true });

const T0 = 1_000_000; // 假的起始 ms

/** jitterMs: 事件時間戳抖動幅度；gapAtMs/gapMs: 中途事件流失模擬 */
function recordSession({ seconds, jitterMs = 0, gapAtMs = null, gapMs = 0 }) {
  const rec = new SessionRecorder();
  const total = seconds * RATE_HZ;
  for (let i = 0; i < total; i++) {
    let t = T0 + i * (1000 / RATE_HZ) + (jitterMs ? (i % 2 ? jitterMs : -jitterMs) : 0);
    if (gapAtMs !== null && t > T0 + gapAtMs && t <= T0 + gapAtMs + gapMs) continue;
    // 合成動作：body = 0.4·sin(2π·2Hz)，total = body＋重力 (0,0,9.81)
    const bx = 0.4 * Math.sin((2 * Math.PI * 2 * (t - T0)) / 1000);
    rec.addSample(t, { x: bx, y: 0.02, z: 0.02 }, { x: bx, y: 0.02, z: 9.83 });
  }
  const resampled = rec.resample();
  return { rec, resampled };
}

const sessions = [
  { base: "S99_dance_choreo_chest", subject: "S99", activity: "dance_choreo",
    position: "chest", seconds: 10, jitterMs: 4, gapAtMs: 4200, gapMs: 600 },
  { base: "S99_yoga_pose_wrist", subject: "S99", activity: "yoga_pose",
    position: "wrist", seconds: 8, jitterMs: 0, gapAtMs: null, gapMs: 0 },
];

const summary = [];
for (const s of sessions) {
  const { rec, resampled } = recordSession(s);
  const windows = rec.windowCount(resampled);
  const meta = {
    subject: s.subject,
    activity: s.activity,
    position: s.position,
    nominal_rate_hz: RATE_HZ,
    window_samples: WINDOW,
    device: "fixture-generator",
    started_at: new Date(T0).toISOString(),
    raw_samples: rec.raw.length,
    resampled_rows: resampled.length,
    segments: new Set(resampled.map((r) => r.seg)).size,
    windows_128: windows,
    null_dropped: rec.nullDropped,
    out_of_order_rejected: rec.rejected,
    format: "t_rel_s,seg,bx,by,bz,tx,ty,tz；bx-bz=body_acc(除重力)，tx-tz=total_acc(含重力)，m/s²，50Hz 線性內插重採樣",
  };
  fs.writeFileSync(path.join(outdir, `${s.base}.csv`), toCsv(resampled));
  fs.writeFileSync(path.join(outdir, `${s.base}.meta.json`),
    JSON.stringify(meta, null, 2));
  summary.push({ base: s.base, resampled_rows: resampled.length,
    segments: meta.segments, windows_128: windows });
}
console.log(JSON.stringify(summary));
