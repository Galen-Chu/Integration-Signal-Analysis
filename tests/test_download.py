"""下載層回歸測試——本地 HTTP server 模擬 UCI archive 的中途斷線.

實際事故（CI 健康檢查首跑）：urlopen().read() 一把抓在 281MB 處
IncompleteRead，整趟作廢。這裡用可投擲「半截回應」的本地 server 釘住
續傳與重試行為.
"""

import http.server
import os
import threading

import pytest

import src.isa.data.download as dl
from src.isa.data.download import _download_with_resume

PAYLOAD = os.urandom((1 << 20) * 3 + 12345)  # 3MB+：跨多個 1MB chunk


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    monkeypatch.setattr(dl, "RETRY_BACKOFF_SECONDS", 0.05)


class _DroppingHandler(http.server.BaseHTTPRequestHandler):
    """支援 Range；drop_budget 次數內在整體回應的中途斷線（謊報 Content-Length）."""

    payload = PAYLOAD
    drop_budget = 0
    supports_range = True
    range_requests = []  # 類別層級：記錄伺服器看到的 Range 標頭

    def do_GET(self):  # http.server 介面要求 camelCase
        rng = self.headers.get("Range")
        type(self).range_requests.append(rng)  # 寫進實際子類，避免影子清單
        if rng and self.supports_range:
            start = int(rng.removeprefix("bytes=").split("-")[0])
            body = self.payload[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
            self.end_headers()
            self.wfile.write(body)
        else:
            body = self.payload
            dropping = type(self).drop_budget > 0
            if dropping:
                type(self).drop_budget -= 1
            self.send_response(200)
            # 只在斷線時謊報 Content-Length（誘發 IncompleteRead）；
            # 完整回應必須報真實長度，否則客戶端的短讀偵測永遠失敗
            self.send_header("Content-Length", str(len(body) + 1000 if dropping else len(body)))
            self.end_headers()
            if dropping:
                self.wfile.write(body[: len(body) // 2])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(body)

    def log_message(self, *args):  # 安靜
        pass


def _serve(handler_cls):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/data.zip"


def test_download_completes_without_interruption(tmp_path):
    handler = type("H", (_DroppingHandler,), {"drop_budget": 0})
    server, url = _serve(handler)
    try:
        dest = tmp_path / "data.zip"
        _download_with_resume(url, dest, attempts=2)
        assert dest.read_bytes() == PAYLOAD
    finally:
        server.shutdown()


def test_resume_after_midstream_drop(tmp_path):
    """斷線一次＋伺服器支援 Range → 從斷點續傳完成，內容完整."""
    handler = type(
        "H", (_DroppingHandler,), {"drop_budget": 1, "supports_range": True, "range_requests": []}
    )
    server, url = _serve(handler)
    try:
        dest = tmp_path / "data.zip"
        _download_with_resume(url, dest, attempts=3)
        assert dest.read_bytes() == PAYLOAD
        assert any(r for r in handler.range_requests if r), "resume must send Range header"
    finally:
        server.shutdown()


def test_restart_when_server_ignores_range(tmp_path):
    """斷線一次＋伺服器不支援 Range（回 200 整檔）→ 客戶端應整檔重下、不接出錯誤內容."""
    handler = type(
        "H", (_DroppingHandler,), {"drop_budget": 1, "supports_range": False, "range_requests": []}
    )
    server, url = _serve(handler)
    try:
        dest = tmp_path / "data.zip"
        _download_with_resume(url, dest, attempts=3)
        assert dest.read_bytes() == PAYLOAD
    finally:
        server.shutdown()
