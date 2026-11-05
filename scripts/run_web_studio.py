r"""Kịch bản khởi chạy Giao diện Web Studio & FastAPI Backend Server
Hệ thống Trích xuất Hóa đơn Bán lẻ Việt Nam (Nested Learning - NeurIPS 2025).

Khởi chạy bằng lệnh:
    python scripts/run_web_studio.py --port 8000
Hoặc:
    uvicorn receipt_extraction.api.server:app --reload --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Đảm bảo đường dẫn import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Khởi chạy Vietnamese Receipt Document AI Studio (FastAPI).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Địa chỉ IP lắng nghe (mặc định 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Cổng dịch vụ (mặc định 8000).")
    parser.add_argument("--reload", action="store_true", default=False, help="Tự động tải lại mã nguồn khi sửa đổi.")
    args = parser.parse_args()

    import uvicorn

    print("=" * 80)
    print("KHỞI CHẠY VIETNAMESE RECEIPT DOCUMENT AI STUDIO (FASTAPI + MODERN WEB SPA)")
    print("=" * 80)
    print(f"  - Giao diện Web Studio : http://localhost:{args.port}")
    print(f"  - Tài liệu Swagger Docs: http://localhost:{args.port}/docs")
    print(f"  - Trạng thái Hệ thống  : http://localhost:{args.port}/api/status")
    print("=" * 80)
    print("Nhấn CTRL+C để dừng server.\n")

    uvicorn.run(
        "receipt_extraction.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
