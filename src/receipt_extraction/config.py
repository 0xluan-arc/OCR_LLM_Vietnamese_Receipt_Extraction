"""Module quản lý cấu hình tập trung cho toàn bộ hệ thống."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional

# Xác định ROOT của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_dotenv_if_present() -> None:
    """Nạp các biến từ file .env tại gốc dự án vào os.environ (không ghi đè biến đã đặt sẵn)."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


_load_dotenv_if_present()


def get_env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class PathSettings:
    """Đường dẫn các thư mục chuẩn trong dự án và kho dữ liệu tập trung."""
    root_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    samples_dir: Path = PROJECT_ROOT / "data" / "samples"
    configs_dir: Path = PROJECT_ROOT / "configs"
    experiments_dir: Path = PROJECT_ROOT / "experiments"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    # Kho dữ liệu tập trung D:\NCKH\Data_HoaDon_NCKH
    external_data_hub_dir: Path = PROJECT_ROOT.parent / "Data_HoaDon_NCKH"
    mcocr_standard_dir: Path = PROJECT_ROOT.parent / "Data_HoaDon_NCKH" / "01_MC_OCR_2021_Chuan_CuocThi"
    self_captured_300_dir: Path = PROJECT_ROOT.parent / "Data_HoaDon_NCKH" / "02_HoaDon_ThucTe_TuChup_300Anh"
    synthetic_test_10_dir: Path = PROJECT_ROOT.parent / "Data_HoaDon_NCKH" / "03_HoaDon_MoPhong_KiemThu_10Mau"
    splits_and_cms_dir: Path = PROJECT_ROOT.parent / "Data_HoaDon_NCKH" / "04_DuLieu_DaChiaTap_Train_Val_Test"


@dataclass
class ImageValidationSettings:
    """Ngưỡng kiểm tra tính hợp lệ của ảnh đầu vào."""
    max_size_mb: float = field(default_factory=lambda: get_env_float("MAX_IMAGE_SIZE_MB", 15.0))
    min_width: int = field(default_factory=lambda: get_env_int("MIN_IMAGE_WIDTH", 200))
    min_height: int = field(default_factory=lambda: get_env_int("MIN_IMAGE_HEIGHT", 200))
    blur_threshold: float = field(default_factory=lambda: get_env_float("BLUR_THRESHOLD", 50.0))


@dataclass
class OCRSettings:
    """Cấu hình cho OCR engine (Mặc định: PaddleVietOCREngine 'paddleocr', fallback về 'tesseract')."""
    engine: str = field(default_factory=lambda: os.getenv("OCR_ENGINE", "paddleocr").lower())
    tesseract_cmd: str = field(
        default_factory=lambda: os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    )


@dataclass
class LLMSettings:
    """Cấu hình cho LLM Extractor."""
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "none").lower())
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:1.5b-instruct"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    timeout_sec: int = field(default_factory=lambda: get_env_int("LLM_TIMEOUT_SEC", 180))


@dataclass
class AppConfig:
    """Cấu hình tổng thể của ứng dụng."""
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    debug: bool = field(default_factory=lambda: get_env_bool("DEBUG", True))
    paths: PathSettings = field(default_factory=PathSettings)
    image: ImageValidationSettings = field(default_factory=ImageValidationSettings)
    ocr: OCRSettings = field(default_factory=OCRSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)


# Khởi tạo singleton config
config = AppConfig()
