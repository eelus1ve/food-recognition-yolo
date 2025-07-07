import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
CONFIG_PATH = ROOT_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# * Директории
DATA_DIR = ROOT_DIR / cfg["paths"]["data_dir"]
VIDEO_DIR = ROOT_DIR / cfg["paths"]["videos"]
FRAMES_EXTRACTED_DIR = ROOT_DIR / cfg["paths"]["frames_extracted"]
FRAMES_PROCESSED_DIR = ROOT_DIR / cfg["paths"]["frames_processed"]
OUTPUTS_VIDEOS_DIR = ROOT_DIR / cfg["paths"]["output_videos"]
MODELS_DIR = ROOT_DIR / cfg["paths"]["models_dir"]
TEST_DIR = ROOT_DIR / cfg["paths"]["test_dir"]


# * Пути
CURRENT_MODEL_PATH = ROOT_DIR / cfg["paths"]["current_model"]
DATA_YAML_PATH = ROOT_DIR / cfg["paths"]["dataset_yaml"]

# * Параметры YOLO
YOLO_CONFIG = cfg["yolo"]

# * FPS видео
FPS = cfg["video"]["fps"]
