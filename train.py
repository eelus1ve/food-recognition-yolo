"""
Скрипт запуска обучения модели YOLO.
"""

from scripts.train.train_new_model import train_run
from scripts.utils.config import MODELS_DIR, DATA_YAML_PATH, TEST_DIR, YOLO_CONFIG


if __name__ == "__main__":
    train_run(
        model_path=MODELS_DIR / "default" / "yolo11n.pt",
        data_yaml_path=DATA_YAML_PATH,
        test_images_dir=TEST_DIR / "images",
        config=YOLO_CONFIG,
        start_test=True,
    )
