"""
Основной скрипт для запуска инференса модели на видео.
"""

from pathlib import Path

from scripts.utils import config
from scripts.predict import (
    my_run,  # ! Этот скрипт написан для тестирования модели и оставлен на случае возникновения проблем с yolo_run
    yolo_run  # * Использовать этот скрипт если не возникают никакие проблемы
)


def run(use_yolo_run: bool = True):
    """
    Запускает процесс инференса модели YOLO на видео из директории.

    Args:
        use_yolo_run (bool, optional): Если True — использовать основной скрипт yolo_run.
                                       Если False — использовать вспомогательный my_run
    """
    video_dir = Path(config.VIDEO_DIR)
    video_paths = sorted([
        p for p in video_dir.glob("*")
        if p.is_file() and p.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]
    ])
    
    if use_yolo_run:
        yolo_run(
            models=[config.CURRENT_MODEL_PATH],
            video_paths=video_paths,
            output_dir=config.OUTPUTS_VIDEOS_DIR,
            save=True,
            save_txt=False,
            conf=0.25,
            imgsz=640,
        )
    else:
        my_run(
            models=[config.CURRENT_MODEL_PATH],
            video_paths=video_paths,
            frames_extracted_dir=config.FRAMES_EXTRACTED_DIR,
            frames_processed_dir=config.FRAMES_PROCESSED_DIR,
            output_dir=config.OUTPUTS_VIDEOS_DIR,
            fps="auto",
        )

    
if __name__ == "__main__":
    use_yolo_run = True  # * При необходимости можно заменить на False
    run(use_yolo_run=use_yolo_run)
