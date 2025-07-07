"""
! ВАЖНО
! Этот скрипт создан для локальной отладки работы модели.
! Используйте только при проблемах с основным скриптом: ./scripts/predict/test_by_YOLO.py
! Код не оптимизирован: медленный, много весит, захламляет диск.
"""

from pathlib import Path

from scripts.utils.videolib import extract_frames, frames_to_video, get_video_fps
from scripts.utils.model_on_frame import run_model_on_frames


def my_run(
    models: list[str | Path],
    video_paths: list[str | Path],
    frames_extracted_dir: str | Path,
    frames_processed_dir: str | Path,
    output_dir: str | Path,
    fps: int | str = "auto",
) -> None:
    """
    ! ВАЖНО !
    ---
    ! Этот скрипт создан для локальной отладки работы модели.
    ! Используйте только при проблемах с основным скриптом: ./scripts/predict/test_by_YOLO.py
    ! from scripts.predict import yolo_run
    ! Код не оптимизирован: медленный, много весит, захламляет диск.
    ---
    Обрабатывает список видеофайлов с помощью указанных моделей:
    - Извлекает кадры из видео.
    - Применяет модель к каждому кадру.
    - Сохраняет итоговое видео после обработки.

    Args:
        models: Пути к моделям
        video_paths: Пути к видеофайлам
        frames_extracted_dir: Директория для извлечённых кадров
        frames_processed_dir: Директория для обработанных кадров
        output_dir: Директория для финальных видео
        fps: int или "auto" — частота кадров
    """

    frames_extracted_dir = Path(frames_extracted_dir)
    frames_processed_dir = Path(frames_processed_dir)
    output_dir = Path(output_dir)

    frames_extracted_dir.mkdir(parents=True, exist_ok=True)
    frames_processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_path in models:
        model_path = Path(model_path)
        model_name = model_path.stem

        if not model_path.suffix:
            model_path = model_path.with_suffix(".pt")

        for video_path in video_paths:
            video_path = Path(video_path)
            video_filename = video_path.stem

            actual_fps = get_video_fps(video_path) if fps == "auto" else int(fps)

            output_video_path = output_dir / f"{video_filename}_{model_name}.mp4"

            extract_frames(str(video_path), str(frames_extracted_dir), fps=actual_fps)

            run_model_on_frames(
                model_path=str(model_path),
                input_folder=str(frames_extracted_dir),
                output_folder=str(frames_processed_dir),
            )

            frames_to_video(
                frames_folder=str(frames_processed_dir),
                output_video_path=str(output_video_path),
                fps=actual_fps,
            )
