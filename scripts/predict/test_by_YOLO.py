from ultralytics import YOLO
from pathlib import Path


def yolo_run(models: str | Path, video_paths: str | Path, output_dir: str | Path, **kwargs) -> list:
    """
    Загружает одну или несколько моделей YOLO и выполняет инференс на одном или нескольких видеофайлах.
    Результаты предсказаний сохраняются в отдельные папки в указанной директории вывода.

    Args:
        models (str | Path | list[str | Path]): Путь или список путей к .pt файлам моделей YOLO.
        video_paths (str | Path | list[str | Path]): Путь или список путей к видеофайлам для анализа.
        output_dir (str | Path): Путь к директории, куда будут сохранены результаты (создаётся, если не существует).
        **kwargs: Дополнительные аргументы, передаваемые в метод predict() модели YOLO (например, conf, imgsz, save, save_txt и др.).

    Returns:
        list: Список объектов результатов инференса для каждого сочетания модели и видео.
    
    Raises:
        FileNotFoundError: Если модель или видеофайл не найдены по указанному пути.
    """
    results = []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_path in models:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        model = YOLO(str(model_path))
        model_name = model_path.stem

        if not model_path.suffix:
            model_path = model_path.with_suffix(".pt")

        for video_path in video_paths:
            video_path = Path(video_path)
            video_filename = video_path.stem
            
            if not video_path.exists():
                raise FileNotFoundError(f"Видео не найдено: {video_path}")

            run_name = f"{video_filename}_{model_name}"

            results.append(model.predict(
                source=str(video_path),
                project=output_dir,
                name=run_name,
                **kwargs
            ))

    return results
