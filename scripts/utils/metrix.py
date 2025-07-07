from ultralytics import YOLO
from pathlib import Path


def metrix_run(model_path: str | Path, data_path: str | Path) -> None:
    """
    Выполняет валидацию модели YOLO на указанном датасете и выводит основные метрики качества.

    Args:
        model_path (str | Path): Путь к файлу с предобученной моделью YOLO (.pt).
        data_path (str | Path): Путь к YAML-файлу с описанием датасета (структура и пути к данным).

    Raises:
        FileNotFoundError: Если модель или файл с данными не найдены по указанному пути.

    Prints:
        Precision (среднее значение точности детекции),
        Recall (среднее значение полноты),
        F1-score (среднее значение F1-метрики),
        mAP@0.5 (mean Average Precision при IoU=0.5),
        mAP@0.5:0.95 (mean Average Precision при IoU от 0.5 до 0.95).
    """
    model_path = Path(model_path)
    data_path = Path(data_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {data_path}")

    model = YOLO(str(model_path))

    metrics = model.val(data=str(data_path))

    precision, recall, map50, map50_95 = metrics.box.mean_results()[:4]
    f1_mean = metrics.box.f1.mean()

    print("\nМетрики модели:")
    print(f"Precision (mean): {precision:.4f}")
    print(f"Recall (mean):    {recall:.4f}")
    print(f"F1-score (mean):  {f1_mean:.4f}")
    print(f"mAP@0.5:          {map50:.4f}")
    print(f"mAP@0.5:0.95:     {map50_95:.4f}")


if __name__ == "__main__":
    from config import DATA_YAML_PATH, CURRENT_MODEL_PATH

    metrix_run(model_path=CURRENT_MODEL_PATH, data_path=DATA_YAML_PATH)
