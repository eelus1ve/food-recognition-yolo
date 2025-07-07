from ultralytics import YOLO
from pathlib import Path


def train_run(
    model_path: str | Path,
    data_yaml_path: str | Path,
    test_images_dir: str | Path,
    config: dict,
    start_test: bool = False,
) -> tuple:
    """
    Обучает YOLO-модель на заданных данных, выполняет валидацию и при необходимости
    последующее тестирование модели на изображениях из указанной папки.

    Args:
        model_path (str | Path): Путь к файлу модели (.pt) для дообучения или инициализации.
        data_yaml_path (str | Path): Путь к YAML-файлу с конфигурацией датасета (train, val, test).
        test_images_dir (str | Path): Путь к папке с изображениями для тестового запуска модели.
        config (dict): Параметры конфигурации для метода train() (например, epochs, batch, imgsz и т.п.).
        start_test (bool, optional): Если True, после обучения и валидации запускает предсказания
            на изображениях из test_images_dir с выводом результата. По умолчанию False.

    Returns:
        tuple: Кортеж из результатов обучения (train_results) и результатов валидации (metrics).
    
    Raises:
        FileNotFoundError: Если указанные пути к модели, YAML-файлу или папке с тестовыми изображениями не найдены.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {data_yaml_path}")

    model = YOLO(str(model_path))

    train_results = model.train(
        data=str(data_yaml_path),
        **config
    )

    metrics = model.val()

    if start_test:
        if not test_images_dir.exists():
            raise FileNotFoundError(
                f"Папка с тестовыми изображениями не найдена: {test_images_dir}"
            )

        for image_path in sorted(test_images_dir.glob("*.[jp][pn]g")):
            results = model(str(image_path))
            results[0].show()
            input(f"Нажмите Enter для продолжения (файл: {image_path.name})...")

    return train_results, metrics
