import os
import cv2
import sys
from ultralytics import YOLO
from collections import Counter
from pathlib import Path


def run_model_on_frames(
    model_path: str | Path, input_folder: str | Path, output_folder: str | Path
) -> None:
    """
    Выполняет инференс модели YOLO на изображениях из указанной папки и сохраняет
    аннотированные кадры с наложенными результатами детекции в выходную папку.

    Args:
        model_path (str | Path): Путь к файлу модели YOLO (.pt).
        input_folder (str | Path): Путь к директории с входными изображениями (JPEG, PNG).
        output_folder (str | Path): Путь к директории, куда будут сохранены аннотированные изображения.
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"Загружаем модель: {model_path}")
    model = YOLO(model_path)

    frames = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png"))]
    )
    total = len(frames)

    if not frames:
        print("Нет изображений для обработки.")
        return

    print(f"Обрабатываем кадры из: {input_folder}")
    print(f"Сохраняем результаты в: {output_folder}")
    print(f"Всего кадров: {total}\n")
    print("")
    print("")

    for idx, frame_name in enumerate(frames, start=1):
        input_path = os.path.join(input_folder, frame_name)
        output_path = os.path.join(output_folder, frame_name)

        img = cv2.imread(input_path)
        results = model(img, verbose=False)[0]
        annotated_frame = results.plot()
        cv2.imwrite(output_path, annotated_frame)

        if results.boxes is not None and results.boxes.cls.numel() > 0:
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            class_counts = Counter(class_ids)
            class_names = results.names
            detected_summary = ", ".join(
                f"{class_names[c]}: {class_counts[c]}" for c in class_counts
            )
        else:
            detected_summary = "Объекты не найдены"

        line_data = f"{frame_name} | {detected_summary}"
        line_progress = f"[{idx}/{total}] Обработка кадров..."

        sys.stdout.write("\x1b[2A")
        sys.stdout.write("\r" + " " * 120 + "\r" + line_data + "\n")
        sys.stdout.write(" " * 120 + "\r" + line_progress + "\n")
        sys.stdout.flush()

    print("Обработка завершена!\n")
