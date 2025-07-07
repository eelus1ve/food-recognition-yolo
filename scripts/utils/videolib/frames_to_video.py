import cv2
import os
from tqdm import tqdm
from pathlib import Path


def frames_to_video(
    frames_folder: str | Path, output_video_path: str | Path, fps: int = 30
) -> None:
    """
    Собирает видео из последовательности изображений (кадров) из указанной папки.

    Args:
        frames_folder (str | Path): Путь к папке с изображениями (.jpg, .png).
        output_video_path (str | Path): Путь для сохранения итогового видео файла (.mp4).
        fps (int, optional): Частота кадров итогового видео. По умолчанию 30.
    """
    print(f"Читаем кадры из: {frames_folder}")
    print(f"Собираем в видео: {output_video_path}")
    print(f"FPS: {fps}")

    frame_files = sorted(
        [
            os.path.join(frames_folder, f)
            for f in os.listdir(frames_folder)
            if f.endswith((".jpg", ".png"))
        ]
    )

    if not frame_files:
        print("Не найдено кадров для сборки.")
        return

    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_path in tqdm(frame_files, desc="Сборка видео", unit="кадр"):
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print("Видео успешно создано.")
