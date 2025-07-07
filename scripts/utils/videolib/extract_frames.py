import cv2
import os
from tqdm import tqdm
from pathlib import Path


def extract_frames(
    video_path: str | Path, output_folder: str | Path, fps: int = 30
) -> None:
    """
    Извлекает кадры из видео с указанной частотой кадров и сохраняет их в папку.

    Args:
        video_path (str | Path): Путь к видеофайлу.
        output_folder (str | Path): Путь к папке для сохранения кадров.
        fps (int, optional): Частота кадров для извлечения. По умолчанию 30.
    """
    print(f"Обрабатываем видео: {video_path}")
    print(f"Сохраняем кадры в: {output_folder}")
    print(f"FPS для извлечения: {fps}")

    if not os.path.exists(video_path):
        print(f"Видео не найдено: {video_path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    print(f"Всего кадров: {total_frames}, Длительность: {duration:.2f} секунд")

    frame_interval = int(video_fps / fps)
    count = 0
    saved = 0

    with tqdm(total=total_frames, desc="Извлечение кадров", unit="кадр") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"frame{saved:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved += 1
            count += 1
            pbar.update(1)

    cap.release()
    print(f"Сохранено кадров: {saved}")
