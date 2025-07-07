from pathlib import Path
import cv2


def get_video_fps(video_path: str | Path) -> int:
    """
    Получает частоту кадров (FPS) видеофайла.
    Args:
        video_path (str | Path): Путь к видеофайлу.
    Returns:
        int: Частота кадров видео, округленная до ближайшего целого.
    Raises:
        ValueError: Если видео не удалось открыть.
    Пример:
        fps = get_video_fps("data/videos/1.MOV")
        print(f"FPS видео: {fps}")
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(round(fps))


if __name__ == "__main__":
    fps = get_video_fps("data/videos/1.MOV")
    print(f"FPS видео: {fps}")
