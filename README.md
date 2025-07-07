# Food Detection with YOLOv11

### Тестовое задание — Python Developer (Computer Vision, ML)

Финальный отчёт: [`report.md`](./report.md)

---

## Описание проекта

Цель проекта — построить пайплайн для распознавания блюд на видео из ресторанов с использованием модели **YOLOv11**. Задача включает:
- извлечение кадров из видео,
- аннотацию данных,
- обучение модели,
- инференс с визуализацией результатов,
- анализ метрик качества.

---

## Структура проекта
```
food-recognition-yolo/
├── data/
│   ├── dataset/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── data.yaml
│   ├──frames/
│   │   ├── extracted/
│   │   └── processed/
│   └── videos/
├── models/
│   ├── default/
│   │   └── yolo11n.pt
│   └── my_models/
│       └── best.pt
├── outputs/
│   └── videos/
├── reported_files/*
├── run/*
├── scripts/
│   ├── predict/
│   │   ├── __init__.py
│   │   ├── test_by_my_code.py
│   │   └── test_by_YOLO.py
│   ├── train/
│   │   └── train_new_model.py
│   └── utils/
│       ├── videolib/
│       │   ├── __init__.py
│       │   ├── extract_frames.py
│       │   ├── frames_to_video.py
│       │   └── get_video_fps.py
│       ├── config.py
│       ├── metrix.py
│       └── model_on_frame.py
├── config.yaml
├── predict.py
├── train.py
├── README.md
├── LICENSE
├── report.md
└── requirements.txt

```


---

## Установка


```bash
pip install -r requirements.txt
```

Требуется:
Python 3.12+

---

## Запуск

Перед запуском поместите видеофайлы в директорию [`data/videos/`](./data/videos/), либо обновите пути в конфигурации [`config.py`](./scripts/utils/config.py).  
Запуск: [`python predict.py`](./predict.py)

---

## Обучение

Запуск обучения: [`python train.py`](./train.py)

---
