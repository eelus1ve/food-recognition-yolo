# Отчет по проекту: Распознавание еды на видео с помощью YOLOv11

---

## Цель проекта

Построить модель для распознавания блюд на видео с помощью архитектуры YOLOv11. Результатом является модель, способная в реальном времени определять блюда на кадрах видео.

---

## 1. Подготовка данных

### Извлечение кадров
- Видео разбиты на кадры с частотой **30 FPS** с помощью `OpenCV`.
- Обработаны следующие видео:
  - `1.MOV`, `2_1.MOV`, `3_1.MOV`, `3_2.MOV`, `4.MOV`
- Примерный объем: **~ 6 623 кадров**.

### Разметка
- Разметка произведена вручную в формате YOLO.
- Классы: `['borsch', 'empty glass saucer', 'empty glass tea cup', 'empty plate of borsch', 'empty plate of greek salad', 'empty plate of pumpkin soup', 'empty plate of vegetable salad', 'empty shot glass', 'full glass cup of tea', 'full shot glass', 'glass teapot', 'greek salad', 'meat', 'pile of onions', 'pumpkin soup', 'sandwiches with lard', 'the chicken', 'vegetable salad']`.
- Инструмент аннотации: Roboflow.

### Аугментации
- Применены:
  - Горизонтальное отражение
  - Изменение яркости и контраста
  - Вращение и обрезка

### Разделение
- `Train`: 70%  
- `Val`: 20%  
- `Test`: 10%

---

## 2. Обучение модели

### Архитектура
- YOLOv11 (модель: [`yolo11n.pt`](./models/default/yolo11n.pt))

### Гиперпараметры

| Параметр     | Значение   |
|--------------|------------|
| Epochs       | 400        |
| Batch Size   | 32         |
| Image Size   | 640        |
| Learning Rate| 0.001      |
| Device       | CUDA       |

### Конфигурация
- Файл: [`data/dataset/data.yaml`](./data/dataset/data.yaml)  
- Кол-во классов: `18`
- Формат: YOLOv5-style

---

## 3. Метрики

Модель оценивалась на валидационном наборе после завершения обучения [`best.pt`](./reported_files/weights/best.pt):

| Метрика         | Значение     |
|-----------------|--------------|
| **mAP@0.5**     | `0.9893`     |
| **mAP@0.5:0.95**| `0.9252`     |
| **Precision**   | `0.9737`     |
| **Recall**      | `0.9811`     |
| **F1-score**    | `0.9772`     |

Подробности в: [`reported_files/results.csv`](./reported_files/results.csv), [`runs/detect/train/results.csv`](./runs/detect/train/results.csv)  
Графики обучения (loss, mAP): [`reported_files/`](./reported_files/), [`runs/detect/train/`](./runs/detect/train/)

---

## 4. Результаты инференса на видео

- Инференс выполнен на всех исходных видео.
- Кадры размечались моделью, затем собраны обратно в видео

---

## 5. Временные затраты

| Этап                     | Время         |
|--------------------------|---------------|
| Разметка данных          | ~ 15 ч        |
| Извлечение кадров        | ~ 1 ч         |
| Обучение и валидация     | ~ 12 ч        |
| Инференс и визуализация  | ~ 3 ч         |
| Подготовка отчета        | ~ 5 ч         |

---

## 6. Использование YOLO ранее
До этого проекта не работал с YOLO

---

## 7. Сложности и выводы

### Сложности:
- Ограниченный объем данных
- Ограниченные ресурсы GPU

---

## 8. Ссылки
- Исходный код: [`/`](./)
- Метрики: [`reported_files/results.csv`](./reported_files/results.csv), [`runs/detect/train/results.csv`](./runs/detect/train/results.csv)
- Графики обучения (loss, mAP): [`reported_files/`](./reported_files/), [`runs/detect/train/`](./runs/detect/train/)
- Видео: [`outputs/video`](./outputs/video)
- Модель: [`reported_files/weights/best.pt`](./reported_files/weights/best.pt), [`models/my_models/best.pt`](./models/my_models/best.pt)

---
