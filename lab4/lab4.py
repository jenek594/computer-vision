import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow
# URL изображения
image_url = 'https://images.mentoday.ru/upload/img_cache/8fb/8fb0cd1c296ea17a6c196a79e8b91792_cropped_666x444.jpg'

# Загрузка изображения из URL
import urllib.request
urllib.request.urlretrieve(image_url, "image.jpg")

# Чтение изображения
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертируем в RGB для корректного отображения

# Отображение изображения
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8n.pt")  # Вы можете заменить на yolov8s.pt, yolov8m.pt и т.д., если у вас достаточно ресурсов

# Прогон изображения через модель
results = model("image.jpg")

# Отображение результатов
annotations = results[0].boxes.data.cpu().numpy()  # Получение данных о рамках
labels = results[0].names  # Получение списка классов

# Вывод информации о каждом найденном объекте
for annotation in annotations:
    x_min, y_min, x_max, y_max, confidence, class_id = annotation

    # Координаты с двумя знаками после запятой
    x_min = round(float(x_min), 2)
    y_min = round(float(y_min), 2)
    x_max = round(float(x_max), 2)
    y_max = round(float(y_max), 2)

    # Название класса
    label = labels[int(class_id)]

    # Уровень уверенности
    confidence = round(float(confidence), 2)

    # Вывод информации
    print(f"Объект: {label}, Координаты: ({x_min}, {y_min}, {x_max}, {y_max}), Уверенность: {confidence}")

    # Отрисовка рамки на изображении
    cv2.rectangle(image_rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    cv2.putText(image_rgb, f"{label}: {confidence}", (int(x_min), int(y_min) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Отображение изображения с рамками
plt.imshow(image_rgb)
plt.axis("off")
plt.show()


# Загрузка предобученной модели YOLO (например, YOLOv8n)
model = YOLO("yolov8n.pt")  # Можно использовать yolov8s.pt или другие, в зависимости от доступных ресурсов

# Укажи путь к файлу .yaml, который указывает на данные (train и valid)
model.train(data='Synthetic Fruit.v1i.yolov8\data.yaml', epochs=5, imgsz=640)

print("Обучение завершено!")