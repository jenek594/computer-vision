import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

# Загружаем логи обучения
log_path = "runs/detect/train6/results.csv"
df = pd.read_csv(log_path)

# Проверяем, какие метрики есть в логах
print(df.columns)

# Строим график mAP@50 по эпохам
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50", marker="o", linestyle="-")

plt.xlabel("Эпоха")
plt.ylabel("mAP@50")
plt.title("Точность модели YOLO на валидационной выборке")
plt.legend()
plt.grid()
plt.show()


# Загружаем обученную модель
best_model = YOLO("runs/detect/train6/weights/best.pt")

# Пути к изображениям (замени на свои файлы)
image_paths = [
    "C:/Users/user/OneDrive/Desktop/computer vision/computer-vision/lab4/test1.jpg",
    "C:/Users/user/OneDrive/Desktop/computer vision/computer-vision/lab4/test2.jpg",
    "C:/Users/user/OneDrive/Desktop/computer vision/computer-vision/lab4/test3.jpg",
    "C:/Users/user/OneDrive/Desktop/computer vision/computer-vision/lab4/test4.jpg",
    "C:/Users/user/OneDrive/Desktop/computer vision/computer-vision/lab4/test5.jpg"
]

# Применяем модель к каждому изображению
for img_path in image_paths:
    results = best_model(img_path)

    # Отображаем изображение с предсказаниями
    res_plotted = results[0].plot()  # Получаем изображение с боксами
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
