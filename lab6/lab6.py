import cv2
import mediapipe as mp

# Инициализация модели Pose Detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Открываем видеофайл
video_path = "Download.mp4"  # Укажи путь к видео
cap = cv2.VideoCapture(video_path)

# Переменные для отслеживания движения
prev_hip_y = None
jump_threshold = 0.1  # Порог для определения прыжка
cadr = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Если видео закончилось, выйти

    # Преобразуем изображение в RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Получаем ключевые точки позы
    results = pose.process(frame_rgb)

    # Проверяем, есть ли ключевые точки
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Получаем координаты бедра и колена
        landmarks = results.pose_landmarks.landmark
        hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y  # Координата бедра
        knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y  # Координата колена

        
        # Определение приседания
        if hip_y > knee_y and cadr % 10 == 0:  # Бедро ниже колена
            print("Человек присел!")

        # Определение прыжка
        if cadr % 25 == 0:
            if prev_hip_y is not None :
                if prev_hip_y - hip_y > jump_threshold:
                    print("Человек прыгает!")
            prev_hip_y = hip_y  # Обновляем предыдущее значение бедра    
            

        
    
    cadr+= 1
    if cadr == 50: cadr = 0
    # Отображаем кадр
    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):  # Нажать 'q' для выхода
        break

cap.release()
cv2.destroyAllWindows()
