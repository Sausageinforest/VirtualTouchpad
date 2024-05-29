import cv2
import mediapipe as mp
import pyautogui
import time

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Инициализация захвата видео с камеры
cap = cv2.VideoCapture(0)

# Коэффициент чувствительности
sensitivity = 30

# Инициализация предыдущих координат пальца и курсора
prev_x, prev_y = 0, 0
first_frame = True

# Получение размеров экрана
screen_width, screen_height = pyautogui.size()

# Инициализация координат для сглаживания
smoothed_mouse_x, smoothed_mouse_y = pyautogui.position()
alpha = 0.2  # Коэффициент сглаживания

# Частота кадров камеры
fps = 60
interp_steps = 2  # Количество интерполяционных шагов

# Использование MediaPipe для отслеживания рук
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        start_time = time.time()

        success, image = cap.read()
        if not success:
            print("Не удалось получить кадр с камеры")
            break

        # Поворот изображения для правильного отображения
        image = cv2.flip(image, 1)

        # Преобразование изображения в формат RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Обнаружение рук
        results = hands.process(image_rgb)

        # Рисование аннотаций на изображении
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Получение координат среднего пальца
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

                image_height, image_width, _ = image.shape
                x = int(middle_finger_tip.x * image_width)
                y = int(middle_finger_tip.y * image_height)

                if first_frame:
                    prev_x, prev_y = x, y
                    first_frame = False

                # Вычисление изменений в координатах
                dx = (x - prev_x) * sensitivity
                dy = (prev_y - y) * sensitivity  # Инверсия оси Y

                # Получение текущих координат курсора
                current_mouse_x, current_mouse_y = pyautogui.position()

                # Вычисление новых координат курсора
                new_mouse_x = current_mouse_x + dx
                new_mouse_y = current_mouse_y + dy

                # Применение сглаживания
                smoothed_mouse_x = alpha * new_mouse_x + (1 - alpha) * smoothed_mouse_x
                smoothed_mouse_y = alpha * new_mouse_y + (1 - alpha) * smoothed_mouse_y

                # Проверка границ экрана
                smoothed_mouse_x = max(0, min(screen_width - 1, smoothed_mouse_x))
                smoothed_mouse_y = max(0, min(screen_height - 1, smoothed_mouse_y))

                # Интерполяция движения курсора
                for i in range(1, interp_steps + 1):
                    interp_x = current_mouse_x + (smoothed_mouse_x - current_mouse_x) * i / interp_steps
                    interp_y = current_mouse_y + (smoothed_mouse_y - current_mouse_y) * i / interp_steps
                    pyautogui.moveTo(interp_x, interp_y)
                    time.sleep(1 / (fps * interp_steps))

                # Обновление предыдущих координат
                prev_x, prev_y = x, y

                # Проверка нажатия большого пальца (если большой палец поднят)
                if thumb_tip.y < thumb_ip.y:
                    pyautogui.click()
                # Проверка нажатия мизинца (если мизинец поднят)
                if middle_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y:
                    pyautogui.rightClick()

        # Отображение изображения
        cv2.imshow('Hand Tracking', image)

        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
