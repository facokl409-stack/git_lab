import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
from collections import deque
import win32api
import win32con
import sys
import os


# Конфигурация для максимально плавного управления
class Config:
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FPS_TARGET = 60
    # Плавные параметры сглаживания
    BASE_SMOOTHING = 0.3
    VELOCITY_SMOOTHING = 0.15
    DEPTH_SMOOTHING_FACTOR = 0.4
    DEAD_ZONE = 8
    MOVE_THRESHOLD = 1.5
    CLICK_DISTANCE_THRESHOLD = 30
    DRAG_DISTANCE_THRESHOLD = 45
    SCROLL_SPEED = 80
    SCROLL_DEAD_ZONE = 15
    # Новые параметры для горизонтальной прокрутки
    HORIZONTAL_SCROLL_THRESHOLD = 25
    HORIZONTAL_SCROLL_SPEED = 100
    HORIZONTAL_SCROLL_COOLDOWN = 0.05
    SWIPE_MIN_DISTANCE = 80
    SWIPE_MIN_SPEED = 150
    SWIPE_COOLDOWN = 0.3
    HISTORY_SIZE = 8
    HAND_SIZE_REF = 120
    PREDICTION_FACTOR = 0.1
    STABILITY_THRESHOLD = 0.3
    # Пороги для жестов пролистывания изображений
    FLIP_GESTURE_THRESHOLD = 35
    FLIP_GESTURE_COOLDOWN = 0.5
    # Пороги для новых жестов
    DRAG_HOLD_THRESHOLD = 0.3  # Порог для "захвата" (сжатие пальцев)
    ZOOM_THRESHOLD = 30  # Порог для масштабирования
    ZOOM_MIN_DISTANCE = 50  # Мин. расстояние между пальцами для масштабирования
    ZOOM_COOLDOWN = 0.2  # Ожидание между масштабированием


# Улучшенный оценщик глубины с плавным переходом
class SmoothDepthEstimator:
    def __init__(self):
        self.depth_history = deque(maxlen=10)
        self.last_depth = 128
        self.smoothed_depth = 128
        self.depth_velocity = 0
        self.last_update_time = time.time()

    def estimate_depth_from_hand(self, hand_landmarks, frame_width, frame_height):
        """Плавная оценка глубины на основе размера руки"""
        wrist = hand_landmarks.landmark[0]
        middle_tip = hand_landmarks.landmark[12]

        wrist_coords = (wrist.x * frame_width, wrist.y * frame_height)
        middle_tip_coords = (middle_tip.x * frame_width, middle_tip.y * frame_height)

        hand_size_pixels = math.hypot(
            wrist_coords[0] - middle_tip_coords[0],
            wrist_coords[1] - middle_tip_coords[1]
        )

        if hand_size_pixels > 5:
            depth_value = 128 + (Config.HAND_SIZE_REF - hand_size_pixels) * 1.2
            depth_value = min(max(int(depth_value), 40), 215)
            self.depth_history.append(depth_value)

            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time

            if len(self.depth_history) > 1:
                current_velocity = (depth_value - self.depth_history[-2]) / max(dt, 0.001)
                self.depth_velocity = self.depth_velocity * 0.7 + current_velocity * 0.3

            smoothing_factor = 0.6 + (1 - abs(self.depth_velocity) / 100) * 0.3
            smoothing_factor = max(0.4, min(0.95, smoothing_factor))

            self.smoothed_depth = self.smoothed_depth * (1 - smoothing_factor) + depth_value * smoothing_factor

            return int(self.smoothed_depth)

        return int(self.smoothed_depth)


# Продвинутый трекер позиции с предсказанием
class SmoothCursorTracker:
    def __init__(self):
        self.position_history = deque(maxlen=Config.HISTORY_SIZE)
        self.velocity_history = deque(maxlen=5)
        self.acceleration_history = deque(maxlen=3)
        self.last_position = None
        self.last_time = time.time()
        self.smoothed_position = None
        self.prediction_enabled = True
        # Для отслеживания горизонтальных жестов
        self.last_horizontal_position = None
        self.horizontal_velocity = 0

    def update(self, raw_x, raw_y, depth_value):
        """Обновление позиции с продвинутым сглаживанием"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Отслеживание горизонтального движения
        if self.last_horizontal_position is not None and dt > 0.001:
            self.horizontal_velocity = (raw_x - self.last_horizontal_position) / dt
        self.last_horizontal_position = raw_x

        # Инициализация
        if self.smoothed_position is None:
            self.smoothed_position = (raw_x, raw_y)
            self.last_position = (raw_x, raw_y)
            return self.smoothed_position

        # Расчет скорости и ускорения
        if self.last_position is not None and dt > 0.001:
            velocity_x = (raw_x - self.last_position[0]) / dt
            velocity_y = (raw_y - self.last_position[1]) / dt

            self.velocity_history.append((velocity_x, velocity_y))

            if len(self.velocity_history) > 1:
                prev_velocity = self.velocity_history[-2]
                acceleration_x = (velocity_x - prev_velocity[0]) / dt
                acceleration_y = (velocity_y - prev_velocity[1]) / dt
                self.acceleration_history.append((acceleration_x, acceleration_y))

        self.last_position = (raw_x, raw_y)
        self.position_history.append((raw_x, raw_y))

        # Базовое сглаживание с учетом глубины
        depth_factor = depth_value / 255.0
        base_smoothing = Config.BASE_SMOOTHING + depth_factor * Config.DEPTH_SMOOTHING_FACTOR

        # Адаптивное сглаживание на основе скорости
        if self.velocity_history:
            avg_velocity = np.mean(self.velocity_history, axis=0)
            velocity_magnitude = math.hypot(avg_velocity[0], avg_velocity[1])

            velocity_smoothing = Config.VELOCITY_SMOOTHING * (1 + velocity_magnitude / 500)
            velocity_smoothing = min(0.5, velocity_smoothing)

            total_smoothing = base_smoothing * (1 - velocity_smoothing) + velocity_smoothing

        else:
            total_smoothing = base_smoothing

        # Экспоненциальное сглаживание
        smoothed_x = self.smoothed_position[0] * (1 - total_smoothing) + raw_x * total_smoothing
        smoothed_y = self.smoothed_position[1] * (1 - total_smoothing) + raw_y * total_smoothing

        # Предсказание движения для компенсации задержки
        if self.prediction_enabled and self.velocity_history and len(self.velocity_history) >= 3:
            avg_velocity = np.mean(self.velocity_history, axis=0)
            prediction_x = smoothed_x + avg_velocity[0] * 0.03 * Config.PREDICTION_FACTOR
            prediction_y = smoothed_y + avg_velocity[1] * 0.03 * Config.PREDICTION_FACTOR

            max_prediction = 15 * (1 + depth_factor * 2)
            prediction_distance = math.hypot(prediction_x - smoothed_x, prediction_y - smoothed_y)

            if prediction_distance > max_prediction:
                scale = max_prediction / prediction_distance
                prediction_x = smoothed_x + (prediction_x - smoothed_x) * scale
                prediction_y = smoothed_y + (prediction_y - smoothed_y) * scale

            smoothed_x, smoothed_y = prediction_x, prediction_y

        self.smoothed_position = (smoothed_x, smoothed_y)
        return self.smoothed_position

    def get_horizontal_velocity(self):
        """Получение скорости горизонтального движения"""
        return self.horizontal_velocity

    def reset(self):
        """Сброс трекера при потере руки"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self.last_position = None
        self.smoothed_position = None
        self.last_horizontal_position = None
        self.horizontal_velocity = 0


# Улучшенная обработка жестов с плавными переходами
class SmoothGestureProcessor:
    def __init__(self):
        self.last_click_time = 0
        self.click_cooldown = 0.15
        self.is_dragging = False
        self.drag_start_time = 0
        self.drag_threshold = 0.25
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.04
        self.last_horizontal_scroll_time = 0
        self.horizontal_scroll_cooldown = Config.HORIZONTAL_SCROLL_COOLDOWN
        self.last_swipe_time = 0
        self.swipe_cooldown = Config.SWIPE_COOLDOWN
        self.last_flip_time = 0
        self.flip_cooldown = Config.FLIP_GESTURE_COOLDOWN
        self.gesture_confidence = {'MOVE': 1.0}
        self.stable_position_time = 0
        self.last_stable_position = None
        self.gesture_state_history = deque(maxlen=5)
        # Для отслеживания жестов пролистывания
        self.swipe_start_position = None
        self.is_swiping = False
        # Для жеста "отпуска" — флаг, когда пальцы разжимаются
        self.is_releasing = False
        self.release_start_time = 0
        # Для жеста "масштабирования"
        self.is_zooming = False
        self.zoom_start_position = None
        self.zoom_threshold = Config.ZOOM_THRESHOLD
        self.last_zoom_time = 0
        self.zoom_cooldown = Config.ZOOM_COOLDOWN

    def perform_horizontal_scroll(self, amount):
        """Выполнение горизонтальной прокрутки"""
        try:
            win32api.mouse_event(win32con.MOUSEEVENTF_HWHEEL, 0, 0, int(amount), 0)
            return True
        except:
            try:
                if amount > 0:
                    pyautogui.hotkey('ctrl', 'right')
                else:
                    pyautogui.hotkey('ctrl', 'left')
                return True
            except:
                return False

    def perform_image_flip(self, direction):
        """Выполнение пролистывания изображений (жест "флип")"""
        try:
            if direction == "NEXT":
                pyautogui.press('right')
            else:
                pyautogui.press('left')
            return True
        except:
            return False

    def process_gestures(self, hand_landmarks, frame_width, frame_height, depth_value, cursor_pos, cursor_tracker):
        """Плавная обработка жестов с гистерезисом, включая новые жесты"""
        current_time = time.time()
        landmarks = hand_landmarks.landmark

        # Ключевые точки
        index_tip = (landmarks[8].x * frame_width, landmarks[8].y * frame_height)
        thumb_tip = (landmarks[4].x * frame_width, landmarks[4].y * frame_height)
        middle_tip = (landmarks[12].x * frame_width, landmarks[12].y * frame_height)
        ring_tip = (landmarks[16].x * frame_width, landmarks[16].y * frame_height)
        pinky_tip = (landmarks[20].x * frame_width, landmarks[20].y * frame_height)
        index_base = (landmarks[5].x * frame_width, landmarks[5].y * frame_height)
        wrist = (landmarks[0].x * frame_width, landmarks[0].y * frame_height)

        # Адаптивные пороги на основе глубины
        depth_factor = depth_value / 255.0 + 0.2
        click_threshold = Config.CLICK_DISTANCE_THRESHOLD * depth_factor
        drag_threshold = Config.DRAG_DISTANCE_THRESHOLD * depth_factor
        scroll_threshold = Config.SCROLL_DEAD_ZONE * depth_factor
        horizontal_scroll_threshold = Config.HORIZONTAL_SCROLL_THRESHOLD * depth_factor
        flip_gesture_threshold = Config.FLIP_GESTURE_THRESHOLD * depth_factor
        zoom_threshold = Config.ZOOM_THRESHOLD * depth_factor

        # Расстояния между пальцами
        thumb_index_dist = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
        thumb_middle_dist = math.hypot(thumb_tip[0] - middle_tip[0], thumb_tip[1] - middle_tip[1])
        index_middle_dist = math.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])
        middle_ring_dist = math.hypot(middle_tip[0] - ring_tip[0], middle_tip[1] - ring_tip[1])
        ring_pinky_dist = math.hypot(ring_tip[0] - pinky_tip[0], ring_tip[1] - pinky_tip[1])

        # Стабильность позиции для предотвращения ложных срабатываний
        current_pos = (index_tip[0], index_tip[1])

        if self.last_stable_position is None:
            self.last_stable_position = current_pos
            self.stable_position_time = current_time

        position_distance = math.hypot(
            current_pos[0] - self.last_stable_position[0],
            current_pos[1] - self.last_stable_position[1]
        )

        if position_distance < 5:
            stability_factor = min(1.0, (current_time - self.stable_position_time) / Config.STABILITY_THRESHOLD)
        else:
            self.stable_position_time = current_time
            self.last_stable_position = current_pos
            stability_factor = 0.0

        # Плавный переход между жестами
        gesture = "MOVE"
        confidence = 0.7 + stability_factor * 0.3

        # Проверка жестов с приоритетом
        is_click_ready = (current_time - self.last_click_time) > self.click_cooldown
        is_scroll_ready = (current_time - self.last_scroll_time) > self.scroll_cooldown
        is_horizontal_scroll_ready = (current_time - self.last_horizontal_scroll_time) > self.horizontal_scroll_cooldown
        is_flip_ready = (current_time - self.last_flip_time) > self.flip_cooldown
        is_zoom_ready = (current_time - self.last_zoom_time) > self.zoom_cooldown

        # Левый клик
        if thumb_index_dist < click_threshold * 0.9 and is_click_ready and stability_factor > 0.5:
            try:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            except:
                pyautogui.click()
            self.last_click_time = current_time
            gesture = "LEFT_CLICK"
            confidence = 0.95

        # Правый клик
        elif thumb_middle_dist < click_threshold * 0.9 and is_click_ready and stability_factor > 0.5:
            try:
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            except:
                pyautogui.rightClick()
            self.last_click_time = current_time
            gesture = "RIGHT_CLICK"
            confidence = 0.95

        # Жест "смахивание рукой" (движение запястья)
        elif index_tip[1] > wrist[1] and not self.is_dragging and not self.is_swiping:
            horizontal_velocity = cursor_tracker.get_horizontal_velocity()
            horizontal_speed = abs(horizontal_velocity)

            if not self.is_swiping and horizontal_speed > Config.SWIPE_MIN_SPEED:
                self.swipe_start_position = wrist[0]
                self.is_swiping = True

            if self.is_swiping:
                swipe_distance = abs(wrist[0] - self.swipe_start_position)
                if swipe_distance > Config.SWIPE_MIN_DISTANCE and (
                        current_time - self.last_swipe_time) > self.swipe_cooldown:
                    if horizontal_velocity > 0:
                        self.perform_image_flip("NEXT")
                        gesture = "SWIPE_RIGHT"
                    else:
                        self.perform_image_flip("PREVIOUS")
                        gesture = "SWIPE_LEFT"
                    confidence = 0.95
                    self.last_swipe_time = current_time
                    self.is_swiping = False

        # Жест "флип" (большой и указательный пальцы)
        elif index_middle_dist > flip_gesture_threshold * 1.5 and thumb_index_dist > flip_gesture_threshold * 2:
            if index_tip[1] < middle_tip[1] - 20:
                if is_flip_ready:
                    self.perform_image_flip("NEXT")
                    gesture = "FLIP_NEXT"
                    confidence = 0.95
                    self.last_flip_time = current_time
            elif index_tip[1] > middle_tip[1] + 20:
                if is_flip_ready:
                    self.perform_image_flip("PREVIOUS")
                    gesture = "FLIP_PREV"
                    confidence = 0.95
                    self.last_flip_time = current_time

        # Жест "горизонтальная прокрутка" (средний, безымянный, мизинец сжаты)
        elif middle_ring_dist < click_threshold * 0.8 and ring_pinky_dist < click_threshold * 0.8:
            horizontal_movement = index_tip[0] - index_base[0]
            if abs(horizontal_movement) > horizontal_scroll_threshold and is_horizontal_scroll_ready:
                scroll_amount = int(horizontal_movement * Config.HORIZONTAL_SCROLL_SPEED / 200)
                scroll_amount = max(-50, min(50, scroll_amount))

                if abs(scroll_amount) > 5:
                    self.perform_horizontal_scroll(scroll_amount)
                    self.last_horizontal_scroll_time = current_time
                    gesture = "H_SCROLL_RIGHT" if scroll_amount > 0 else "H_SCROLL_LEFT"
                    confidence = 0.85

        # Жест "захват рукой" — сжатие большого и указательного пальца
        elif thumb_index_dist < drag_threshold * 0.7 and thumb_middle_dist < drag_threshold * 0.7:
            if not self.is_dragging:
                if current_time - self.drag_start_time > self.drag_threshold and stability_factor > 0.3:
                    try:
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    except:
                        pyautogui.mouseDown()
                    self.is_dragging = True
                    gesture = "DRAG_START"
                    self.drag_start_time = current_time
                    # Запоминаем позицию для "отпуска"
                    self.release_start_time = current_time
                    self.is_releasing = False

        # Жест "отпускание" — когда пальцы разжимаются
        elif self.is_dragging and thumb_index_dist > drag_threshold * 1.2:
            if current_time - self.release_start_time > 0.1:
                try:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                except:
                    pyautogui.mouseUp()
                self.is_dragging = False
                gesture = "DRAG_END"
                self.is_releasing = True

        # Жест "масштабирование" — сжатие среднего и безымянного пальца
        elif middle_ring_dist < zoom_threshold * 0.8 and ring_pinky_dist < zoom_threshold * 0.8:
            if not self.is_zooming and current_time - self.last_zoom_time > self.zoom_cooldown:
                self.is_zooming = True
                self.zoom_start_position = (middle_tip[0], middle_tip[1])
                self.last_zoom_time = current_time

        # Увеличение масштаба
        elif self.is_zooming and middle_ring_dist < zoom_threshold * 0.8 and ring_pinky_dist < zoom_threshold * 0.8:
            zoom_distance = math.hypot(middle_tip[0] - self.zoom_start_position[0],
                                       middle_tip[1] - self.zoom_start_position[1])
            if zoom_distance > Config.ZOOM_MIN_DISTANCE:
                try:
                    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, 1, 0)
                except:
                    pyautogui.scroll(1)
                self.last_zoom_time = current_time
                gesture = "ZOOM_IN"
                confidence = 0.85

        # Уменьшение масштаба
        elif self.is_zooming and middle_ring_dist > zoom_threshold * 1.5 and ring_pinky_dist > zoom_threshold * 1.5:
            zoom_distance = math.hypot(middle_tip[0] - self.zoom_start_position[0],
                                       middle_tip[1] - self.zoom_start_position[1])
            if zoom_distance > Config.ZOOM_MIN_DISTANCE:
                try:
                    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -1, 0)
                except:
                    pyautogui.scroll(-1)
                self.last_zoom_time = current_time
                gesture = "ZOOM_OUT"
                confidence = 0.85

        # Прокрутка с плавным управлением
        vertical_movement = index_tip[1] - index_base[1]
        if abs(vertical_movement) > scroll_threshold * 1.5 and is_scroll_ready and stability_factor > 0.2:
            scroll_amount = int(vertical_movement * Config.SCROLL_SPEED / 200)
            scroll_amount = max(-50, min(50, scroll_amount))

            if abs(scroll_amount) > 5:
                try:
                    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, scroll_amount, 0)
                except:
                    pyautogui.scroll(scroll_amount // 10)
                self.last_scroll_time = current_time
                gesture = "SCROLL_DOWN" if scroll_amount > 0 else "SCROLL_UP"
                confidence = 0.85

        # Сброс состояния свайпа при потере жеста
        if not (index_tip[1] > wrist[1]):
            self.is_swiping = False

        # История состояний для сглаживания
        self.gesture_state_history.append((gesture, confidence))

        # Возврат текущего состояния
        return gesture, confidence

    def reset(self):
        """Сброс состояния при потере руки"""
        if self.is_dragging:
            try:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            except:
                pyautogui.mouseUp()
            self.is_dragging = False
        self.gesture_state_history.clear()
        self.is_swiping = False
        self.swipe_start_position = None
        self.is_releasing = False
        self.is_zooming = False
        self.zoom_start_position = None
        self.last_zoom_time = 0


# Инициализация MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0
)

screen_width, screen_height = pyautogui.size()
screen_center_x, screen_center_y = screen_width // 2, screen_height // 2

# Инициализация компонентов
depth_estimator = SmoothDepthEstimator()
cursor_tracker = SmoothCursorTracker()
gesture_processor = SmoothGestureProcessor()


def create_smooth_visualization(frame, depth_value, gesture, confidence, cursor_pos=None, hand_center=None):
    """Создание плавной визуализации для отладки с поддержкой новых жестов"""
    overlay = frame.copy()

    # Цветовая индикация глубины
    depth_color = int(depth_value)
    depth_color_bgr = (255 - depth_color, depth_color // 2, depth_color)

    # Визуализация глубины
    if hand_center:
        circle_radius = int(15 + (depth_value / 255.0) * 25)
        cv2.circle(overlay, hand_center, circle_radius, depth_color_bgr, -1, cv2.LINE_AA)
        cv2.circle(overlay, hand_center, circle_radius, (255, 255, 255), 1, cv2.LINE_AA)

    # Визуализация курсора
    if cursor_pos:
        cursor_radius = int(8 + confidence * 5)
        cursor_color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        cv2.circle(overlay, (int(cursor_pos[0] * frame.shape[1] / screen_width),
                             int(cursor_pos[1] * frame.shape[0] / screen_height)),
                   cursor_radius, cursor_color, -1, cv2.LINE_AA)

    # Специальная визуализация для жестов пролистывания
    if "SWIPE" in gesture or "FLIP" in gesture or "H_SCROLL" in gesture or "ZOOM_IN" in gesture or "ZOOM_OUT" in gesture:
        swipe_color = (0, 200, 255) if "SWIPE" in gesture else (255, 100, 0) if "FLIP" in gesture else (150, 0, 200) if "H_SCROLL" in gesture else (0, 255, 0) if "ZOOM_IN" in gesture else (0, 0, 255) if "ZOOM_OUT" in gesture else (100, 100, 100)
        swipe_text = "ЛИСТОВАНИЕ" if "SWIPE" in gesture else "ФЛИП" if "FLIP" in gesture else "ГОР.ПРОКРУТКА" if "H_SCROLL" in gesture else "УВЕЛИЧ" if "ZOOM_IN" in gesture else "УМЕНЬШ" if "ZOOM_OUT" in gesture else "NONE"
        cv2.putText(overlay, swipe_text, (frame.shape[1] - 220, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, swipe_color, 2, cv2.LINE_AA)

        # Анимация стрелки для направления
        arrow_start = (frame.shape[1] - 150, 90)
        arrow_end = (arrow_start[0] + (40 if "RIGHT" in gesture or "NEXT" in gesture else -40), arrow_start[1])
        cv2.arrowedLine(overlay, arrow_start, arrow_end, swipe_color, 3, cv2.LINE_AA, tipLength=0.4)

    # Смешивание с оригинальным кадром
    result = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # Индикатор жеста
    gesture_colors = {
        "MOVE": (0, 200, 0),
        "LEFT_CLICK": (0, 0, 255),
        "RIGHT_CLICK": (255, 0, 255),
        "SCROLL_UP": (255, 100, 0),
        "SCROLL_DOWN": (255, 150, 0),
        "DRAG_START": (0, 255, 255),
        "DRAGGING": (0, 255, 255),
        "DRAG_END": (0, 150, 150),
        "SWIPE_RIGHT": (0, 200, 100),
        "SWIPE_LEFT": (0, 200, 100),
        "FLIP_NEXT": (200, 100, 0),
        "FLIP_PREV": (200, 100, 0),
        "H_SCROLL_RIGHT": (150, 0, 200),
        "H_SCROLL_LEFT": (150, 0, 200),
        "ZOOM_IN": (0, 255, 0),
        "ZOOM_OUT": (0, 0, 255),
        "NONE": (100, 100, 100)
    }

    color = gesture_colors.get(gesture, (100, 100, 100))
    cv2.putText(result, f'{gesture} ({confidence:.2f})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Рамка активной области
    cv2.rectangle(result,
                  (frame.shape[1] // 4, frame.shape[0] // 4),
                  (frame.shape[1] * 3 // 4, frame.shape[0] * 3 // 4),
                  (0, 200, 0), 2, cv2.LINE_AA)

    return result


def main():
    print("🚀 Запуск системы с плавным управлением жестами...")
    print("⌨️  Управление: 'q' - выход, 'd' - отладка, 't' - отображение визуализации")
    print("🖼️  Новые жесты для пролистывания изображений:")
    print("   • Смахивание рукой влево/вправо")
    print("   • Жест 'флип' большим и указательным пальцами")
    print("   • Горизонтальная прокрутка")
    print("   • Захват рукой (сжатие пальцев)")
    print("   • Масштабирование (сжатие среднего и безымянного пальцев)")

    # Настройка камеры с буферизацией
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.FPS_TARGET)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print("❌ Не удалось открыть камеру")
        return

    fps_counter = 0
    start_time = time.time()
    last_frame_time = time.time()
    debug_mode = False
    show_visualization = True

    print("✅ Система готова к работе!")
    print("✨ Улучшенное плавное управление с поддержкой листания и захвата")
    print(f"🎯 Разрешение: {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT}")

    while cap.isOpened():
        current_time = time.time()
        frame_time = current_time - last_frame_time
        last_frame_time = current_time

        # Управление FPS с компенсацией задержки
        target_frame_time = 1.0 / Config.FPS_TARGET
        if frame_time < target_frame_time * 0.8:
            sleep_time = target_frame_time - frame_time
            if sleep_time > 0.001:
                time.sleep(sleep_time * 0.7)

        success, frame = cap.read()
        if not success:
            continue

        # Предварительная обработка кадра
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обработка MediaPipe
        results = hands.process(rgb_frame)

        gesture = "NONE"
        confidence = 0.0
        depth_value = 128
        hand_center = None
        cursor_pos = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Оценка глубины
                depth_value = depth_estimator.estimate_depth_from_hand(
                    hand_landmarks, frame.shape[1], frame.shape[0]
                )

                # Центр руки для визуализации
                wrist = hand_landmarks.landmark[0]
                hand_center = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))

                # Позиция указательного пальца
                index_tip = hand_landmarks.landmark[8]
                raw_cursor_x = np.interp(index_tip.x * frame.shape[1],
                                         [frame.shape[1] // 4, frame.shape[1] * 3 // 4],
                                         [0, screen_width])
                raw_cursor_y = np.interp(index_tip.y * frame.shape[0],
                                         [frame.shape[0] // 4, frame.shape[0] * 3 // 4],
                                         [0, screen_height])

                # Продвинутое сглаживание позиции
                smoothed_cursor = cursor_tracker.update(raw_cursor_x, raw_cursor_y, depth_value)
                cursor_pos = smoothed_cursor

                # Проверка значимости движения
                if cursor_tracker.last_position:
                    move_distance = math.hypot(
                        smoothed_cursor[0] - cursor_tracker.last_position[0],
                        smoothed_cursor[1] - cursor_tracker.last_position[1]
                    )

                    if move_distance > Config.MOVE_THRESHOLD:
                        try:
                            win32api.SetCursorPos((int(smoothed_cursor[0]), int(smoothed_cursor[1])))
                        except:
                            pyautogui.moveTo(smoothed_cursor[0], smoothed_cursor[1], _pause=False)

                # Обработка жестов с поддержкой новых жестов пролистывания
                gesture, confidence = gesture_processor.process_gestures(
                    hand_landmarks, frame.shape[1], frame.shape[0], depth_value, smoothed_cursor, cursor_tracker
                )

                # Визуальная обратная связь для кликов
                if gesture in ["LEFT_CLICK", "RIGHT_CLICK"]:
                    click_radius = 15 + int(10 * math.sin(time.time() * 10))
                    click_color = (0, 255, 0) if gesture == "LEFT_CLICK" else (255, 0, 255)
                    cv2.circle(frame, (int(smoothed_cursor[0] * frame.shape[1] / screen_width),
                                       int(smoothed_cursor[1] * frame.shape[0] / screen_height)),
                               click_radius, click_color, 2, cv2.LINE_AA)
        else:
            # Сброс состояний при потере руки
            cursor_tracker.reset()
            gesture_processor.reset()
            depth_value = 128

        # Подсчет FPS
        fps_counter += 1
        elapsed = current_time - start_time
        fps = fps_counter / elapsed if elapsed > 0 else 0

        if elapsed > 1.0:
            fps_counter = 0
            start_time = current_time

        # Создание отображаемого кадра
        display_frame = frame.copy()

        # Плавная визуализация
        if show_visualization:
            display_frame = create_smooth_visualization(
                display_frame, depth_value, gesture, confidence, cursor_pos, hand_center
            )

        # Отладочная информация
        if debug_mode:
            cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f'Depth: {int(depth_value)}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            if cursor_pos:
                cv2.putText(display_frame, f'Pos: ({int(cursor_pos[0])}, {int(cursor_pos[1])})', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        # Отображение кадра
        cv2.imshow('Плавное управление жестами с поддержкой листания и захвата', display_frame)

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"🔧 Режим отладки: {'ВКЛ' if debug_mode else 'ВЫКЛ'}")
        elif key == ord('t'):
            show_visualization = not show_visualization
            print(f"👁️  Визуализация: {'ВКЛ' if show_visualization else 'ВЫКЛ'}")

    # Очистка ресурсов
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Завершение перетаскиванияq
    if gesture_processor.is_dragging:
        try:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        except:
            pyautogui.mouseUp()

    print("✅ Приложение завершено успешно")
    print(f"✨ Средняя производительность: {fps:.1f} FPS")
    print("👍 Благодарим за использование системы с поддержкой пролистывания изображений и захвата!")


if __name__ == "__main__":
    print("🌟 Система управления жестами с интеллектуальным сглаживанием и поддержкой листания и захвата")
    print("💡 Ключевые особенности:")
    print("  • Продвинутый трекер позиции с предсказанием движения")
    print("  • Адаптивное сглаживание на основе скорости и глубины")
    print("  • 🖼️ Три новых жеста для пролистывания изображений:")
    print("    - Смахивание рукой")
    print("    - Жест 'флип' большим и указательным пальцами")
    print("    - Горизонтальная прокрутка")
    print("  • 🖐️ Новый жест 'захват рукой' — сжатие пальцев")
    print("  • 🖐️ Новый жест 'масштабирование' — сжатие среднего и безымянного пальцев")
    print("  • Плавные переходы между жестами")
    print("  • Гистерезис для предотвращения ложных срабатываний")

    try:
        main()
    except Exception as e:
        print(f"❌ Критическая ошибка: {str(e)}")
        import traceback

        traceback.print_exc()
        print("🔧 Рекомендации:")
        print("1. Убедитесь, что все зависимости установлены")
        print("2. Проверьте доступность камеры")
        print("3. При необходимости уменьшите разрешение камеры")
