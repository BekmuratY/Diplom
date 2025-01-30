import cv2
import numpy as np
import pickle
import sqlite3
import time
import os
import logging
import mediapipe as mp
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Пути к файлам
FACES_DIR = "faces"
EMBEDDINGS_FILE = "embeddings.pkl"
DB_FILE = "face_logs.db"

# Инициализация Mediapipe
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Удаление старой базы данных
def delete_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        logger.info(f"🗑️ База данных {DB_FILE} удалена.")

# Инициализация базы данных
def init_db():
    delete_db()  # Опционально: удалить старую базу данных перед созданием новой
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            entry_time TEXT,
            recognition_count INTEGER DEFAULT 0
        )
    """)

    cursor.execute("PRAGMA table_info(face_logs)")
    columns = [col[1] for col in cursor.fetchall()]
    if "recognition_count" not in columns:
        cursor.execute("ALTER TABLE face_logs ADD COLUMN recognition_count INTEGER DEFAULT 0")
        logger.info("✅ Добавлено поле `recognition_count`.")

    conn.commit()
    conn.close()
    logger.info("📂 База данных face_logs.db загружена/обновлена.")

# Запись неизвестного лица в базу
def log_unknown_face():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    entry_time = time.strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("SELECT * FROM face_logs WHERE name = ?", ("Unknown",))
    existing_entry = cursor.fetchone()

    if existing_entry:
        cursor.execute("UPDATE face_logs SET recognition_count = recognition_count + 1 WHERE name = ?", ("Unknown",))
    else:
        cursor.execute("INSERT INTO face_logs (name, entry_time, recognition_count) VALUES (?, ?, ?)", 
                       ("Unknown", entry_time, 1))
    
    conn.commit()
    conn.close()
    logger.info("❌ Лицо не распознано! Записано в базу как `Unknown`.")

# Увеличение счетчика распознавания для известных лиц
def increment_recognition_count(name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM face_logs WHERE name = ?", (name,))
    existing_entry = cursor.fetchone()

    if existing_entry:
        cursor.execute("UPDATE face_logs SET recognition_count = recognition_count + 1 WHERE name = ?", (name,))
    else:
        entry_time = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO face_logs (name, entry_time, recognition_count) VALUES (?, ?, ?)", 
                       (name, entry_time, 1))

    conn.commit()
    conn.close()
    logger.info(f"✅ Лицо {name} распознано! Счетчик увеличен.")

# Функция сравнения лица с базой данных
def recognize_face(face_image):
    for person_name in os.listdir(FACES_DIR):
        person_folder = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_file in os.listdir(person_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                reference_img = os.path.join(person_folder, img_file)

                try:
                    result = DeepFace.verify(face_image, reference_img, model_name="Facenet", enforce_detection=False)
                    if result["verified"]:
                        return person_name
                except Exception as e:
                    logger.error(f"⚠️ Ошибка при сравнении с {reference_img}: {e}")

    return None

# Обработка кадров
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    face_names = []
    face_locations = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_box, h_box = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

            left = max(x, 0)
            top = max(y, 0)
            right = min(x + w_box, w)
            bottom = min(y + h_box, h)

            face = frame[top:bottom, left:right]

            name = recognize_face(face)
            if not name:
                log_unknown_face()
                name = "Unknown"
            else:
                increment_recognition_count(name)

            face_names.append(name)
            face_locations.append((top, right, bottom, left))

    return face_locations, face_names

# Основная функция работы с камерой
def main():
    init_db()

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logger.error("❌ Ошибка: Не удалось открыть камеру!")
        return

    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logger.error("❌ Не удалось получить кадр, пропускаем итерацию...")
                time.sleep(1)
                continue

            future = executor.submit(process_frame, frame.copy())
            face_locations, face_names = future.result()

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()