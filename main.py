import cv2
import face_recognition
import numpy as np
import pickle

def load_embeddings(pickle_file="embeddings.pkl"):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def main():
    # 1. Открываем веб-камеру
    video_capture = cv2.VideoCapture(0)

    # 2. Загружаем embeddings.pkl
    try:
        known_face_encodings, known_face_names = load_embeddings("embeddings.pkl")
        print("База успешно загружена. Всего лиц:", len(known_face_encodings))
    except Exception as e:
        print("Ошибка загрузки базы лиц:", e)
        known_face_encodings, known_face_names = [], []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Не удалось получить кадр")
            break

        # Уменьшим кадр, чтобы ускорить распознавание
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Конвертируем в RGB
        rgb_small_frame = small_frame[:, :, ::-1]

        # 3. Найдём лица
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            name = "Unknown"
            # Если база не пуста
            if len(known_face_encodings) > 0:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

        # 4. Отрисовка рамок и имён
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Рамка
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Подпись
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        # Показываем результат
        cv2.imshow('Video', frame)

        # Нажмите 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
