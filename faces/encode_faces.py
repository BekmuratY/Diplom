import os
import pickle
import face_recognition

FACES_DIR = "faces"  # Папка с папками пользователей
OUTPUT_FILE = "embeddings.pkl"

def main():
    encodings = []
    names = []

    # Перебираем всех пользователей (каждый в своей подпапке)
    for person_name in os.listdir(FACES_DIR):
        person_folder = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue  # Если это не папка, пропускаем

        # Перебираем все изображения внутри папки
        for img_file in os.listdir(person_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_folder, img_file)
                print(f"Обрабатываем: {img_path}")

                # Загружаем фото
                image = face_recognition.load_image_file(img_path)
                # Извлекаем эмбеддинги (может быть несколько лиц, но возьмём первый)
                face_encs = face_recognition.face_encodings(image)
                if len(face_encs) > 0:
                    encodings.append(face_encs[0])
                    names.append(person_name)
                else:
                    print(f"Лицо не найдено: {img_path}")

    # Сохраняем в pickle
    data = {"encodings": encodings, "names": names}
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"Готово! Сохранён файл: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
