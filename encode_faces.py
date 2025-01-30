import os
import pickle
import face_recognition
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FACES_DIR = "faces"
OUTPUT_FILE = "embeddings.pkl"

def is_face_unique(new_encoding, encodings, threshold=0.6):
    if not encodings:
        return True
    similarities = cosine_similarity([new_encoding], encodings)
    return all(similarity < threshold for similarity in similarities[0])

def process_image(img_path, person_name):
    try:
        image = face_recognition.load_image_file(img_path)
        face_encs = face_recognition.face_encodings(image)
        if len(face_encs) > 0:
            return face_encs[0], person_name
        else:
            logging.warning(f"Лицо не найдено: {img_path}")
    except Exception as e:
        logging.error(f"Ошибка при обработке {img_path}: {e}")
    return None, None

def load_existing_embeddings(output_file):
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def main():
    existing_data = load_existing_embeddings(OUTPUT_FILE)
    encodings = existing_data["encodings"]
    names = existing_data["names"]

    with ThreadPoolExecutor() as executor:
        futures = []
        for person_name in os.listdir(FACES_DIR):
            person_folder = os.path.join(FACES_DIR, person_name)
            if not os.path.isdir(person_folder):
                continue

            for img_file in os.listdir(person_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_folder, img_file)
                    futures.append(executor.submit(process_image, img_path, person_name))

        for future in futures:
            encoding, name = future.result()
            if encoding is not None and name is not None and is_face_unique(encoding, encodings):
                encodings.append(encoding)
                names.append(name)

    data = {"encodings": encodings, "names": names}
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    logging.info(f"Готово! Сохранён файл: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

