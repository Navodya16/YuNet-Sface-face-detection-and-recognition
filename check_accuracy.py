import os
import time
import cv2
import numpy as np
import pickle

COSINE_THRESHOLD = 0.5

def match(recognizer, feature1, dictionary):
    max_score = 0.0
    sim_user_id = ""
    for user_id, feature2 in dictionary.items():
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= max_score:
            max_score = score
            sim_user_id = user_id
    if max_score < COSINE_THRESHOLD:
        return False, ("Unknown", 0.0)
    return True, (sim_user_id, max_score)

def recognize_face(image, face_detector, face_recognizer):
    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(image)

    if faces is None:
        return None, None

    features = []
    for face in faces:
        aligned_face = face_recognizer.alignCrop(image, face)
        feat = face_recognizer.feature(aligned_face)
        features.append(feat)
    return features, faces

def generate_embeddings(train_dir, face_detector, face_recognizer, output_file="embeddings.pickle"):
    embeddings = {}
    
    for person_name in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        person_features = []
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            print(f"[INFO] Processing image: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Skipping {image_path}, cannot read image.")
                continue

            features, faces = recognize_face(image, face_detector, face_recognizer)
            if features is not None:
                person_features.extend(features)

        if person_features:
            # Average features for the person
            average_feature = np.mean(person_features, axis=0)
            embeddings[person_name] = average_feature

    # Save embeddings to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"[INFO] Embeddings saved to {output_file}")

    return embeddings

def load_embeddings(file_path="embeddings.pickle"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            dictionary = pickle.load(f)
        print(f"[INFO] Embeddings loaded from {file_path}")
        return dictionary
    else:
        print(f"[WARNING] Embedding file {file_path} does not exist!")
        return {}

def evaluate_test_set(test_dir, dictionary, face_detector, face_recognizer):
    TP, FP, FN = 0, 0, 0
    total_images = 0
    total_time = 0

    for person_name in os.listdir(test_dir):
        person_folder = os.path.join(test_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            print(f"[INFO] Processing image: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Skipping {image_path}, cannot read image.")
                continue

            total_images += 1
            start_time = time.time()

            features, faces = recognize_face(image, face_detector, face_recognizer)
            if features is None:
                FN += 1  # No face detected
                continue

            predicted_name = "Unknown"
            for feature in features:
                match_found, (predicted_name, score) = match(face_recognizer, feature, dictionary)

            print(f"[RESULT] Actual: {person_name}, Predicted: {predicted_name}")

            if predicted_name == person_name:
                TP += 1  # Correct identification
            elif predicted_name == "Unknown":
                FN += 1  # Missed identification
            else:
                FP += 1  # Incorrect identification

            total_time += (time.time() - start_time)

    accuracy = (TP / total_images) * 100 if total_images > 0 else 0
    precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0
    recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0
    avg_time = total_time / total_images if total_images > 0 else 0
    avg_fps = 1 / avg_time if avg_time > 0 else 0

    print(f"[INFO] Total images processed: {total_images}")
    print(f"[INFO] Accuracy: {accuracy:.2f}%")
    print(f"[INFO] Precision: {precision:.2f}")
    print(f"[INFO] Recall: {recall:.2f}")
    print(f"[INFO] Average FPS: {avg_fps:.2f}")
    print(f"[INFO] missed identifications: {FN}")
    print(f"[INFO] incorrect identifications: {FP}")

def main():
    model_path = "data"
    train_dir = "data/accuracy dataset/train"  # Path to train set
    test_dir = "data/accuracy dataset/test"  # Path to test set

    # Load face detection and recognition models
    face_detector = cv2.FaceDetectorYN_create(
        os.path.join(model_path, "models", "face_detection_yunet_2023mar.onnx"), "", (0, 0)
    )
    face_detector.setScoreThreshold(0.87)
    face_recognizer = cv2.FaceRecognizerSF_create(
        os.path.join(model_path, "models", "face_recognizer_fast.onnx"), ""
    )

    # Generate embeddings
    embeddings_file = "embeddings.pickle"
    #dictionary = generate_embeddings(train_dir, face_detector, face_recognizer, embeddings_file)
    dictionary = load_embeddings(embeddings_file)
    print("embeddings generated")

    # Evaluate test set
    evaluate_test_set(test_dir, dictionary, face_detector, face_recognizer)

if __name__ == "__main__":
    main()
