import os
import sys
import time
import cv2
import numpy as np

COSINE_THRESHOLD = 0.5

def match(recognizer, feature1, dictionary):
    max_score = 0.0
    sim_user_id = ""
    for user_id, feature2 in zip(dictionary.keys(), dictionary.values()):
        score = recognizer.match(
            feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= max_score:
            max_score = score
            sim_user_id = user_id
    if max_score < COSINE_THRESHOLD:
        return False, ("", 0.0)
    return True, (sim_user_id, max_score)

def recognize_face(image, face_detector, face_recognizer, file_name=None):
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.shape[0] > 1000:
        image = cv2.resize(image, (0, 0),
                           fx=500 / image.shape[0], fy=500 / image.shape[0])

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    try:
        dts = time.time()
        _, faces = face_detector.detect(image)
        if file_name is not None:
            assert len(faces) > 0, f'the file {file_name} has no face'

        faces = faces if faces is not None else []
        features = []
        for face in faces:
            rts = time.time()

            aligned_face = face_recognizer.alignCrop(image, face)
            feat = face_recognizer.feature(aligned_face)

            features.append(feat)
        return features, faces
    except Exception as e:
        print(e)
        print(file_name)
        return None, None

def main():
    # contain npy for embeddings and registration photos
    directory = 'data'

    # Init models face detection & recognition
    weights = os.path.join(directory, "models", "face_detection_yunet_2023mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    face_detector.setScoreThreshold(0.87)

    weights = os.path.join(directory, "models", "face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # Get registered photos and return as npy files
    dictionary = {}
    for subdir, _, files_in_subdir in os.walk(os.path.join(directory, 'images')):
        user_id = os.path.basename(subdir)
        if not files_in_subdir:
            continue

        for file_name in files_in_subdir:
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(subdir, file_name)
                image = cv2.imread(file_path)
                feats, faces = recognize_face(image, face_detector, face_recognizer, file_path)
                if faces is None:
                    continue

                # Store features for the user ID
                if user_id not in dictionary:
                    dictionary[user_id] = feats[0]
                else:
                    dictionary[user_id] = np.mean([dictionary[user_id], feats[0]], axis=0)

    print(f'there are {len(dictionary)} ids')

    capture = cv2.VideoCapture('test.mp4')
    if not capture.isOpened():
       sys.exit()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (640, 480))

    total_processing_time = 0
    total_frames = 0

    while True:
        start_hand = time.time()  # Start time for frame processing
        result, image = capture.read()
        if result is False:
            break  # End of video

        fetures, faces = recognize_face(image, face_detector, face_recognizer)
        if faces is None:
            continue

        for idx, (face, feature) in enumerate(zip(faces, fetures)):
            result, user = match(face_recognizer, feature, dictionary)
            box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            id_name, score = user if result else (f"unknown_{idx}", 0.0)
            text = "{0} ({1:.2f})".format(id_name, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(image, text, position, font, scale,
                        color, thickness, cv2.LINE_AA)

        # Write the processed frame to the output video
        out.write(image)

        # Show frame
        cv2.imshow("face recognition", image)

        # Calculate the processing time for this frame
        end_hand = time.time()
        frame_time = end_hand - start_hand
        total_processing_time += frame_time
        total_frames += 1

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # After the video ends, calculate and print the average FPS
    if total_processing_time > 0:
        avg_fps = total_frames / total_processing_time
        print(f"Average FPS: {avg_fps:.2f}")
    else:
        print("No frames processed.")

    capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
