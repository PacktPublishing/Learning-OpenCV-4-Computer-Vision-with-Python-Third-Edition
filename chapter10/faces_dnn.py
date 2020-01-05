import cv2
import numpy as np


face_model = cv2.dnn.readNetFromCaffe(
    'faces_data/detection/deploy.prototxt',
    'faces_data/detection/res10_300x300_ssd_iter_140000.caffemodel')
face_blob_height = 300
face_average_color = (104, 177, 123)
face_confidence_threshold = 0.995

age_model = cv2.dnn.readNetFromCaffe(
    'faces_data/age_gender_classification/age_net_deploy.prototxt',
    'faces_data/age_gender_classification/age_net.caffemodel')
age_labels = ['0-2', '4-6', '8-12', '15-20',
              '25-32', '38-43', '48-53', '60+']

gender_model = cv2.dnn.readNetFromCaffe(
    'faces_data/age_gender_classification/gender_net_deploy.prototxt',
    'faces_data/age_gender_classification/gender_net.caffemodel')
gender_labels = ['male', 'female']

age_gender_blob_size = (256, 256)
age_gender_average_image = np.load(
    'faces_data/age_gender_classification/average_face.npy')

cap = cv2.VideoCapture(0)

success, frame = cap.read()
while success:

    h, w = frame.shape[:2]
    aspect_ratio = w/h

    # Detect faces in the frame.

    face_blob_width = int(face_blob_height * aspect_ratio)
    face_blob_size = (face_blob_width, face_blob_height)

    face_blob = cv2.dnn.blobFromImage(
        frame, size=face_blob_size, mean=face_average_color)

    face_model.setInput(face_blob)
    face_results = face_model.forward()

    # Iterate over the detected faces.
    for face in face_results[0, 0]:
        face_confidence = face[2]
        if face_confidence > face_confidence_threshold:

            # Get the face coordinates.
            x0, y0, x1, y1 = (face[3:7] * [w, h, w, h]).astype(int)

            # Classify the age and gender of the face based on a
            # square region of interest that includes the neck.

            y1_roi = y0 + int(1.2*(y1-y0))
            x_margin = ((y1_roi-y0) - (x1-x0)) // 2
            x0_roi = x0 - x_margin
            x1_roi = x1 + x_margin
            if x0_roi < 0 or x1_roi > w or y0 < 0 or y1_roi > h:
                # The region of interest is partly outside the
                # frame. Skip this face.
                continue

            age_gender_roi = frame[y0:y1_roi, x0_roi:x1_roi]
            scaled_age_gender_roi = cv2.resize(
                age_gender_roi, age_gender_blob_size,
                interpolation=cv2.INTER_LINEAR).astype(np.float32)
            scaled_age_gender_roi[:] -= age_gender_average_image
            age_gender_blob = cv2.dnn.blobFromImage(
                scaled_age_gender_roi, size=age_gender_blob_size)

            age_model.setInput(age_gender_blob)
            age_results = age_model.forward()
            age_id = np.argmax(age_results)
            age_label = age_labels[age_id]
            age_confidence = age_results[0, age_id]

            gender_model.setInput(age_gender_blob)
            gender_results = gender_model.forward()
            gender_id = np.argmax(gender_results)
            gender_label = gender_labels[gender_id]
            gender_confidence = gender_results[0, gender_id]

            # Draw a blue rectangle around the face.
            cv2.rectangle(frame, (x0, y0), (x1, y1),
                          (255, 0, 0), 2)

            # Draw a yellow square around the region of interest
            # for age and gender classification.
            cv2.rectangle(frame, (x0_roi, y0), (x1_roi, y1_roi),
                          (0, 255, 255), 2)

            # Draw the age and gender classification results.
            text = '%s years (%.1f%%), %s (%.1f%%)' % (
                age_label, age_confidence * 100.0,
                gender_label, gender_confidence * 100.0)
            cv2.putText(frame, text, (x0_roi, y0 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Faces, age, and gender', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Escape
        break

    success, frame = cap.read()
