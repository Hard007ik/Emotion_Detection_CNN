import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image  import img_to_array

# loading model
model = tf.keras.models.load_model('./Emotion_Detection/ResNet50_Transfer_Learning/ResNet50_Emotion.keras')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start capturing video from the webcam (device 0 by default)
cap = cv2.VideoCapture(0)

# Continuous loop for live video feed
while True:
    # Read each frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # convert rgb image into gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detects multiple faces in an image.
    faces = face_classifier.detectMultiScale(gray,
                                             scaleFactor=1.1, # Specifies how much the image is scaled down during face detection
                                             minNeighbors=5,# Controls how many neighboring rectangles must be found for a face to be considered valid
                                             minSize=(30,30),# The minimum size of a detected face (in pixels)
                                             flags=cv2.CASCADE_SCALE_IMAGE # Ensures that the image is rescaled properly during face detection
                                             )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = gray[y:y + h, x:x + w]

        # Ensure face is in RGB (3 channels)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB if needed

        face = cv2.resize(face, (224, 224))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)[0]
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print('done')