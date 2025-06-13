from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load models
face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("models/age_deploy.prototxt", "models/age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("models/gender_deploy.prototxt", "models/gender_net.caffemodel")

AGE_LABELS = ["0-2", "4-6", "8-12", "15-20", "20-25", "25-32", "38-43", "48-53", "60-100"]
GENDER_LABELS = ["Male", "Female"]

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                                           frame.shape[1], frame.shape[0]])
                (x, y, x_max, y_max) = box.astype("int")
                face = frame[y:y_max, x:x_max]

                if face.size == 0:
                    continue

                blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                  (78.4, 87.7, 114.9), swapRB=False)
                gender_net.setInput(blob_face)
                gender = GENDER_LABELS[gender_net.forward()[0].argmax()]

                age_net.setInput(blob_face)
                age = AGE_LABELS[age_net.forward()[0].argmax()]

                label = f"{gender}, {age}"
                cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode image as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)