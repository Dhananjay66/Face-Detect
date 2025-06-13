import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# Get the base directory where the script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models with full absolute paths
face_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODELS_DIR, "deploy.prototxt"),
    os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
)
age_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODELS_DIR, "age_deploy.prototxt"),
    os.path.join(MODELS_DIR, "age_net.caffemodel")
)
gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODELS_DIR, "gender_deploy.prototxt"),
    os.path.join(MODELS_DIR, "gender_net.caffemodel")
)

GENDER_LABELS = ["Male", "Female"]
# Original labels from model (cannot be changed unless retrained)
ORIGINAL_AGE_LABELS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(20-25)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# Custom remapped labels (just visual)
AGE_LABELS = ["Infant (1-3)", "Toddler (4-6)", "Child (7-11)", "Teen (13-19)", 
              "Young Adult (20-24)", "Adult (25-31)", "Middle Age (35-45)", 
              "Older Adult (46-55)", "Senior (60+)"]

cap = cv2.VideoCapture(0)

def detect_and_display():
    ret, frame = cap.read()
    if not ret:
        return

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

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl.imgtk = imgtk
    lbl.configure(image=imgtk)
    lbl.after(10, detect_and_display)

# GUI setup
root = tk.Tk()
root.title("Real-time Age & Gender Detection")
root.geometry("800x600")
root.configure(bg="#1e1e2f")

# Title Label
title = tk.Label(root, text="Live Age & Gender Detection", font=("Helvetica", 20, "bold"), fg="white", bg="#1e1e2f")
title.pack(pady=10)

# Frame Label
lbl = tk.Label(root, bg="#1e1e2f")
lbl.pack(pady=20)

# Styled Button
btn_style = {"font": ("Helvetica", 14), "bg": "#00b894", "fg": "white", "activebackground": "#00cec9", "activeforeground": "black"}
btn = tk.Button(root, text="Start Camera", command=detect_and_display, **btn_style)
btn.pack(pady=10)

# Run the app
root.mainloop()
cap.release()
cv2.destroyAllWindows()
