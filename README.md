# 🎯 Real-Time Face Detection with Age, Gender & Speech Recognition (Desktop App)

This Python desktop application uses your webcam to detect human faces in real-time and predicts their **age** and **gender** using deep learning models. It also supports **speech recognition**, capturing spoken text in the background.

---

## 📦 Features

✅ Real-time **face detection** using OpenCV DNN  
✅ Accurate **age and gender prediction** using pre-trained Caffe models  
✅ Live **speech-to-text recognition** via microphone input  
✅ **Threading** support for simultaneous video and audio processing  
✅ Modern **Tkinter GUI** for desktop experience  
✅ Optionally build a `.exe` using PyInstaller

---

## 🖥️ Technologies Used

- Python 3.8+  
- OpenCV (Deep Neural Network module)  
- Tkinter (GUI)  
- NumPy  
- Pillow (for image conversion)  
- SpeechRecognition (Google Web Speech API)  
- Threading (Python Standard Library)

---

## 📁 Folder Structure

```

D:.
│   desktop\_app.py
│   app.py
│   facedetection.py
│   README.md
│   requirements.txt
│
├───models
│       age\_deploy.prototxt
│       age\_net.caffemodel
│       deploy.prototxt
│       gender\_deploy.prototxt
│       gender\_net.caffemodel
│       res10\_300x300\_ssd\_iter\_140000.caffemodel

````

---

## 🔧 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/face-age-gender-recognition.git
cd face-age-gender-recognition
````

2. **Create a Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Download Pre-trained Models**

Ensure the following models are inside the `/models` folder:

* Face detection: `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`
* Age prediction: `age_deploy.prototxt`, `age_net.caffemodel`
* Gender prediction: `gender_deploy.prototxt`, `gender_net.caffemodel`

---

## ▶️ Run the App

```bash
python desktop_app.py
```

The GUI window will open. Click **Start Camera** to begin.

---

## 🛠 Build Executable (Optional)

Using [PyInstaller](https://www.pyinstaller.org/):

```bash
pyinstaller --onefile --noconsole desktop_app.py
```

The `.exe` will be created inside the `dist/` directory.

---

## 📌 Notes

* Make sure your camera and microphone are accessible.
* Age ranges are approximated based on pre-trained model output.
* Internet is required for speech recognition (Google Speech API).

---

## 🙋‍♂️ Author

**Dhananjay Pratap Singh**
📧 [pratapsinghd665@gmail.com](mailto:pratapsinghd665@gmail.com) (replace with your email)