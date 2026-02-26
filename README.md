# 🖐️ AI Gesture & Voice Desktop Controller

An AI-powered desktop automation system that allows users to control their computer using **hand gestures** and **voice commands**.

Built using **OpenCV**, **MediaPipe**, and **SpeechRecognition**, this project demonstrates real-time computer vision, voice processing, and human-computer interaction.

---

## 🚀 Features

### 🎮 Gesture Control
- Move mouse using index finger
- Left-click using thumb + index pinch
- Smooth cursor tracking
- Real-time webcam processing

### 🎤 Voice Control
- Open browser using voice
- Launch calculator
- Stop voice control via command
- Hands-free desktop interaction

### 🧠 Architecture Highlights
- Modular project structure
- Multithreaded execution
- Clean separation of gesture and voice controllers
- Logging support
- Scalable configuration design

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe**
- **SpeechRecognition**
- **PyAutoGUI**
- **Tkinter**
- **Multithreading**

---

## ⚙️ Installation

> ⚠️ Recommended Python Version: 3.10

### 1️⃣ Clone Repository

```bash
git clone https://github.com/jiya-19/AI-Gesture-Mouse-Controller.git
cd AI-Gesture-Mouse-Controller
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Application

```bash
python main.py
```

---

## 🖥️ How It Works

### Gesture Pipeline

```
Webcam → MediaPipe Hand Tracking → Landmark Detection → Cursor Mapping → OS Control
```

### Voice Pipeline

```
Microphone → Speech Recognition → Command Processing → System Execution
```

---

## 🎯 Use Cases

* Touch-free computer control
* Accessibility enhancement
* Human-Computer Interaction research
* AI-based automation demonstrations
* Computer Vision portfolio projects

---

## 🔐 Limitations

* Currently optimized for Windows
* Requires webcam and microphone
* MediaPipe compatibility may require Python 3.10

---

## 🔮 Future Improvements

* Cross-platform support (Mac/Linux)
* Custom gesture training
* NLP-based voice intent detection
* PyQt-based advanced UI
* Model optimization for lower latency

---

## 🤝 Contributing

Contributions are welcome.
Feel free to open issues or submit pull requests.

---

## 📜 License

This project is licensed under the MIT License.
