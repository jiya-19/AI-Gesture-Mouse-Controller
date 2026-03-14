# 🖐️ AI Gesture & Voice Desktop Controller

> Touch-free computer control powered by deep learning hand gesture recognition, real-time voice commands, and full speech-to-text dictation.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?style=flat-square)

---

## 📖 Overview

This project enables users to control their entire desktop computer **without touching a mouse or keyboard**. It uses three simultaneous AI-powered input modes:

| Mode | Technology | What It Does |
|---|---|---|
| 🖐️ **Gesture Control** | MediaPipe Hands DL | Moves cursor, clicks, scrolls using hand shapes |
| 🎤 **Voice Commands** | Google Speech API | Executes 20+ system actions by spoken command |
| ⌨️ **Voice Typing** | SpeechRecognition | Types text into any application via dictation |

All three modes run on **separate threads simultaneously** — zero blocking, zero lag.

---

## ✨ Features

### 🖐️ Hand Gesture Control (7 Gestures)

| Gesture | Hand Shape | Action |
|---|---|---|
| Move Cursor | ☝ Index finger only | Moves mouse to finger position |
| Hover | ✌ Index + Middle up | Moves mouse, no click |
| Left Click | 🤏 Thumb + Index pinch | Single left click |
| Right Click | 🤌 Thumb + Middle pinch | Single right click |
| Double Click | 🤘 Index + Pinky up | Double left click |
| Scroll Up | ✋ Open hand (all fingers) | Scrolls page upward |
| Scroll Down | ✊ Closed fist | Scrolls page downward |

### 🎤 Voice Commands (20 Commands)

| Command | Action |
|---|---|
| `"open browser"` | Launch default browser |
| `"open calculator"` | Open Windows Calculator |
| `"open notepad"` | Open Notepad |
| `"take screenshot"` | Save timestamped screenshot |
| `"scroll up"` / `"scroll down"` | Scroll active window |
| `"zoom in"` / `"zoom out"` | Ctrl+`+` / Ctrl+`-` |
| `"go back"` / `"go forward"` | Browser navigation |
| `"new tab"` / `"close tab"` | Tab management |
| `"close window"` | Alt+F4 |
| `"copy"` / `"paste"` / `"undo"` | Clipboard & history |
| `"select all"` | Ctrl+A |
| `"minimize"` / `"maximize"` | Window state |
| `"switch window"` | Alt+Tab |
| `"stop voice"` | Stop voice listener |

### ⌨️ Voice Typing — Full Dictation Mode

Speak naturally and the AI types every word into the active text field — in any app (Notepad, browser, Word, chat, etc.).

**Spoken Punctuation:**

| Say | Types | Say | Types |
|---|---|---|---|
| `"full stop"` | `.` | `"comma"` | `,` |
| `"question mark"` | `?` | `"exclamation"` | `!` |
| `"colon"` | `:` | `"semicolon"` | `;` |
| `"open bracket"` | `(` | `"close bracket"` | `)` |
| `"at sign"` | `@` | `"asterisk"` | `*` |
| `"dash"` | `-` | `"underscore"` | `_` |
| `"slash"` | `/` | `"equals"` | `=` |
| `"open quote"` | `"` | `"apostrophe"` | `'` |

**Dictation Control Commands:**

| Say | Action |
|---|---|
| `"new line"` | Press Enter |
| `"delete that"` | Delete last word |
| `"clear all"` | Select all + delete |
| `"capital"` | Capitalise next word |
| `"tab"` | Press Tab key |
| `"undo"` | Ctrl+Z |
| `"stop typing"` | Exit dictation mode |

---

## 🛠️ Technology Stack

| Library | Version | Role |
|---|---|---|
| **MediaPipe** | 0.10.14 | Deep learning hand tracking, 21 landmarks, model_complexity=1 |
| **OpenCV** | 4.9+ | Camera capture, frame processing, HUD overlay |
| **PyAutoGUI** | 0.9.54+ | OS-level mouse & keyboard automation |
| **SpeechRecognition** | 3.10+ | Google Speech API for voice input |
| **PyAudio** | 0.2.14+ | Microphone stream with ambient noise calibration |
| **NumPy** | 1.26+ | Smoothing buffer for jitter-free cursor |
| **Tkinter** | stdlib | Control panel GUI with live activity log |
| **threading + queue** | stdlib | Concurrent processing, zero blocking |

---

## ⚙️ Installation

> ⚠️ **Required: Python 3.12** — MediaPipe 0.10.14 is optimised for Python 3.12

### Step 1 — Fix Dependency Conflict (IMPORTANT — do this first)

If you have TensorFlow installed, it conflicts with MediaPipe's protobuf version. Run these commands once to fix it:

```bash
pip uninstall tensorflow tensorflow-intel mediapipe protobuf -y
pip install protobuf==3.20.3
pip install mediapipe==0.10.14
```

Or simply double-click **`fix_and_install.bat`** (included in the repo).

### Step 2 — Install All Dependencies

```bash
pip install opencv-python pyautogui numpy SpeechRecognition pyaudio
```

**Full one-liner:**
```bash
pip install protobuf==3.20.3 mediapipe==0.10.14 opencv-python pyautogui numpy SpeechRecognition pyaudio
```

### Step 3 — Run

```bash
python ai_gesture_controller.py
```

---

## 🚀 Usage

1. **Run the script** — the control panel window opens
2. Click **▶ START GESTURE** — your camera feed opens in a separate window
3. Click **🎤 START VOICE COMMANDS** — speak commands to control your system
4. Click **⌨️ START VOICE TYPING** — click into any text field and start dictating
5. Press **Q** or **ESC** in the camera window to stop gesture control
6. Click **⏹ EXIT** to close everything cleanly

### Tips for Best Gesture Recognition
- Keep your hand **30–60 cm** from the webcam
- Ensure **good lighting** on your hand
- Face your palm **toward the camera**
- Make gestures **deliberately** — hold each shape for ~0.3 seconds

---

## 🏗️ Architecture

```
ai_gesture_controller.py
│
├── Config              — All tunable parameters in one place
├── GestureDetector     — MediaPipe DL hand tracking + gesture classifier
│   ├── _fingers_up()   — Determine which fingers are extended
│   ├── _classify()     — Map finger states → gesture name
│   └── process()       — Per-frame: detect → classify → execute mouse action
│
├── HUDRenderer         — OpenCV overlay: guide panel, status dots, FPS, badge
│
├── CameraThread        — Dedicated thread: low-latency frame capture (CAP_DSHOW)
│
├── VoiceController     — Background thread: Google Speech → system commands
│   └── _execute()      — Tokenised command matching against 20-command dict
│
├── VoiceTypist         — Background thread: Google Speech → typed text
│   └── _process()      — Word-by-word parser: punctuation / control / literal
│
└── App                 — Tkinter GUI: buttons, status labels, live activity log
    ├── _g_loop()        — OpenCV display loop (gesture thread)
    ├── _watch_typist()  — Monitors if user said "stop typing"
    └── _poll_log()      — Drains log queue into GUI text widget (150ms tick)
```

### Threading Model

```
Main Thread      →  Tkinter GUI event loop
CameraThread     →  cv2.VideoCapture reads (daemon)
_g_loop thread   →  MediaPipe process + cv2.imshow (daemon)
VoiceController  →  sr.Microphone listen loop (daemon)
VoiceTypist      →  sr.Microphone listen + typewrite (daemon)
```

All inter-thread communication uses `queue.Queue` — thread-safe, non-blocking.

---

## 🔧 Configuration

All tuning parameters are in the `Config` class at the top of the file:

```python
class Config:
    CAMERA_INDEX         = 0      # Change if you have multiple cameras
    DETECTION_CONFIDENCE = 0.80   # Lower = more sensitive, more false positives
    TRACKING_CONFIDENCE  = 0.80   # Lower = smoother but less accurate tracking
    MODEL_COMPLEXITY     = 1      # 0 = lite/fast, 1 = full accuracy
    SMOOTH_BUFFER        = 6      # Higher = smoother cursor, more lag
    CLICK_THRESHOLD      = 0.045  # Smaller = need tighter pinch to click
    MOUSE_SPEED          = 1.4    # Higher = cursor moves faster
    SCROLL_SPEED         = 15     # Scroll amount per gesture
    CLICK_COOLDOWN       = 0.35   # Seconds between allowed clicks
```

---

## 🐛 Troubleshooting

### `ImportError: cannot import name 'runtime_version' from 'google.protobuf'`
**Cause:** TensorFlow and MediaPipe conflict over protobuf versions.  
**Fix:** Run `fix_and_install.bat` or:
```bash
pip uninstall tensorflow tensorflow-intel mediapipe protobuf -y
pip install protobuf==3.20.3 mediapipe==0.10.14
```

### Camera not opening / black screen
- Check another app isn't using your webcam
- Try changing `CAMERA_INDEX = 1` in Config if you have multiple cameras

### Voice not working
- Ensure `SpeechRecognition` and `pyaudio` are installed
- Check microphone permissions in Windows Settings → Privacy → Microphone
- Requires an internet connection (Google Speech API)

### Cursor is too jittery
- Increase `SMOOTH_BUFFER` from 6 to 8–10
- Improve lighting on your hand
- Reduce `MOUSE_SPEED` slightly

### Gestures triggering accidentally
- Increase `CLICK_THRESHOLD` from 0.045 to 0.055
- Increase `CLICK_COOLDOWN` from 0.35 to 0.5

---

## 📋 Requirements

```
opencv-python>=4.9.0
mediapipe==0.10.14
pyautogui>=0.9.54
numpy>=1.26.0
SpeechRecognition>=3.10.4
pyaudio>=0.2.14
protobuf==3.20.3
```

---

## 🔮 Future Roadmap

- [ ] Cross-platform support (macOS / Linux)
- [ ] Custom gesture training with user-defined hand shapes
- [ ] Offline speech recognition (Whisper / Vosk)
- [ ] NLP intent detection for natural language commands
- [ ] Multi-hand support (two-hand gestures)
- [ ] Gesture macro recording and playback
- [ ] PyQt6 advanced UI with settings panel

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
