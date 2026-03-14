import os, sys, types

os.environ["TF_CPP_MIN_LOG_LEVEL"]               = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]              = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

class _Stub(types.ModuleType):
    def __getattr__(self, name):
        child = _Stub(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child
    def __call__(self, *a, **kw):
        return a[0] if a else None

def _stub(name):
    m = _Stub(name)
    sys.modules[name] = m
    return m

_tf_spec = None
try:
    import importlib.util as _ilu
    _tf_spec = _ilu.find_spec("tensorflow")
except Exception:
    pass

if _tf_spec is not None:
    for _name in (
        "tensorflow",
        "tensorflow.tools",
        "tensorflow.tools.docs",
        "tensorflow.tools.docs.doc_controls",
        "tensorflow._api",
        "tensorflow.python",
    ):
        if _name not in sys.modules:
            _stub(_name)
    _dc = sys.modules.setdefault(
        "tensorflow.tools.docs.doc_controls", _Stub("tensorflow.tools.docs.doc_controls"))
    _dc.do_not_generate_docs = lambda f: f
    _dc.do_not_doc_in_subclasses = lambda f: f

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import threading
import logging
import time
import math
import subprocess
import queue
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from collections import deque

try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AIController")

class Config:
    CAMERA_INDEX         = 0
    FRAME_WIDTH          = 1280
    FRAME_HEIGHT         = 720
    CAMERA_FPS           = 30
    MAX_HANDS            = 1
    DETECTION_CONFIDENCE = 0.80
    TRACKING_CONFIDENCE  = 0.80
    MODEL_COMPLEXITY     = 1
    SMOOTH_BUFFER        = 6
    CLICK_THRESHOLD      = 0.045
    SCROLL_SPEED         = 15
    MOUSE_SPEED          = 1.4
    SCREEN_W, SCREEN_H   = pyautogui.size()
    pyautogui.FAILSAFE   = False
    pyautogui.PAUSE      = 0.0
    CLICK_COOLDOWN        = 0.35
    RIGHT_CLICK_COOLDOWN  = 0.35
    DOUBLE_CLICK_COOLDOWN = 0.55
    SCROLL_COOLDOWN       = 0.10
    C_GREEN  = (0,  230, 100)
    C_BLUE   = (255, 140,   0)
    C_RED    = (0,   60, 220)
    C_YELLOW = (0,  220, 220)
    C_WHITE  = (255, 255, 255)
    C_PURPLE = (200,   0, 200)
    C_CYAN   = (220, 200,   0)
    C_GREY   = (120, 120, 120)

class LM:
    THUMB_IP   = 3
    THUMB_TIP  = 4
    INDEX_PIP  = 6
    INDEX_TIP  = 8
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_PIP   = 14
    RING_TIP   = 16
    PINKY_PIP  = 18
    PINKY_TIP  = 20

class GestureDetector:
    def __init__(self):
        self._mp_hands  = mp.solutions.hands
        self._mp_draw   = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self.hands = self._mp_hands.Hands(
            static_image_mode        = False,
            max_num_hands            = Config.MAX_HANDS,
            min_detection_confidence = Config.DETECTION_CONFIDENCE,
            min_tracking_confidence  = Config.TRACKING_CONFIDENCE,
            model_complexity         = Config.MODEL_COMPLEXITY,
        )
        self._xbuf    = deque(maxlen=Config.SMOOTH_BUFFER)
        self._ybuf    = deque(maxlen=Config.SMOOTH_BUFFER)
        self._t_click  = self._t_rclick = self._t_dclick = self._t_scroll = 0.0
        self.gesture      = "No Hand"
        self.hand_visible = False
        logger.info("GestureDetector ready [model_complexity=%d]",
                    Config.MODEL_COMPLEXITY)
    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)
    def _fingers_up(self, lm) -> list:
        up = [lm[LM.THUMB_TIP].x < lm[LM.THUMB_IP].x]
        for tip, pip in (
            (LM.INDEX_TIP,  LM.INDEX_PIP),
            (LM.MIDDLE_TIP, LM.MIDDLE_PIP),
            (LM.RING_TIP,   LM.RING_PIP),
            (LM.PINKY_TIP,  LM.PINKY_PIP),
        ):
            up.append(lm[tip].y < lm[pip].y)
        return up
    def _classify(self, lm) -> str:
        f  = self._fingers_up(lm)
        ti = self._dist(lm[LM.THUMB_TIP], lm[LM.INDEX_TIP])
        tm = self._dist(lm[LM.THUMB_TIP], lm[LM.MIDDLE_TIP])
        _, index, middle, ring, pinky = f
        if all(f):                                                    return "SCROLL_UP"
        if not any(f):                                                return "SCROLL_DOWN"
        if ti < Config.CLICK_THRESHOLD and not middle and not ring and not pinky: return "LEFT_CLICK"
        if tm < Config.CLICK_THRESHOLD and not index  and not ring and not pinky: return "RIGHT_CLICK"
        if index and pinky and not middle and not ring:               return "DOUBLE_CLICK"
        if index and middle and not ring  and not pinky:              return "HOVER"
        if index and not middle and not ring and not pinky:           return "MOVE"
        return "UNKNOWN"
    def _smooth(self, x: float, y: float) -> tuple:
        self._xbuf.append(x)
        self._ybuf.append(y)
        return int(np.mean(self._xbuf)), int(np.mean(self._ybuf))
    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res   = self.hands.process(rgb)
        rgb.flags.writeable = True
        self.gesture = "No Hand"
        self.hand_visible = False
        if res.multi_hand_landmarks:
            self.hand_visible = True
            hlm = res.multi_hand_landmarks[0]
            lm  = hlm.landmark
            self._mp_draw.draw_landmarks(
                frame, hlm,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )
            self.gesture = self._classify(lm)
            now = time.time()
            tip = lm[LM.INDEX_TIP]
            rx  = max(0, min(Config.SCREEN_W - 1,
                             int((1.0 - tip.x) * Config.SCREEN_W * Config.MOUSE_SPEED)))
            ry  = max(0, min(Config.SCREEN_H - 1,
                             int(tip.y          * Config.SCREEN_H * Config.MOUSE_SPEED)))
            sx, sy = self._smooth(rx, ry)
            g = self.gesture
            if g in ("MOVE", "HOVER"):
                pyautogui.moveTo(sx, sy)
            elif g == "LEFT_CLICK":
                pyautogui.moveTo(sx, sy)
                if now - self._t_click > Config.CLICK_COOLDOWN:
                    pyautogui.click(); self._t_click = now
            elif g == "RIGHT_CLICK":
                pyautogui.moveTo(sx, sy)
                if now - self._t_rclick > Config.RIGHT_CLICK_COOLDOWN:
                    pyautogui.rightClick(); self._t_rclick = now
            elif g == "DOUBLE_CLICK":
                pyautogui.moveTo(sx, sy)
                if now - self._t_dclick > Config.DOUBLE_CLICK_COOLDOWN:
                    pyautogui.doubleClick(); self._t_dclick = now
            elif g == "SCROLL_UP":
                if now - self._t_scroll > Config.SCROLL_COOLDOWN:
                    pyautogui.scroll(Config.SCROLL_SPEED); self._t_scroll = now
            elif g == "SCROLL_DOWN":
                if now - self._t_scroll > Config.SCROLL_COOLDOWN:
                    pyautogui.scroll(-Config.SCROLL_SPEED); self._t_scroll = now
        return frame
    def release(self):
        self.hands.close()

class HUDRenderer:
    GUIDE = [
        ("Index only",   "Move Cursor",  Config.C_GREEN),
        ("Index+Middle", "Hover",        Config.C_CYAN),
        ("Thumb+Index",  "Left Click",   Config.C_YELLOW),
        ("Thumb+Middle", "Right Click",  Config.C_PURPLE),
        ("Index+Pinky",  "Double Click", Config.C_RED),
        ("Open Hand",    "Scroll Up",    Config.C_BLUE),
        ("Fist",         "Scroll Down",  Config.C_RED),
    ]
    GCOLOR = {
        "MOVE": Config.C_GREEN, "HOVER": Config.C_CYAN,
        "LEFT_CLICK": Config.C_YELLOW, "RIGHT_CLICK": Config.C_PURPLE,
        "DOUBLE_CLICK": Config.C_RED,
        "SCROLL_UP": Config.C_BLUE, "SCROLL_DOWN": Config.C_RED,
    }
    def render(self, frame, gesture, fps, voice_on, hand_visible, typing_on=False):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 58), (10, 10, 20), -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, Config.C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "AI GESTURE CONTROLLER",
                    (w // 2 - 195, 38), cv2.FONT_HERSHEY_DUPLEX, 0.95, Config.C_GREEN, 2, cv2.LINE_AA)
        fps_c = Config.C_GREEN if fps >= 20 else Config.C_YELLOW
        cv2.putText(frame, f"FPS: {fps}", (w - 125, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, fps_c, 2, cv2.LINE_AA)
        hc = Config.C_GREEN if hand_visible else Config.C_RED
        cv2.circle(frame, (18, 78), 8, hc, -1)
        cv2.putText(frame, "HAND DETECTED" if hand_visible else "NO HAND",
                    (32, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hc, 1, cv2.LINE_AA)
        vc = Config.C_GREEN if voice_on else Config.C_GREY
        cv2.circle(frame, (18, 108), 8, vc, -1)
        cv2.putText(frame, "VOICE CMDS ON" if voice_on else "VOICE CMDS OFF",
                    (32, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vc, 1, cv2.LINE_AA)
        tc = (200, 0, 200) if typing_on else Config.C_GREY
        cv2.circle(frame, (18, 138), 8, tc, -1)
        cv2.putText(frame, "VOICE TYPING ON" if typing_on else "VOICE TYPING OFF",
                    (32, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 1, cv2.LINE_AA)
        px, py = w - 295, 68
        ph = len(self.GUIDE) * 28 + 44
        ov2 = frame.copy()
        cv2.rectangle(ov2, (px - 8, py), (w - 4, py + ph), (10, 10, 20), -1)
        cv2.addWeighted(ov2, 0.68, frame, 0.32, 0, frame)
        cv2.putText(frame, "GESTURE GUIDE", (px, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.C_CYAN, 1, cv2.LINE_AA)
        cv2.line(frame, (px - 4, py + 28), (w - 8, py + 28), Config.C_CYAN, 1)
        for i, (sign, action, color) in enumerate(self.GUIDE):
            y = py + 50 + i * 28
            cv2.putText(frame, sign,   (px,       y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, Config.C_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, action, (px + 145, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, color, 1, cv2.LINE_AA)
        if gesture not in ("No Hand", "UNKNOWN", "None", ""):
            color = self.GCOLOR.get(gesture, Config.C_WHITE)
            label = f"Active: {gesture}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
            bx = w // 2 - tw // 2 - 12
            ov3 = frame.copy()
            cv2.rectangle(ov3, (bx, h - 58), (bx + tw + 24, h - 18), (10, 10, 20), -1)
            cv2.addWeighted(ov3, 0.72, frame, 0.28, 0, frame)
            cv2.putText(frame, label, (bx + 12, h - 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)
        return frame

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame_q  = queue.Queue(maxsize=2)
        self._running = False
        self.cap      = None
        self.fps      = 0
        self._fc = 0; self._ft = time.time()
    def run(self):
        self._running = True
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          Config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not self.cap.isOpened():
            logger.error("Cannot open camera %d", Config.CAMERA_INDEX); return
        logger.info("Camera %dx%d @ %dfps",
                    Config.FRAME_WIDTH, Config.FRAME_HEIGHT, Config.CAMERA_FPS)
        while self._running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            if not self.frame_q.full():
                self.frame_q.put(frame)
            self._fc += 1
            now = time.time()
            if now - self._ft >= 1.0:
                self.fps = self._fc; self._fc = 0; self._ft = now
    def stop(self):
        self._running = False
        if self.cap: self.cap.release()

class VoiceController:
    COMMANDS = {
        "open browser":    lambda: subprocess.Popen(["start", "https://google.com"], shell=True),
        "open calculator": lambda: subprocess.Popen("calc",    shell=True),
        "open notepad":    lambda: subprocess.Popen("notepad", shell=True),
        "scroll up":       lambda: pyautogui.scroll(20),
        "scroll down":     lambda: pyautogui.scroll(-20),
        "take screenshot": lambda: pyautogui.screenshot(
            f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
        "zoom in":         lambda: pyautogui.hotkey("ctrl", "+"),
        "zoom out":        lambda: pyautogui.hotkey("ctrl", "-"),
        "go back":         lambda: pyautogui.hotkey("alt", "left"),
        "go forward":      lambda: pyautogui.hotkey("alt", "right"),
        "close window":    lambda: pyautogui.hotkey("alt", "f4"),
        "new tab":         lambda: pyautogui.hotkey("ctrl", "t"),
        "close tab":       lambda: pyautogui.hotkey("ctrl", "w"),
        "copy":            lambda: pyautogui.hotkey("ctrl", "c"),
        "paste":           lambda: pyautogui.hotkey("ctrl", "v"),
        "undo":            lambda: pyautogui.hotkey("ctrl", "z"),
        "select all":      lambda: pyautogui.hotkey("ctrl", "a"),
        "minimize":        lambda: pyautogui.hotkey("win", "down"),
        "maximize":        lambda: pyautogui.hotkey("win", "up"),
        "switch window":   lambda: pyautogui.hotkey("alt", "tab"),
    }
    def __init__(self, log_q: queue.Queue):
        self._log_q   = log_q
        self._running = False
        self._thread  = None
        self.last_cmd = ""
        if VOICE_AVAILABLE:
            self.rec = sr.Recognizer()
            self.rec.energy_threshold = 300
            self.rec.dynamic_energy_threshold = True
            self.rec.pause_threshold = 0.6
    def _log(self, msg):
        self._log_q.put(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    def start(self):
        if not VOICE_AVAILABLE:
            self._log("⚠  Voice unavailable — pip install SpeechRecognition pyaudio"); return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log("🎤  Voice listener started")
    def stop(self):
        self._running = False
        self._log("🔇  Voice listener stopped")
    def _loop(self):
        try:
            with sr.Microphone() as src:
                self.rec.adjust_for_ambient_noise(src, duration=1.0)
                self._log("🔊  Mic calibrated — listening…")
                while self._running:
                    try:
                        audio = self.rec.listen(src, timeout=3, phrase_time_limit=5)
                        text  = self.rec.recognize_google(audio).lower().strip()
                        self._log(f"🗣  \"{text}\"")
                        self._execute(text)
                    except sr.WaitTimeoutError:
                        pass
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        self._log(f"⚠  API: {e}")
        except Exception as e:
            self._log(f"⚠  Mic: {e}")
    def _execute(self, text):
        if any(k in text for k in ("exit", "quit", "stop controller")):
            self.stop(); return
        if "stop voice" in text:
            self.stop(); return
        for cmd, action in self.COMMANDS.items():
            if cmd in text:
                self._log(f"✅  → {cmd}")
                self.last_cmd = cmd
                try: action()
                except Exception as e: self._log(f"⚠  {cmd}: {e}")
                return
        self._log(f"❓  Unknown: \"{text}\"")

class VoiceTypist:
    _PUNCT = {
        "full stop":       ".",
        "period":          ".",
        "comma":           ",",
        "question mark":   "?",
        "exclamation mark":"!",
        "exclamation":     "!",
        "colon":           ":",
        "semicolon":       ";",
        "open bracket":    "(",
        "close bracket":   ")",
        "dash":            "-",
        "hyphen":          "-",
        "underscore":      "_",
        "at sign":         "@",
        "hash":            "#",
        "percent":         "%",
        "ampersand":       "&",
        "asterisk":        "*",
        "equals":          "=",
        "plus":            "+",
        "slash":           "/",
        "backslash":       "\\",
        "open quote":      '"',
        "close quote":     '"',
        "apostrophe":      "'",
    }
    _CONTROL = {
        "new line":        lambda: pyautogui.press("enter"),
        "enter":           lambda: pyautogui.press("enter"),
        "delete that":     lambda: (pyautogui.hotkey("ctrl", "shift", "left"),
                                    pyautogui.press("backspace")),
        "backspace":       lambda: pyautogui.press("backspace"),
        "clear all":       lambda: (pyautogui.hotkey("ctrl", "a"),
                                    pyautogui.press("delete")),
        "tab":             lambda: pyautogui.press("tab"),
        "space":           lambda: pyautogui.press("space"),
        "undo":            lambda: pyautogui.hotkey("ctrl", "z"),
        "capital":         None,
    }
    def __init__(self, log_q: queue.Queue):
        self._log_q      = log_q
        self._running    = False
        self._thread     = None
        self._capitalise = False
        self.is_active   = False
        if VOICE_AVAILABLE:
            self.rec = sr.Recognizer()
            self.rec.energy_threshold         = 250
            self.rec.dynamic_energy_threshold = True
            self.rec.pause_threshold          = 0.5
    def _log(self, msg: str):
        self._log_q.put(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    def start(self):
        if not VOICE_AVAILABLE:
            self._log("⚠  Voice Typing unavailable — install SpeechRecognition & pyaudio")
            return
        self._running  = True
        self.is_active = True
        self._thread   = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log("⌨️   Voice Typing STARTED — speak to type")
        self._log("     Say 'stop typing' to exit dictation mode")
    def stop(self):
        self._running  = False
        self.is_active = False
        self._log("⌨️   Voice Typing STOPPED")
    def _loop(self):
        try:
            with sr.Microphone() as src:
                self.rec.adjust_for_ambient_noise(src, duration=0.8)
                self._log("🎙  Dictation ready — start speaking…")
                while self._running:
                    try:
                        audio = self.rec.listen(src, timeout=4, phrase_time_limit=8)
                        raw   = self.rec.recognize_google(audio).strip()
                        self._log(f"📝  Heard: \"{raw}\"")
                        self._process(raw)
                    except sr.WaitTimeoutError:
                        pass
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        self._log(f"⚠  Speech API: {e}")
        except Exception as e:
            self._log(f"⚠  Dictation mic error: {e}")
        self.is_active = False
    def _process(self, raw: str):
        lower = raw.lower().strip()
        if lower in ("stop typing", "stop dictation", "exit typing"):
            self.stop()
            return
        if lower in self._CONTROL:
            if lower == "capital":
                self._capitalise = True
                self._log("     (next word will be capitalised)")
            else:
                try:
                    result = self._CONTROL[lower]()
                except Exception as e:
                    self._log(f"⚠  Control: {e}")
            return
        if lower in self._PUNCT:
            pyautogui.typewrite(self._PUNCT[lower], interval=0.02)
            return
        words        = raw.split()
        lower_words  = lower.split()
        typed_any    = False
        i = 0
        while i < len(lower_words):
            w_low = lower_words[i]
            two = " ".join(lower_words[i:i+2]) if i + 1 < len(lower_words) else ""
            if two in self._CONTROL:
                if typed_any:
                    pyautogui.press("space")
                if two == "capital":
                    self._capitalise = True
                else:
                    try: self._CONTROL[two]()
                    except: pass
                i += 2
                typed_any = False
                continue
            if two in self._PUNCT:
                pyautogui.typewrite(self._PUNCT[two], interval=0.02)
                i += 2
                typed_any = True
                continue
            if w_low in self._CONTROL:
                if typed_any: pyautogui.press("space")
                if w_low == "capital":
                    self._capitalise = True
                else:
                    try: self._CONTROL[w_low]()
                    except: pass
                i += 1
                typed_any = False
                continue
            if w_low in self._PUNCT:
                pyautogui.typewrite(self._PUNCT[w_low], interval=0.02)
                i += 1
                typed_any = True
                continue
            word = words[i]
            if self._capitalise:
                word = word.capitalize()
                self._capitalise = False
            if typed_any:
                pyautogui.press("space")
            try:
                pyautogui.typewrite(word, interval=0.03)
            except Exception:
                import pyperclip
                pyperclip.copy(word)
                pyautogui.hotkey("ctrl", "v")
            typed_any = True
            i += 1

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Gesture & Voice Controller")
        self.root.resizable(False, False)
        self.root.configure(bg="#0d0d1a")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self._g_active  = False
        self._v_active  = False
        self._vt_active = False
        self._running   = False
        self._log_q     = queue.Queue()
        self._cam       = None
        self._det       = None
        self._hud       = HUDRenderer()
        self._voice     = VoiceController(self._log_q)
        self._typist    = VoiceTypist(self._log_q)
        self._build_ui()
        self._poll_log()
        self._emit("🚀  AI Controller ready")
        self._emit(f"🖥  Screen  {Config.SCREEN_W} × {Config.SCREEN_H}")
        self._emit(f"🎤  Voice   {'Available ✓' if VOICE_AVAILABLE else 'Unavailable ✗'}")
        self._emit(f"⌨️   Voice Typing  {'Available ✓' if VOICE_AVAILABLE else 'Unavailable ✗'}")
        self._emit("─" * 38)
    def _build_ui(self):
        BG = "#0d0d1a"; PANEL = "#16162a"; ACC = "#00e676"; BORD = "#2a2a44"
        hdr = tk.Frame(self.root, bg="#13132a", pady=14); hdr.pack(fill="x")
        tk.Label(hdr, text="🖐  AI GESTURE & VOICE CONTROLLER",
                 font=("Segoe UI", 17, "bold"), bg="#13132a", fg=ACC).pack()
        tk.Label(hdr, text="MediaPipe  ·  OpenCV  ·  SpeechRecognition  |  Python 3.12",
                 font=("Segoe UI", 9), bg="#13132a", fg="#666688").pack()
        body = tk.Frame(self.root, bg=BG, padx=18, pady=14); body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=PANEL, padx=14, pady=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        def sec(t):
            tk.Label(left, text=t, font=("Segoe UI", 10, "bold"), bg=PANEL, fg=ACC
                     ).pack(anchor="w", pady=(10, 4))
            tk.Frame(left, bg=BORD, height=1).pack(fill="x")
        sec("CONTROLS")
        self._btn_g = tk.Button(left, text="▶  START GESTURE",
                                font=("Segoe UI", 11, "bold"),
                                bg="#00c853", fg="white", activebackground="#009624",
                                relief="flat", width=22, pady=8, cursor="hand2",
                                command=self._toggle_g)
        self._btn_g.pack(fill="x", pady=(8, 3))
        self._btn_v = tk.Button(left, text="🎤  START VOICE COMMANDS",
                                font=("Segoe UI", 11, "bold"),
                                bg="#1565c0", fg="white", activebackground="#0d47a1",
                                relief="flat", width=22, pady=8, cursor="hand2",
                                command=self._toggle_v)
        self._btn_v.pack(fill="x", pady=3)
        self._btn_vt = tk.Button(left, text="⌨️   START VOICE TYPING",
                                 font=("Segoe UI", 11, "bold"),
                                 bg="#6a1b9a", fg="white", activebackground="#4a0072",
                                 relief="flat", width=22, pady=8, cursor="hand2",
                                 command=self._toggle_vt)
        self._btn_vt.pack(fill="x", pady=3)
        tk.Frame(left, bg=BORD, height=1).pack(fill="x", pady=10)
        self._lbl_g = tk.Label(left, text="● Gesture:  INACTIVE",
                               font=("Segoe UI", 10), bg=PANEL, fg="#555577")
        self._lbl_g.pack(anchor="w", pady=1)
        self._lbl_v = tk.Label(left, text="● Voice Cmds: INACTIVE",
                               font=("Segoe UI", 10), bg=PANEL, fg="#555577")
        self._lbl_v.pack(anchor="w", pady=1)
        self._lbl_vt = tk.Label(left, text="● Voice Typing: INACTIVE",
                                font=("Segoe UI", 10), bg=PANEL, fg="#555577")
        self._lbl_vt.pack(anchor="w", pady=1)
        sec("GESTURE GUIDE")
        for sign, action, color in HUDRenderer.GUIDE:
            hx = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
            row = tk.Frame(left, bg=PANEL); row.pack(fill="x", pady=1)
            tk.Label(row, text=sign,   width=15, anchor="w",
                     font=("Segoe UI", 9), bg=PANEL, fg="white").pack(side="left")
            tk.Label(row, text=action, anchor="w",
                     font=("Segoe UI", 9), bg=PANEL, fg=hx).pack(side="left")
        sec("VOICE TYPING COMMANDS")
        typing_guide = [
            ('"new line"',      "→ Press Enter"),
            ('"delete that"',   "→ Delete last word"),
            ('"clear all"',     "→ Erase everything"),
            ('"capital"',       "→ Capitalise next word"),
            ('"full stop"',     "→ Type  ."),
            ('"comma"',         "→ Type  ,"),
            ('"question mark"', "→ Type  ?"),
            ('"stop typing"',   "→ Exit dictation"),
        ]
        for spoken, effect in typing_guide:
            row = tk.Frame(left, bg=PANEL); row.pack(fill="x", pady=1)
            tk.Label(row, text=spoken, width=17, anchor="w",
                     font=("Consolas", 8), bg=PANEL, fg="#ce93d8").pack(side="left")
            tk.Label(row, text=effect, anchor="w",
                     font=("Segoe UI", 8), bg=PANEL, fg="#aaaacc").pack(side="left")
        tk.Frame(left, bg=BORD, height=1).pack(fill="x", pady=10)
        tk.Button(left, text="⏹  EXIT APPLICATION",
                  font=("Segoe UI", 11, "bold"),
                  bg="#b71c1c", fg="white", activebackground="#7f0000",
                  relief="flat", width=22, pady=8, cursor="hand2",
                  command=self._quit).pack(fill="x", pady=3)
        right = tk.Frame(body, bg=PANEL, padx=10, pady=10)
        right.grid(row=0, column=1, sticky="nsew")
        tk.Label(right, text="ACTIVITY LOG", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=ACC).pack(anchor="w", pady=(0, 4))
        lf = tk.Frame(right, bg=PANEL); lf.pack(fill="both", expand=True)
        self._txt = tk.Text(lf, width=48, height=32,
                            font=("Consolas", 9), bg="#09091a", fg="#c0c0e0",
                            relief="flat", state="disabled", wrap="word",
                            selectbackground="#2a2a5a")
        self._txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self._txt.yview); sb.pack(side="right", fill="y")
        self._txt.configure(yscrollcommand=sb.set)
        body.columnconfigure(1, weight=1)
        ft = tk.Frame(self.root, bg="#13132a", pady=5); ft.pack(fill="x")
        tk.Label(ft,
                 text="Camera window opens when gesture control is active  |  Press Q or ESC to stop",
                 font=("Segoe UI", 8), bg="#13132a", fg="#44445a").pack()
    def _emit(self, msg):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
        self._txt.configure(state="normal")
        self._txt.insert("end", entry)
        self._txt.see("end")
        self._txt.configure(state="disabled")
    def _poll_log(self):
        try:
            while True: self._emit(self._log_q.get_nowait())
        except queue.Empty: pass
        self.root.after(150, self._poll_log)
    def _toggle_g(self):
        if not self._g_active:
            self._g_active = True
            self._btn_g.configure(text="⏹  STOP GESTURE", bg="#b71c1c")
            self._lbl_g.configure(text="● Gesture:  ACTIVE ✓", fg="#00e676")
            self._emit("▶  Gesture controller starting…")
            self._det  = GestureDetector()
            self._cam  = CameraThread(); self._cam.start()
            self._running = True
            threading.Thread(target=self._g_loop, daemon=True).start()
        else:
            self._stop_g()
    def _stop_g(self):
        self._running = False; self._g_active = False
        self._btn_g.configure(text="▶  START GESTURE", bg="#00c853")
        self._lbl_g.configure(text="● Gesture:  INACTIVE", fg="#555577")
        self._emit("⏹  Gesture controller stopped")
        if self._cam:  self._cam.stop()
        if self._det:  self._det.release()
        cv2.destroyAllWindows()
    def _g_loop(self):
        WIN = "AI Gesture Controller — Camera Feed  [Q / ESC to stop]"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 960, 540)
        self._emit("📷  Camera feed open")
        while self._running:
            try:
                frame = self._cam.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            frame = self._det.process(frame)
            frame = self._hud.render(frame, self._det.gesture,
                                     self._cam.fps, self._v_active,
                                     self._det.hand_visible,
                                     self._vt_active)
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.root.after(0, self._stop_g); break
        cv2.destroyAllWindows()
    def _toggle_v(self):
        if not self._v_active:
            self._v_active = True
            self._btn_v.configure(text="🔇  STOP VOICE COMMANDS", bg="#b71c1c")
            self._lbl_v.configure(text="● Voice Cmds: ACTIVE ✓", fg="#00e676")
            self._voice.start()
        else:
            self._v_active = False
            self._btn_v.configure(text="🎤  START VOICE COMMANDS", bg="#1565c0")
            self._lbl_v.configure(text="● Voice Cmds: INACTIVE", fg="#555577")
            self._voice.stop()
    def _toggle_vt(self):
        if not self._vt_active:
            self._vt_active = True
            self._btn_vt.configure(text="⏹  STOP VOICE TYPING", bg="#b71c1c")
            self._lbl_vt.configure(text="● Voice Typing: ACTIVE ✓", fg="#ce93d8")
            self._emit("⌨️   Voice Typing mode ON — click into any text field and speak!")
            self._typist.start()
            threading.Thread(target=self._watch_typist, daemon=True).start()
        else:
            self._vt_active = False
            self._btn_vt.configure(text="⌨️   START VOICE TYPING", bg="#6a1b9a")
            self._lbl_vt.configure(text="● Voice Typing: INACTIVE", fg="#555577")
            self._typist.stop()
    def _watch_typist(self):
        while self._typist.is_active:
            time.sleep(0.3)
        if self._vt_active:
            self._vt_active = False
            self.root.after(0, lambda: (
                self._btn_vt.configure(text="⌨️   START VOICE TYPING", bg="#6a1b9a"),
                self._lbl_vt.configure(text="● Voice Typing: INACTIVE", fg="#555577"),
            ))
    def _quit(self):
        self._emit("👋  Shutting down…")
        self._running = False
        if self._g_active:  self._stop_g()
        if self._v_active:  self._voice.stop()
        if self._vt_active: self._typist.stop()
        time.sleep(0.25)
        self.root.destroy(); sys.exit(0)
    def run(self):
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth(); sh = self.root.winfo_screenheight()
        w  = self.root.winfo_width();       h  = self.root.winfo_height()
        self.root.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")
        self.root.mainloop()

def check_deps():
    import importlib.util as ilu
    required = {"cv2": "opencv-python", "mediapipe": "mediapipe==0.10.14",
                "pyautogui": "pyautogui", "numpy": "numpy"}
    missing = [pkg for mod, pkg in required.items() if not ilu.find_spec(mod)]
    if missing:
        print("\n❌  Missing packages — run:")
        print("      pip install " + " ".join(missing))
        print("\n  Full install:")
        print("      pip uninstall tensorflow tensorflow-intel mediapipe protobuf -y")
        print("      pip install protobuf==3.20.3 mediapipe==0.10.14")
        print("      pip install opencv-python pyautogui numpy SpeechRecognition pyaudio")
        sys.exit(1)
    import mediapipe as _mp
    ver = tuple(int(x) for x in _mp.__version__.split(".")[:2])
    if ver < (0, 10):
        print(f"\n⚠  mediapipe {_mp.__version__} too old.")
        print("   pip install --upgrade mediapipe==0.10.14")
        sys.exit(1)
    print(f"   mediapipe {_mp.__version__}  ✓")

if __name__ == "__main__":
    print("=" * 62)
    print("  AI GESTURE & VOICE DESKTOP CONTROLLER")
    print("  Python 3.12  |  MediaPipe DL  |  OpenCV  |  PyAutoGUI")
    print("=" * 62)
    check_deps()
    print(f"   Screen  {Config.SCREEN_W} × {Config.SCREEN_H}  ✓")
    print(f"   Voice   {'Available' if VOICE_AVAILABLE else 'Unavailable'}  ✓")
    print("\n  Launching…\n")
    App().run()
