
import tkinter as tk
from tkinter import ttk
import logging
from gesture_controller import GestureController
from voice_controller import VoiceController

settings = {
    "gesture_sensitivity": 40,
    "cooldown": 0.7
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log"
)

gesture = GestureController(settings)
voice = VoiceController(settings)

def launch_app():
    root = tk.Tk()
    root.title("AI Desktop Controller")
    root.geometry("400x300")

    ttk.Label(root, text="AI Desktop Controller", font=("Arial", 14)).pack(pady=10)

    ttk.Button(root, text="Start Gesture", command=gesture.start).pack(pady=5)
    ttk.Button(root, text="Stop Gesture", command=gesture.stop).pack(pady=5)
    ttk.Button(root, text="Start Voice", command=voice.start).pack(pady=5)
    ttk.Button(root, text="Stop Voice", command=voice.stop).pack(pady=5)

    ttk.Button(root, text="Exit", command=root.destroy).pack(pady=20)

    root.mainloop()
