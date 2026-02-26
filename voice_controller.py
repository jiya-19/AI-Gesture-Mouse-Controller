
import speech_recognition as sr
import pyttsx3
import threading
import logging
import webbrowser
import os

class VoiceController:
    def __init__(self, settings):
        self.settings = settings
        self.running = False
        self.thread = None
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    text = self.recognizer.recognize_google(audio)
                    logging.info(f"Recognized: {text}")
                    self.handle_command(text.lower())
                except:
                    continue

    def handle_command(self, cmd):
        if "browser" in cmd:
            webbrowser.open("https://google.com")
            self.speak("Opening browser")
        elif "calculator" in cmd:
            os.system("calc")
            self.speak("Opening calculator")
        elif "stop" in cmd:
            self.stop()
            self.speak("Stopping voice control")
