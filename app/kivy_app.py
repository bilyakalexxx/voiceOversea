from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import time
import threading
from queue import Queue
from collections import Counter

import pyttsx3
from ultralytics import YOLO
from app.scene_interpreter import describe_scene


class CameraApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # UI
        self.img = Image()
        self.layout.add_widget(self.img)

        self.button = Button(text="Start Detection", size_hint=(1, 0.1))
        self.button.bind(on_press=self.toggle_detection)
        self.layout.add_widget(self.button)

        # Camera
        self.cap = cv2.VideoCapture(0)

        # State
        self.running = False

        # Model
        self.model = YOLO("yolov8n.pt")

        # TTS engine
        self.engine = pyttsx3.init()

        # Queues
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue()
        self.tts_queue = Queue()

        # Timing
        self.last_spoken_time = 0
        self.speak_interval = 2
        self.last_description = ""
        self.first_detection = False

        # Start threads
        threading.Thread(target=self.inference_worker, daemon=True).start()
        threading.Thread(target=self.tts_worker, daemon=True).start()

        # UI update loop
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return self.layout

    # ----------------------
    # Threads
    # ----------------------

    def inference_worker(self):
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break

            results = self.model(frame, verbose=False)

            detected_objects = []
            annotated_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    detected_objects.append(class_name)

                plotted = result.plot()
                if plotted is not None:
                    annotated_frame = plotted.copy()

            self.result_queue.put((annotated_frame, detected_objects))

    def tts_worker(self):
        while True:
            text = self.tts_queue.get()
            if text is None:
                break

            self.engine.say(text)
            self.engine.runAndWait()

    # ----------------------
    # UI Logic
    # ----------------------

    def toggle_detection(self, instance):
        self.running = not self.running

        if self.running:
            self.button.text = "Detection ON"
            self.first_detection = True
            self.last_spoken_time = 0
            self.last_description = ""
        else:
            self.button.text = "Detection OFF"

            # ✅ Speak when stopping detection
            self.tts_queue.put("Detection stopped.")

    def speak(self, text):
        self.tts_queue.put(text)

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        display_frame = frame.copy()
        detected_objects = []

        # Push frame to inference queue (non-blocking)
        if self.running and self.frame_queue.empty():
            self.frame_queue.put(frame)

        # Get latest inference result if available
        if not self.result_queue.empty():
            display_frame, detected_objects = self.result_queue.get()

            if detected_objects:
                counts = Counter(detected_objects)
                description = describe_scene(list(counts.elements()), mode="guide")

                current_time = time.time()

                should_speak = False

                # First detection
                if self.first_detection:
                    should_speak = True
                    self.first_detection = False

                # Normal speaking logic
                elif (
                    current_time - self.last_spoken_time >= self.speak_interval
                    and description != self.last_description
                ):
                    should_speak = True

                if should_speak:
                    self.speak(description)
                    self.last_spoken_time = current_time
                    self.last_description = description

        # Render frame
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)

        buf = frame_rgb.tobytes()
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

        self.img.texture = texture

    def on_stop(self):
        self.cap.release()
        self.frame_queue.put(None)
        self.tts_queue.put(None)


if __name__ == "__main__":
    CameraApp().run()