from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import time
import threading
import pyttsx3
from ultralytics import YOLO
from app.scene_interpreter import describe_scene


class CameraApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Camera display
        self.img = Image()
        self.layout.add_widget(self.img)

        # Toggle button
        self.button = Button(text="Start Detection", size_hint=(1, 0.1))
        self.button.bind(on_press=self.toggle_detection)
        self.layout.add_widget(self.button)

        # Camera
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # optional Windows fix

        # Detection state
        self.running = False
        
        # extra detection off
        self.first_detection = False

        # YOLO + TTS
        self.model = YOLO("yolov8n.pt")
        self.engine = pyttsx3.init()

        # TTS control
        self.last_spoken_time = 0
        self.speak_interval = 2
        self.last_description = ""

        # Performance control
        self.frame_count = 0
        self.process_every_n_frames = 5

        # Start camera loop immediately
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return self.layout

    # 🔊 Non-blocking TTS
    def speak(self, text):
        def _speak():
            self.engine.say(text)
            self.engine.runAndWait()

        threading.Thread(target=_speak, daemon=True).start()

    def toggle_detection(self, instance):
        self.running = not self.running

        if self.running:
            self.button.text = "Detection ON"
            
            # Force immediate description
            self.first_detection = True
            self.last_spoken_time = 0
            self.last_description = ""

        else:
            self.button.text = "Detection OFF"
            self.speak("Detection stopped")

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        annotated_frame = frame.copy()
        detected_objects = []

        # Frame skipping for performance
        self.frame_count += 1

        if self.running and self.frame_count % self.process_every_n_frames == 0:
            results = self.model(frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    detected_objects.append(class_name)

                plotted = result.plot()
                if plotted is not None:
                    annotated_frame = plotted.copy()

            # TTS logic
            description = describe_scene(detected_objects, mode="guide")
            current_time = time.time()
            
            should_speak = False

            # 🔥 First detection → speak immediately
            if self.first_detection and detected_objects:
                should_speak = True
                self.first_detection = False

            # Normal behavior
            elif (
                current_time - self.last_spoken_time >= self.speak_interval
                and description != self.last_description
                and detected_objects
            ):
                should_speak = True

            if should_speak:
                self.speak(description)
                self.last_spoken_time = current_time
                self.last_description = description

        # Convert frame for Kivy display
        frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)

        buf = frame.tobytes()
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='rgb'
        )
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

        self.img.texture = texture

    def on_stop(self):
        self.cap.release()


if __name__ == "__main__":
    CameraApp().run()