# app/kivy_app.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import threading
import time
import cv2
import pyttsx3
from ultralytics import YOLO
from app.scene_interpreter import describe_scene


class CameraApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Image widget for live camera
        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)

        # Start detection button
        self.start_button = Button(text="Start Detection", size_hint=(1, 0.1))
        self.start_button.bind(on_press=self.start_camera_thread)
        self.layout.add_widget(self.start_button)

        return self.layout

    def start_camera_thread(self, instance):
        # Run camera in a thread to keep UI responsive
        self.camera_thread = threading.Thread(target=self.run_camera_loop, daemon=True)
        self.camera_thread.start()
        self.start_button.disabled = True

    def run_camera_loop(self):
        # Initialize YOLO model and TTS
        model = YOLO("yolov8n.pt")
        engine = pyttsx3.init()

        cap = cv2.VideoCapture(0)
        last_spoken_time = 0
        speak_interval = 4
        last_description = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            detected_objects = []
            annotated_frame = frame.copy()  # default fallback

            # Process YOLO results safely
            for result in results:
                temp_frame = result.plot()
                if temp_frame is not None:
                    annotated_frame = temp_frame.copy()
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    detected_objects.append(class_name)

            # Scene description
            description = describe_scene(detected_objects, mode="guide")
            current_time = time.time()
            should_speak = (
                current_time - last_spoken_time >= speak_interval
                and description != last_description
            )
            if should_speak and detected_objects:
                engine.say(description)
                engine.runAndWait()
                last_spoken_time = current_time
                last_description = description

            # Convert BGR -> RGB for Kivy texture
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            buf = frame_rgb.flatten().tobytes()
            texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

            # Update UI safely from main thread
            Clock.schedule_once(lambda dt, tex=texture: self.update_image(tex))

        cap.release()

    def update_image(self, texture):
        self.image_widget.texture = texture


if __name__ == "__main__":
    CameraApp().run()