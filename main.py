import os
import base64
import requests
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from plyer import tts, camera

class VoiceOverseaApp(App):
    def build(self):
        layout = BoxLayout()
        self.action_button = Button(
            text="TAP ANYWHERE\nTO SCAN", 
            font_size='36sp',
            color=(1, 1, 0, 1),              
            background_color=(0, 0, 0, 1),    
            background_normal=''             
        )
        self.action_button.bind(on_press=self.trigger_camera)
        layout.add_widget(self.action_button)
        return layout

    def on_start(self):
        print("\n=== SYSTEM INITIALIZED ===")
        self.speak("Voice Oversea initialized. Tap anywhere on the screen to capture your surroundings.")

    def trigger_camera(self, instance):
        print("\n--- BUTTON TAPPED ---")
        self.speak("Capturing surroundings.")
        
        # Define the target image name in your workspace directory
        self.photo_path = "last_captured_photo.jpg"
        
        # MOBILE VS DESKTOP HARDWARE CHECK:
        try:
            # This works on actual smartphones
            camera.take_picture(filename=self.photo_path, on_complete=self.upload_to_server)
        except Exception:
            print("[Desktop Mode] Bypassing mobile camera hardware. Using local test file.")
            self.upload_to_server(self.photo_path)

    def upload_to_server(self, filepath):
        if not os.path.exists(filepath):
            self.speak("Error. No image file found to analyze.")
            print(f"[!] Error: Image path '{filepath}' does not exist.")
            return

        self.speak("Analyzing photo over the network. Please wait.")
        print(f"[Server Workflow] Connecting to your RTX 4090 Ollama server...")
        
        try:
            # Android-safe Base64 conversion
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            # Pure Python requests implementation targeting your local IP
            url = 'http://192.168.0'
            payload = {
                "model": "qwen2.5vl:3b",
                "prompt": "Describe this image concisely for a blind person. Focus strictly on major obstacles or layout.",
                "images": [encoded_string],
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                description = response.json().get('response', 'No description received.')
                print(f"[Server Output] Result:\n{description}")
                self.speak(description)
            else:
                self.speak("Server returned an error response.")
                print(f"[!] Server Error Code: {response.status_code}")
            
        except Exception as e:
            self.speak("Failed to connect to the online server.")
            print(f"[!] Server Communication Error: {e}")

    def speak(self, text_string):
        print(f"[Voice Output]: {text_string}")
        try:
            # Mobile voice engine trigger
            tts.speak(text_string)
        except Exception:
            # Desktop fallback
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.say(text_string)
            engine.runAndWait()

if __name__ == '__main__':
    VoiceOverseaApp().run()