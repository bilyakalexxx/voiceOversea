import os
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from plyer import tts, camera

# Import the network client library we used in app.py
from ollama import Client

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
            # DESKTOP FALLBACK: If running on your laptop, bypass the mobile camera code 
            # and immediately process the static test photo sitting in D:\voiceOversea
            print("[Desktop Mode] Bypassing mobile camera hardware. Using local test file.")
            self.upload_to_server(self.photo_path)

    def upload_to_server(self, filepath):
        if not os.path.exists(filepath):
            self.speak("Error. No image file found to analyze.")
            print(f"[!] Error: Place an image named '{filepath}' inside D:\\voiceOversea first.")
            return

        self.speak("Analyzing photo over the network. Please wait.")
        print(f"[Server Workflow] Connecting to your RTX 4090 Ollama server...")
        
        try:
            # Connect to your local server (localhost works while testing on the same laptop)
            client = Client(host='http://192.168.0.104:11434')
            
            response = client.generate(
                model='qwen2.5vl:3b',
                prompt="Describe this image concisely for a blind person. Focus strictly on major obstacles or layout.",
                images=[filepath]
            )
            
            description = response['response']
            print(f"[Server Output] Result:\n{description}")
            
            # Speak the server's description out loud
            self.speak(description)
            
        except Exception as e:
            self.speak("Failed to connect to the online server.")
            print(f"[!] Server Communication Error: {e}")

    def speak(self, text_string):
        # Keeps terminal clean and logs exactly what a user would hear
        print(f"[Voice Output]: {text_string}")
        try:
            # Mobile voice engine trigger
            tts.speak(text_string)
        except Exception:
            # Desktop fallback: Use our working desktop audio module from app.py
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.say(text_string)
            engine.runAndWait()

if __name__ == '__main__':
    VoiceOverseaApp().run()