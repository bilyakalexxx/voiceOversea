import os
import pyttsx3
from ollama import Client

def run_prototype():
    photo_name = "last_captured_photo.jpg"
    
    print("[1/4] Checking for image file...")
    if not os.path.exists(photo_name):
        print(f"\n[!] Missing file. Please place a test image named '{photo_name}' in D:\\voiceOversea.")
        return
    
    print(f"-> Found '{photo_name}' successfully.")
    print("\n[2/4] Sending photo to your RTX 4090 Ollama server...")
    
    client = Client(host='http://localhost:11434')
    prompt_text = "Describe this image concisely for a blind person. Focus strictly on major obstacles, furniture layout, or people present."
    
    try:
        response = client.generate(
            model='qwen2.5vl:3b',
            prompt=prompt_text,
            images=[photo_name]
        )
        
        description_text = response['response']
        print("\n[3/4] Description Generation Complete:")
        print("-" * 50)
        print(description_text)
        print("-" * 50)
        
        # --- NEW CODE FOR STEP 5: SPEAKING THE DESCRIPTION ---
        print("\n[4/4] Speaking description out loud...")
        
        # Initialize the native Windows offline text-to-speech engine
        engine = pyttsx3.init()
        
        # Optional: Adjust voice speed (words per minute) to make it highly legible
        engine.setProperty('rate', 180) 
        
        # Queue the text and play it through your laptop speakers
        engine.say(description_text)
        engine.runAndWait()
        print("-> Audio playback complete.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_prototype()