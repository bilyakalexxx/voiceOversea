import time
import cv2
import pyttsx3
from ultralytics import YOLO
from scene_interpreter import describe_scene

# Load YOLO model
model = YOLO("yolov8n.pt")

# Init TTS
engine = pyttsx3.init()

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: could not open webcam.")
    exit()

last_spoken_time = 0
speak_interval = 4  # minimum seconds between speech
last_description = ""

def add_article(word: str) -> str:
    if word[0].lower() in "aeiou":
        return f"an {word}"
    return f"a {word}"

def build_description(objects):
    unique_objects = list(dict.fromkeys(objects))

    if not unique_objects:
        return "I cannot detect any clear objects right now."

    described = [add_article(obj) for obj in unique_objects]

    if len(described) == 1:
        return f"There is {described[0]} in view."
    if len(described) == 2:
        return f"There is {described[0]} and {described[1]} in view."

    return "There are " + ", ".join(described[:-1]) + f", and {described[-1]} in view."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: could not read frame.")
        break

    results = model(frame, verbose=False)
    detected_objects = []
    annotated_frame = frame

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_objects.append(class_name)

        annotated_frame = result.plot()

    cv2.imshow("Smart Webcam Detection", annotated_frame)

    current_time = time.time()
    description = describe_scene(detected_objects, mode="guide")

    should_speak = (
        current_time - last_spoken_time >= speak_interval
        and description != last_description
    )

    if should_speak:
        print(description)
        engine.say(description)
        engine.runAndWait()
        last_spoken_time = current_time
        last_description = description

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()