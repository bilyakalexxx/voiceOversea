from ultralytics import YOLO
import pyttsx3

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Path to image
image_path = "Street_Park.jpg"   # change this to your image file name

# Run detection
results = model(image_path)

# Collect detected object names
detected_objects = []

for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        detected_objects.append(class_name)

# Remove duplicates and keep order
unique_objects = list(dict.fromkeys(detected_objects))

# Build description
if unique_objects:
    if len(unique_objects) == 1:
        description = f"There is a {unique_objects[0]} in view."
    elif len(unique_objects) == 2:
        description = f"There is a {unique_objects[0]} and a {unique_objects[1]} in view."
    else:
        description = "There are " + ", ".join(unique_objects[:-1]) + f", and {unique_objects[-1]} in view."
else:
    description = "I could not detect any objects."

print(description)

# Speak description
engine = pyttsx3.init()
engine.say(description)
engine.runAndWait()

