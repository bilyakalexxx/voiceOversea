def run_camera(mode="guide"):
    import time
    import cv2
    import pyttsx3
    from ultralytics import YOLO
    from app.scene_interpreter import describe_scene

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
        annotated_frame = frame

        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_objects.append(class_name)

            annotated_frame = result.plot()

        print(detected_objects)

        description = describe_scene(detected_objects, mode=mode)

        current_time = time.time()
        should_speak = (
            current_time - last_spoken_time >= speak_interval
            and description != last_description
        )

        if should_speak:
            engine.say(description)
            engine.runAndWait()
            last_spoken_time = current_time
            last_description = description

        cv2.imshow("Camera", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    run_camera()