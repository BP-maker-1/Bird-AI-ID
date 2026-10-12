import cv2
from ultralytics import YOLO
import csv

# 1. Load detector and species classifier
detector = YOLO("yolo26n.pt")
classifier = YOLO("/Users/bennettpotter/Desktop/AI-Birds/Bird-AI-ID/best.pt")

# 2. Set input source (0 for live webcam, or "path/to/video.mp4")
source = 0

# verbose=False suppresses default Ultralytics detection printouts in terminal
# Change imgsz from default 640 to 1024 or 1280
results = detector(source=source, imgsz=1024, show=False, stream=True, verbose=False,device="mps")
frame_count = 0

for result in results:
    frame_count += 1
    frame = result.orig_img.copy()
    
    # Store all detected birds in the current frame for a single output line
    frame_detected_birds = []

    # Process all bounding boxes detected in this frame
    for box in result.boxes:
        class_id = int(box.cls)
        detected_label = detector.names[class_id]

        if detected_label == "bird":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_bird = result.orig_img[y1:y2, x1:x2]

            if cropped_bird.size > 0:
                # Classify species (silent in terminal)
                species_res = classifier(cropped_bird,imgsz=640,conf=0.9, verbose=False,device="mps")
                
                species_name = species_res[0].names[species_res[0].probs.top1]
                species_conf = species_res[0].probs.top1conf.item()

                # Add to frame summary list
                bird_summary = f"{species_name} ({species_conf:.0%})"
                frame_detected_birds.append(bird_summary)

                # Draw bounding box and updated species label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                (text_w, text_h), _ = cv2.getTextSize(bird_summary, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(y1 - 25, 0)), (x1 + text_w, max(y1, 25)), (0, 255, 0), -1)
                cv2.putText(frame, bird_summary, (x1, max(y1 - 7, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Print a single clean line per frame to the terminal
    if frame_detected_birds:
        print(f"[Frame {frame_count:04d}] Found ({len(frame_detected_birds)}): {', '.join(frame_detected_birds)}")
    else:
        print(f"[Frame {frame_count:04d}] No birds detected")

    # Display video frame on screen
    cv2.imshow("NZ Bird Species Pipeline", frame)

    # 1ms delay allows the pipeline to run at full speed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
