```python
import cv2
import time

# Load the video
video_path = '1.MOV'  # replace with your video path
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read the video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Select the bounding box in the first frame
bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Frame")

# Choose your tracker
tracker_types = {
    'BOOSTING': cv2.legacy.TrackerBoosting_create, # bad -> 35 fps
    'MIL': cv2.legacy.TrackerMIL_create, # bad -> 14 fps
    'KCF': cv2.legacy.TrackerKCF_create, # good -> 14 fps
    'TLD': cv2.legacy.TrackerTLD_create, # garbage -> 4 fps
    'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create, # bad even for a slight occlusion -> 90 fps
    'MOSSE': cv2.legacy.TrackerMOSSE_create, # fair performance when there is no harsh movement -> 100 fps
    'CSRT': cv2.TrackerCSRT_create # Awesome performance -> 30 fps
}

# You can change the tracker type here
tracker_type = 'CSRT'  # Change to 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', or 'MOSSE' as needed
tracker = tracker_types[tracker_type]()

# Initialize the tracker with the selected bounding box
ret = tracker.init(frame, bbox)

fps = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    ret, bbox = tracker.update(frame)

    if ret:
        # Draw the bounding box
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    start_time = time.time()

cap.release()
cv2.destroyAllWindows()
```
