import cv2
from deepface import DeepFace
import numpy as np

print("*** Face Detection AI Started! ***")
print("*** Detecting Face + Emotion + Age + Gender ***")
print("=" * 50)
print("Press Q to quit")
print("=" * 50)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
   print("ERROR: Cannot open webcam!")
         exit()

frame_count = 0
result = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read from webcam!")
        break

    # Analyze every 30 frames to avoid lag
    frame_count += 1
    if frame_count % 30 == 0:
        try:
            # Analyze face using DeepFace
            result = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False
            )
        except Exception as e:
            result = None

    # Display results on frame
    if result:
        try:
            # Handle both list and dict responses
            data = result[0] if isinstance(result, list) else result

            age = data.get('age', 'Unknown')
            gender = data.get('dominant_gender', 'Unknown')
            emotion = data.get('dominant_emotion', 'Unknown')

            # Get emotion confidence
            emotions = data.get('emotion', {})
            emotion_confidence = round(emotions.get(emotion, 0), 1) if emotions else 0

            # Get face region
            region = data.get('region', {})
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display info on frame
            cv2.putText(frame, f"Age: {age} years",
                       (x, y-80), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Gender: {gender}",
                       (x, y-55), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 165, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion} ({emotion_confidence}%)",
                       (x, y-30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected!",
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)

            # Print to terminal
            print(f"Face Detected | Age: {age} | Gender: {gender} | Emotion: {emotion} ({emotion_confidence}%)")

        except Exception as e:
            pass

    # Add title to frame
    cv2.putText(frame, "Face + Emotion + Age + Gender AI",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press Q to Quit",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Face Detection AI", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("*** Face Detection AI Stopped! ***")