from ultralytics import YOLO
import cv2

# Load your trained emotion detection model
model = YOLO('/home/nvidia02/emotional-detection/runs/classify/train5/weights/best.pt')  # Change to your emotion detection model path

# Open webcam
cap = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define emotion labels
emotion_labels = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Define color mapping for emotions
emotion_colors = {
    "Anger": (0, 0, 255),  # Red
    "Contempt": (128, 0, 128),  # Purple
    "Disgust": (0, 128, 0),  # Green
    "Fear": (255, 165, 0),  # Orange
    "Happy": (0, 255, 0),  # Green
    "Neutral": (128, 128, 128),  # Gray
    "Sad": (255, 0, 0),  # Blue
    "Surprise": (255, 255, 0)  # Cyan
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run emotion detection using the model
    result = model.predict(frame, verbose=False)[0]

    # Get top prediction
    class_id = result.probs.top1
    confidence = result.probs.top1conf
    label = result.names[class_id]

    # Format the label and confidence
    percent = int(confidence * 100)
    if confidence < 0.65:
        text = "N/A – No Emotion Detected"
        color = (128, 128, 128)  # Gray
    else:
        text = f"{label} ({percent}%)"
        color = emotion_colors.get(label, (255, 255, 255))  # Default to white if emotion is unknown

    # Draw the text on the frame
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Show the frame with emotion label
    cv2.imshow("Emotion Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close the window
cap.release()
cv2.destroyAllWindows()
