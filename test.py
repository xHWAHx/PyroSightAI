import cv2                                                                           # for real-time video capturing
import numpy as np                                                                   # used for mathematical/numerical operations 
from tensorflow.keras.models import load_model                                       # used to load the pretrained model from train.py 

# load the pre-trained model from train.py
model = load_model('fire_detection_model.h5')

# resize, normalize, and reshape frames to improve model prediction 
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (128, 128))                                   # resized to match the input of the first layer in train.py, 128x128 
    frame_normalized = frame_resized / 255.0                                        # helps improve numerical stability. In general pixels are 0 to 255, if we get numbers down to 0 - 1 matrix calculations will be easier
    frame_reshaped = np.expand_dims(frame_normalized, axis=0)                       # adds an extra dimension, batch size 1, because CNNs expect batches 
    return frame_reshaped

# To classify small, medium, or large
def calculate_fire_area(frame):
                                                                                    # Convert from BGR to HSV and apply color threshold for fire-like colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([18, 50, 50])                                             # Adjust for fire color to bright yellow
    upper_fire = np.array([35, 255, 255])                                           # Adjust for fire color to bright blue 
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    fire_pixels = np.sum(mask > 0)
    total_pixels = frame.shape[0] * frame.shape[1]
    fire_area = (fire_pixels / total_pixels) * 100                                  # Percentage
    return fire_area

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_frame = preprocess_frame(frame)
    prediction = model.predict(input_frame)
    confidence = prediction[0][0]

    fire_area = calculate_fire_area(frame)

    actions = []                                                                    # To store actions to display on the frame

    if confidence > 0.9 and fire_area > 50:
        label = "Large Fire Detected!"
        actions.append("Sprinklers Activated!")
        actions.append("Emergency Responders Signaled!")
        actions.append("Owner Notified!")
    elif confidence > 0.7 and fire_area > 20:
        label = "Medium Fire Detected!"
        actions.append("Sprinklers Activated!")
        actions.append("Owner Notified!")
    elif confidence > 0.5 and fire_area <= 20:
        label = "Small Fire Detected!"
        actions.append("Owner Notified!")
    else:
        label = "Environment Clear"
    
    # Display the main label
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if "Fire" in label else (0, 255, 0), 2)

    # Display the actions below the label
    for i, action in enumerate(actions):
        cv2.putText(frame, action, (10, 60 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('PyroSightAI', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

