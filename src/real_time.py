import cv2  # Import the OpenCV library for computer vision tasks
import numpy as np  # Import the numpy library for numerical operations
from tensorflow.keras.models import load_model  # Import the load_model function to load a pre-trained Keras model
from detect import detect_faces  # Import the detect_faces function for face detection

# Define a dictionary mapping emotion labels to corresponding emoji paths and emotion names
EMOTION_MAP = {
    0: ('emojis/angry.png', 'Angry'),
    1: ('emojis/disgust.png', 'Disgust'),
    2: ('emojis/fear.png', 'Fear'),
    3: ('emojis/happy.png', 'Happy'),
    4: ('emojis/sad.png', 'Sad'),
    5: ('emojis/surprise.png', 'Surprise'),
    6: ('emojis/neutral.png', 'Neutral')
}


def overlay_emoji_and_text(frame, emoji_path, emotion_text, x, y, width, height):
    # Define a function to overlay an emoji and emotion text on the frame
    emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)  # Read the emoji image with alpha channel
    emoji_img = cv2.resize(emoji_img, (width, height))  # Resize the emoji to fit the detected face
    try:
        for c in range(0, 3):  # Overlay the emoji on the frame
            frame[y:y + height, x:x + width, c] = emoji_img[:, :, c] * (emoji_img[:, :, 3] / 255.0) + frame[y:y + height, x:x + width, c] * (1.0 - emoji_img[:, :, 3] / 255.0)
    except Exception as e:  # Handle any exceptions that occur during the overlay process
        print("Failed to overlay emoji:", e)

    font = cv2.FONT_HERSHEY_SIMPLEX  # Define the font for the text
    cv2.putText(frame, emotion_text, (x, y - 10), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # Add the emotion text above the face

    return frame  # Return the frame with the overlaid emoji and text


def real_time_recognition(model_path):
    # Define a function to perform real-time emotion recognition
    model = load_model(model_path)  # Load the pre-trained emotion recognition model
    cap = cv2.VideoCapture(0)  # Start capturing video from the default camera

    while True:  # Start an infinite loop to process the video frames
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:  # If frame reading fails, break the loop
            break

        faces = detect_faces(frame)  # Detect faces in the frame
        for face in faces:  # Iterate over detected faces
            x, y, width, height = face['box']  # Get the coordinates and size of the face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Draw a rectangle around the face
            roi = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)  # Convert the face region to
            # grayscale
            roi = cv2.resize(roi, (48, 48))  # Resize the face region to 48x48 pixels
            roi = roi.astype('float32') / 255.0  # Normalize the pixel values
            roi = np.expand_dims(np.expand_dims(roi, -1), 0)  # Reshape the ROI to match the model input shape
            prediction = model.predict(roi)  # Predict the emotion for the face region
            max_index = np.argmax(prediction[0])  # Get the index of the highest probability emotion
            emoji_path, emotion_text = EMOTION_MAP[max_index]  # Get the corresponding emoji path and emotion text
            frame = overlay_emoji_and_text(frame, emoji_path, emotion_text, x, y, width, height)  # Overlay the emoji
            # and text on the frame

        cv2.imshow("Real-time Emotion Recognition", frame)  # Display the frame with the overlaid emoji and text
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if the 'q' key is pressed
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    real_time_recognition('models/emotion_recognition_model.keras')  # Run the real-time recognition function with
    # the specified model
