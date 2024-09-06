from mtcnn import MTCNN  # Importing MTCNN for face detection

detector = MTCNN()  # Initializing the MTCNN detector


# Function to detect faces in an image
def detect_faces(image):
    results = detector.detect_faces(image)  # Detecting faces
    return results  # Returning the detected faces
