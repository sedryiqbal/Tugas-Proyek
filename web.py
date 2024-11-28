from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import face_recognition
import numpy as np

app = Flask(__name__)

# Create a directory to save images if not exist
if not os.path.exists('static/captured_images'):
    os.makedirs('static/captured_images')

camera = cv2.VideoCapture(0)

def detect_faces_in_image(image_path):
    # Load the uploaded image
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Known faces (add your known faces here)
    known_face_encodings = [
        face_recognition.face_encodings(face_recognition.load_image_file("iqbal.jpg"))[0]
    ]
    known_face_names = [
        "iqbal"
    ]

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names

def annotate_image(image_path, face_locations, face_names):
    # Load the image with OpenCV to annotate
    image = cv2.imread(image_path)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw the label with a name below the face
        cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    annotated_image_path = os.path.join('static/captured_images', 'annotated_image.jpg')
    cv2.imwrite(annotated_image_path, image)
    return annotated_image_path

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', captured_image_url=None)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if success:
        # Flip the frame vertically before saving
        frame = cv2.flip(frame, 1)
        img_path = os.path.join('static/captured_images', 'captured_image.jpg')
        cv2.imwrite(img_path, frame)

        # Perform face recognition
        face_locations, face_names = detect_faces_in_image(img_path)
        annotated_image_path = annotate_image(img_path, face_locations, face_names)

        # Return the rendered page with the annotated image
        return render_template('index.html', captured_image_url=annotated_image_path)
    else:
        return "Failed to capture image", 500

if __name__ == "__main__":
    app.run(debug=True)
