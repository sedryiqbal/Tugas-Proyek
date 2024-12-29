import face_recognition
import cv2
import numpy as np
import datetime
import os
import asyncio
import websockets
import base64
import json

# Load a sample picture and learn how to recognize it.
iqbal_image = face_recognition.load_image_file("iqbal.jpg")
iqbal_face_encoding = face_recognition.face_encodings(iqbal_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    iqbal_face_encoding
]
known_face_names = [
    "iqbal"
]

# Inisialisasi variabel untuk melacak status pengenalan
prev_name = None

# Fungsi utama untuk pengenalan wajah dan streaming
async def recognize_and_stream(websocket):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        recognition_data = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Calculate face distance (similarity score)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # If the best match is below the threshold, consider it a known face
            if face_distances[best_match_index] < 0.45:
                name = known_face_names[best_match_index]
                similarity_score = 1 - face_distances[best_match_index]  # Convert distance to similarity score
                similarity_score = round(similarity_score * 100, 2)  # Convert to percentage
                name = f"{name} ({similarity_score}%)"
            else:
                name = "Unknown"

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Append recognition data
            recognition_data.append({
                'name': name,
                'confidence': similarity_score if name != "Unknown" else 0
            })

            # Save frame if unknown
            if name == "Unknown":
                filename = f"tidak_dikenali/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                os.makedirs("tidak_dikenali", exist_ok=True)
                cv2.imwrite(filename, frame)

        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare data to send
        data = {
            'frame': frame_base64,
            'recognition_data': recognition_data
        }

        # Send data via WebSocket
        await websocket.send(json.dumps(data))
        await asyncio.sleep(0.033)  # ~30 FPS

        # Display the video locally
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Jalankan WebSocket server
async def main():
    async with websockets.serve(recognize_and_stream, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
