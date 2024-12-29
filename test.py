# import face_recognition
# import cv2
# import numpy as np
# import datetime
# import os 
# import time

# # Load a sample picture and learn how to recognize it.
# iqbal_image = face_recognition.load_image_file("iqbal.jpg")
# iqbal_face_encoding = face_recognition.face_encodings(iqbal_image)[0]

# # Load an image from file instead of webcam
# input_image = cv2.imread("captured_images\captured_image.jpg")

# # Create arrays of known face encodings and their names
# known_face_encodings = [
#     iqbal_face_encoding
# ]

# known_face_names = [
#     "iqbal"
# ]
# # Inisialisasi variabel untuk melacak status pengenalan
# prev_name = None
# time_since_last_send = 0
# send_interval = 5  # Delay dalam detik sebelum mengirim gambar berikutnya

# # Process the image for face recognition
# face_locations = face_recognition.face_locations(input_image)
# face_encodings = face_recognition.face_encodings(input_image, face_locations)

# face_names = []

# # Loop through each face in the image
# for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#     # Calculate face distance (similarity score)
#     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

#     # Get the index of the best match
#     best_match_index = np.argmin(face_distances)

#     # If the best match is below the threshold, consider it a known face
#     if face_distances[best_match_index] < 0.45:
#         name = known_face_names[best_match_index]
#         similarity_score = 1 - face_distances[best_match_index]  # Convert distance to similarity score
#         similarity_score = round(similarity_score * 100, 2)  # Convert to percentage
#         name = f"{name} ({similarity_score}%)"
#     else:
#         name = "Unknown"

#     if name != prev_name:
#         prev_name = name
#         time_since_last_send = 0
#         # Draw a box around the face
#         cv2.rectangle(input_image, (left, top), (right, bottom), (0, 0, 255), 2)

#         # Draw a label with a name below the face
#         cv2.rectangle(input_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(input_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
#     else:
#         # If name has not changed, increment time_since_last_send
#         time_since_last_send += 1
    
#     if name == "iqbal":
#         print("Detek Iqbal")

#     elif name == "Unknown":
#         filename = f"tidak_dikenali/tes.jpg"
#         cv2.imwrite(filename, input_image)

# # Display the resulting image
# cv2.imshow('Image', input_image)

# # Hit 'q' on the keyboard to quit!
# cv2.waitKey(0)

# cv2.destroyAllWindows()


import face_recognition
import cv2
import numpy as np
import datetime
import os 
import time

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)


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
time_since_last_send = 0
send_interval = 5  # Delay dalam detik sebelum mengirim gambar berikutnya

while True:
    ret, frame = video_capture.read()

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calculate face distance (similarity score)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Get the index of the best match
        best_match_index = np.argmin(face_distances)

        # If the best match is below the threshold, consider it a known face
        if face_distances[best_match_index] < 0.45:
            name = known_face_names[best_match_index]
            similarity_score = 1 - face_distances[best_match_index]  # Convert distance to similarity score
            similarity_score = round(similarity_score * 100, 2)  # Convert to percentage
            name = f"{name} ({similarity_score}%)"
        else:
            name = "Unknown"

        if name != prev_name:
            prev_name = name
            time_since_last_send = 0
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
            # If name has not changed, increment time_since_last_send
            time_since_last_send += 1
        
        if name == "iqbal":
            print("Detek Iqbal")


        elif name == "Unknown":
            filename = f"tidak_dikenali/tes.jpg"
            cv2.imwrite(filename, frame)
  
            
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
