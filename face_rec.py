import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import dlib


def encoded_known_faces():

    # encodes the known faces from the faces folder.

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./known_faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("known_faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):

    # encodes the test image
    
    face = fr.load_image_file("known_faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def compare_faces(im):

    # checks the test image with the known faces.
    # returns a list of names of the faces in the image.
    # then draws a rectangle around the face detected.

    faces = encoded_known_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    img = cv2.resize(img, (0, 0), fx=1, fy=1)
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:

        # See if the face is a match for the known face(s)

        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"


        # use the known face with the smallest distance to the new face

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):

            # Drawing a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Drawing a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:

        cv2.imshow('Final', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 


print(compare_faces("test1.jpg"))


