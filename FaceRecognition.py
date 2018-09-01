import numpy as np
import cv2
import dlib
from imutils import face_utils
import os, os.path
from mysql import connector
import mysql.connector
from datetime import datetime

conn = connector.Connect(user="root", database="dokanak", password="ibrahim")

face_detector = dlib.get_frontal_face_detector()
face_predictor_68 = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")
face_predictor_5 = dlib.shape_predictor("Models/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("Models/dlib_face_recognition_resnet_model_v1.dat")
face_aligner = face_utils.FaceAligner(face_predictor_68, desiredFaceWidth=256)

def preprocess(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    grey_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    return small_frame, grey_frame

def detectfaces(grey):
    detectedfaces = face_detector(grey, 2)
    return detectedfaces

def alignface(image, grey, rect):
    alignedface = face_aligner.align(image, grey, rect)
    rect = detectfaces(alignedface)[0]
    return alignedface, rect

def predictface(image, rect):
    landmarks = face_predictor_5(image, rect)
    return landmarks

def encodeface(image, landmark):
    embeddings = face_encoder.compute_face_descriptor(image, landmark, 2)
    return np.array(embeddings)

def draw_name(image, name, rect):
    (x, y, w, h) = face_utils.rect_to_bb(rect)

    x *= 4; y *= 4; w *= 4; h *= 4
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.rectangle(image, (x, y + h - 20), (x + w, y + h), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (x, y + h), font, 1.0, (255, 255, 255), 1)

    return image

def load_faces(path):
    image_path_list = []
    images = []
    greys = []
    names = []
    for file in os.listdir(path):
        image_path_list.append(os.path.join(path, file))
        names.append(os.path.splitext(file)[0])

    for imagePath in image_path_list:
        image = cv2.imread(imagePath)
        image, grey = preprocess(image)
        images.append(image)
        greys.append(grey)

    return images,greys, names

def encodeDataset(faces, greys):
    embeddings = []
    for i,face in enumerate(faces):
        grey = greys[i]
        rect = detectfaces(grey)[0]
        #align, rect = alignface(face, grey, rect)
        landmark = predictface(grey,rect)
        embedding = encodeface(face,landmark)

        embeddings.append(embedding)

    return embeddings

def insertName(date,time,bannedID):
    cur = conn.cursor(buffered=True)
    date = datetime.now().date()
    time = datetime.now().time()
    insertquery = "INSERT INTO `dokanak`.`cameralog` (`date`, `time`, `bannedID`) " \
                  "VALUES (%s, %s, %s)"
    cur.execute(insertquery,(date,time,bannedID))
    conn.commit()

