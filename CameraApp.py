from imutils.video import WebcamVideoStream
from FaceRecognition import *
from face_recognition import compare_faces
import cv2


SKIP_FRAME_RATIO = 2
images, greys, names = load_faces("knownfaces")
knownfaces = encodeDataset(images, greys)
ID = [1,2,3,4,5,6,7,8,9]
face_locations = []
face_names = []
process_frame_count = 1
vs = WebcamVideoStream().start()

while True:
    frame = vs.read()

    if(process_frame_count % SKIP_FRAME_RATIO == 0):

        temp_frame, grey_frame = preprocess(frame)
        face_locations = detectfaces(grey_frame)

        face_names = []
        id_iden = []
        for face in face_locations:
            #facealigned, rect = alignface(temp_frame,grey_frame,face)
            facelandmark = predictface(grey_frame,face)
            faceencoded = encodeface(temp_frame, facelandmark)
            matchfaces = compare_faces(knownfaces, faceencoded)

            name = "Unknown"
            id = "id"
            for i,match in enumerate(matchfaces):
                if(match == True):
                    name = names[i]
                    id = ID[i]
            face_names.append(name)
            id_iden.append(id)

    process_frame_count += 1
    for i,location in enumerate(face_locations):
        frame = draw_name(frame,face_names[i],location)
        insertName(None,None,id_iden[i])

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()