#imports
#pip install deepface
#for windows
#https://pythonprogramming.net/facial-recognition-python/
#for raspery
#https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65
import csv
from datetime import datetime
import face_recognition
import imutils
import pickle
import time
from cv2 import face
import cv2
from datetime import date
from deepface import DeepFace
import threading as th
import threading

from imutils.video import VideoStream


def emotions(rgb):
    global emototionsResult
    emototionsResult = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False)
def encoding_process(rgb,top, right, botton, left):
    global f_state
    global encodings
    encodings = face_recognition.face_encodings(rgb, [(top, right, botton, left)], model="small")
    if len(encodings) != 0:
        f_state = True


def recognise_faces_emotions(faces, rgb):

    global f_state
    global emototionsResult

    for face in faces:
        # print(face, "faces")
        start_point = face[0], face[1]
        top = face[1]
        left = face[0]
        end_point = face[0] + face[2], face[1] + face[3]
        right = face[0] + face[2]
        botton = face[1] + face[3]
        cv2.rectangle(rgb, start_point, end_point, (255, 0, 0), 2)

        t1 = th.Thread(target=emotions,args=(rgb,))
        t2 = th.Thread(target=encoding_process,args=(rgb,top, right, botton, left,))

        t1.start()
        t2.start()
        t1.join()

        t2.join()


        # boxes = face_recognition.face_locations(rgb,
        #                                        model=model)
        # print(boxes)
        # print(faces,"\n",boxes)
        # print(boxes)

        names = []
        # loop over the facial embeddings
        if f_state:
            for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding)
                name = "Unknown"

                # check to see if we have found a match
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)
                    print(name)
                    print(emototionsResult["dominant_emotion"])
                # update the list of names
                names.append(name)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                out.writerow([name, emototionsResult["dominant_emotion"], current_time])
        cv2.imshow("ml course", rgb)
        key = cv2.waitKey(1)
        if key==ord("q"):
            f.close()
            keep_going=False
            break

# def key_capture_thread():
#     global keep_going
#     print("Press any key then enter to exit")
#     input()
#     keep_going = False
#     f.close()


keep_going = True

encodingPath = "model.pickle"
# vediopath = r"D:\Jops\emotion detection\faical recognition\face-recognition-opencv\videos"
model = "hog"
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(encodingPath, "rb").read())
recognizer = face.LBPHFaceRecognizer_create()
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
#cam = cv2.VideoCapture(0)
vs = VideoStream(src=0,usePiCamera=False).start()
writer = None
#time.sleep(0.2)
today = date.today()
d4 = today.strftime("%b-%d-%Y")
fileName = d4 + ".csv"
f = open(fileName, 'w')
out = csv.writer(f, delimiter=",")
out.writerow(["name", "state", "time"])
# th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()

# loop over frames from the video file stream
while keep_going:
    # grab the frame from the threaded video stream
    #isValid, frame = cam.read()
    frame = vs.read()

    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame)
    grayImage = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    r = frame.shape[1] / float(rgb.shape[1])
    faces = classifier.detectMultiScale(grayImage, scaleFactor=1.2)
    if len(faces)>0:
        recognise_faces_emotions(faces, rgb)
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face

cv2.destroyAllWindows()

# check to see if the video writer point needs to be released
# if writer is not None:
# 	writer.release()
