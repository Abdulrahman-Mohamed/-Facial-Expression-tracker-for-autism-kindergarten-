#imports
#pip install deepface
#for windows
#https://pythonprogramming.net/facial-recognition-python/
#for raspery
#https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65
import csv
import locale
import time
from datetime import datetime
import face_recognition
import imutils
import pickle
import cv2
from datetime import date
from PIL import ImageFont, ImageDraw, Image
import numpy as np


from cv2 import face
from deepface import DeepFace
import threading as th
import threading

from imutils.video import VideoStream
threads=[]
emototionsResult={}
#kızgın, korku, tarafsız, üzgün, tiksinti, mutlu , sürpriz
#angry, fear, neutral, sad, disgust, happy , surprise

def translate_states(state):
    translated_state=""
    if state =="angry":
        translated_state="kızgın"
    elif state=="fear":
        translated_state = "korku"
    elif state == "neutral":
        translated_state = "tarafsız"
    elif state == "sad":
        translated_state = "üzgün"
    elif state == "disgust":
        translated_state = "tiksinti"
    elif state == "happy":
        translated_state = "mutlu"
    elif state == "surprise":
        translated_state = "sürpriz"
    if translated_state!="":
        return translated_state
    else:
        return None
def one_in_all_thread(face):
    global left
    global video
    global rgb

    state=""
    f_state=False
    encodings = face_recognition.face_encodings(face, model="small")
    if len(encodings) != 0:
        f_state = True
        names = []
        # loop over the facial embeddings
    if f_state:
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"

            t1 = threading.Thread(target=emotions, args=(face,))
            t1.start()
            t1.join()
            print(Y,left)
            if len(emototionsResult) != 0:
                state=translate_states(emototionsResult["dominant_emotion"])
                if state!=None:
                    ignore, encoding = locale.getlocale()
                    print(encoding,ignore)
                    #name = (state.encode('utf-8'), 'utf-8')
                    # cv2.putText(rgb,state.encode('utf-8') , (left, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.75, (160, 0, 0), 2)
                    img_pil = Image.fromarray(rgb)
                    font = ImageFont.truetype("arial.ttf", 24, encoding="utf-8")

                    draw = ImageDraw.Draw(img_pil)
                    draw.text((left, Y-15), state,font=font ,fill=(160, 0, 0))
                    rgb = np.array(img_pil)

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


                print(state.encode('utf-8'))
            cv2.putText(rgb, name, (left, Y-25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (160, 0, 0), 2)


            # update the list of names
            names.append(name)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            out.writerow([name, emototionsResult["dominant_emotion"], current_time])

def emotions(rgb):
    global emototionsResult
    emototionsResult = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False)
def encoding_process(rgb):
    global f_state
    global encodings
    encodings = face_recognition.face_encodings(rgb, model="small")
    if len(encodings) != 0:
        f_state = True

def recognise_multithread(faces,rgb):
    global threads
    global left
    global Y


    for face in faces:
        # print(face, "faces")

        start_point = face[0], face[1]
        top = face[1]
        left = face[0]
        end_point = face[0] + face[2], face[1] + face[3]
        right = face[0] + face[2]
        botton = face[1] + face[3]
        x, y, w, h = [v for v in face]
        # Define the region of interest in the image
        cv2.rectangle(rgb, (left, top), (right, botton),
                      (0, 255, 0), 2)
        Y= top - 15 if top - 15 > 15 else top + 15



        one_in_all_thread(rgb[y:y + h, x:x + w])



        #cv2.rectangle(rgb, start_point, end_point, (255, 0, 0), 2)
    processes = []




# def recognise_faces_emotions(faces, rgb):
#
#
#     global f_state
#     global emototionsResult
#
#     face_crop = []
#
#     for face in faces:
#         # print(face, "faces")
#
#         start_point = face[0], face[1]
#         top = face[1]
#         left = face[0]
#         end_point = face[0] + face[2], face[1] + face[3]
#         right = face[0] + face[2]
#         botton = face[1] + face[3]
#         x, y, w, h = [v for v in face]
#         # Define the region of interest in the image
#         face_crop.append(rgb[y:y + h, x:x + w])
#         cv2.rectangle(rgb, start_point, end_point, (255, 0, 0), 2)
#
#
#     for face in face_crop:
#         #cv2.imshow('face', face)
#         #cv2.waitKey(1)
#         f_state=False
#         t2 = threading.Thread(target=encoding_process,args=(face,))
#
#         t2.start()
#
#
#         t2.join()
#
#
#     # boxes = face_recognition.face_locations(rgb,
#     #                                        model=model)
#     # print(boxes)
#     # print(faces,"\n",boxes)
#     # print(boxes)
#
#         names = []
#         # loop over the facial embeddings
#         if f_state:
#             for encoding in encodings:
#                 # attempt to match each face in the input image to our known
#                 # encodings
#                 matches = face_recognition.compare_faces(data["encodings"],
#                                                          encoding)
#                 name = "Unknown"
#
#                 # check to see if we have found a match
#                 if True in matches:
#                     # find the indexes of all matched faces then initialize a
#                     # dictionary to count the total number of times each face
#                     # was matched
#                     matchedIdxs = [i for (i, b) in enumerate(matches) if b]
#                     counts = {}
#
#                     # loop over the matched indexes and maintain a count for
#                     # each recognized face face
#                     for i in matchedIdxs:
#                         name = data["names"][i]
#                         counts[name] = counts.get(name, 0) + 1
#
#                     # determine the recognized face with the largest number
#                     # of votes (note: in the event of an unlikely tie Python
#                     # will select first entry in the dictionary)
#                     name = max(counts, key=counts.get)
#                 t1 = threading.Thread(target=emotions, args=(face,))
#                 t1.start()
#                 t1.join()
#                 print(name)
#                 print(emototionsResult["dominant_emotion"])
#
#                 # update the list of names
#                 names.append(name)
#                 now = datetime.now()
#                 current_time = now.strftime("%H:%M:%S")
#                 out.writerow([name, emototionsResult["dominant_emotion"], current_time])
#         cv2.imshow("ml course", rgb)
#         key = cv2.waitKey(1)
#         if key == ord("q"):
#             f.close()
#             break

def key_capture_thread():
    global keep_going
    global rgb

    print("Press any key then enter to exit")
    input()
    keep_going = False
    f.close()
    video.release()


keep_going = True
usingPiCamera = True
encodingPath = r"eda.pickle"
#vediopath = r"D:\Jops\emotion detection\faical recognition\face-recognition-opencv\videos"
model = "hog"
classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(encodingPath, "rb").read())
#recognizer = face.LBPHFaceRecognizer_create()
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
#cam = cv2.VideoCapture(0)
vs = VideoStream(src=0,usePiCamera=False).start()
#vs=cv2.VideoCapture(r"D:\Jops\Eval\WIN_20201227_16_49_50_Pro.mp4")
writer = None
time.sleep(0.2)
today = date.today()
d4 = today.strftime("%b-%d-%Y")
fileName = d4 + ".csv"
f = open(fileName, 'w')
out = csv.writer(f, delimiter=",")
out.writerow(["name", "state", "time"])
th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()

# loop over frames from the video file stream
while keep_going :
    global video
    # grab the frame from the threaded video stream
    #isValid, frame = cam.read()
    frame = vs.read()
    print(classifier)

    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if True:
        rgb = imutils.resize(frame,750,750)
        grayImage = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        r = frame.shape[1] / float(rgb.shape[1])
        height , width , layers=rgb.shape
        #if writer is None:
            #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            #writer = cv2.VideoWriter()
            #success = writer.open('output test 1.avi', fourcc, fps=24,
                                     #frameSize=(rgb.shape[1], rgb.shape[0]), )


        faces = classifier.detectMultiScale(rgb, scaleFactor=1.2)

        if len(faces)>0:
            #recognise_faces_emotions(faces, rgb)
            recognise_multithread(faces,rgb)
            cv2.imshow("Frame", rgb)
            #writer.write(rgb)

            # video.write(rgb)
            cv2.waitKey(1)

    else:

        break


    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face

cv2.destroyAllWindows()
writer.release()
vs.release()
# check to see if the video writer point needs to be released
# if writer is not None:
# 	writer.release()
