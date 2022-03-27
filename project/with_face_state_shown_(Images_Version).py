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
from decimal import Decimal
import face_recognition
import imutils
from imutils import paths

import pickle
import cv2
from datetime import date
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from collections import Counter
from appdirs import unicode
from cv2 import face
from deepface import DeepFace
import threading as th
import threading

from imutils.video import VideoStream
threads=[]
emototionsResult={}
emotions_Dictionary = {}
count_down_timer_current = 0.0
count_down_timer_end = 0.0

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
    global count_down_timer_end
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
            if(name not in emotions_Dictionary):
                emotions_Dictionary[name] = [emototionsResult["dominant_emotion"]]
            else:
                emotions_Dictionary[name].append(emototionsResult["dominant_emotion"])
            print(emototionsResult["dominant_emotion"])

            # count_down_timer_current = Decimal((current_time.split(sep = ":") [1])+"."+(current_time.split(sep = ":") [2]))
            #
            # print("current timer:",count_down_timer_current)
            #
            # "edit the time limite to 60 sec"
            # if count_down_timer_end == 0.0 and count_down_timer_current != 0.0:
            #     count_down_timer_end = count_down_timer_current + Decimal(1.0) if count_down_timer_current < 59 else count_down_timer_current -59
            # print("end timer:", count_down_timer_end)
            # if count_down_timer_current >= count_down_timer_end:
            #     time = Decimal((current_time.split(sep=":")[0]) + "." + (current_time.split(sep=":")[1]))
            #
            #     count_down_timer_current,count_down_timer_end = collect_states_in_minute(time ,emotions_Dictionary,out )

            #count_down_timer_current + Decimal(1.0)
            out.writerow([name, emototionsResult["dominant_emotion"], current_time])

def collect_states_in_minute(count_down_timer_ends ,emotions_Dictionary,out):
    for name  in emotions_Dictionary:
        cnt = Counter(emotions_Dictionary[name])
        print("counter" ,cnt.most_common(1)[0][0])
        out.writerow([name, cnt.most_common(1)[0][0], count_down_timer_ends])

        return 0.0,0.0

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

def key_capture_thread():
    global keep_going
    global rgb

    print("Press any key then enter to exit")
    input()
    keep_going = False
    f.close()
   # video.release()
#    print(emotions_Dictionary)


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
#vs = VideoStream(src=0,usePiCamera=False).start()
#vs=cv2.VideoCapture(r"D:\Jops\Eval\WIN_20201227_16_49_50_Pro.mp4")
writer = None
time.sleep(0.2)
today = date.today()
d4 = today.strftime("%b-%d-%Y")
fileName = d4+"images Version" + ".csv"
f = open(fileName, 'w')
out = csv.writer(f, delimiter=",")
out.writerow(["name", "state", "time"])
#th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
#imagePaths = "D:\\Jops\\Done\\final class imotion detector\\dataset\\abdullrahman"
imagePaths = list(paths.list_images("dataset\\abdullrahman"))


from PIL import Image
import os, os.path

imgs = []
path = r"dataset\abdullrahman"
valid_images = [".jpg",".gif",".png",".tga"]
for fs in os.listdir(path):
    ext = os.path.splitext(fs)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,fs)))


# loop over frames from the video file stream
for i in imgs:

    print(i)
    frame = i
    print(frame)
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if True:
        rgb = np.array(frame.resize((400,400)))

        print(rgb)
        grayImage = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        height , width , layers=rgb.shape

        faces = classifier.detectMultiScale(rgb, scaleFactor=1.2)

        if len(faces)>0:
            cv2.imshow("Frame", rgb)
            cv2.waitKey(1)

            recognise_multithread(faces,rgb)


    else:

        break



f.close()
cv2.destroyAllWindows()

