import os
import pickle

import cv2
import face_recognition
from imutils import paths

# تحديد مكان الاوجوه في الماخوذه مسبقا
dataset = "dataset"

# تحديد اسم ال model المخصص للالتقاط الاوجه
encodingsPath = "model_ibrahem.pickle"

# الطريقه المستخدمه لتدريب ال model في هذه الحاله نستخد CNN لضمان ضقه اكثر ولكن لضمان استخراج اسرع نستخدم hog
model = "cnn"

# الحصول على مسارات الصور داخل الملف الذي يحتوي على الاوجه
imagePaths = list(paths.list_images(r"E:\dataset"))

# تعريف القوائم اللتي ستحتوي على قيم الصور واسماء الاشخاص المتعارف عليهم
knownEncodings = []
knownNames = []

# نمر على كل صوره لدينا
for (i, imagePath) in enumerate(imagePaths):
    print("processing image {}/{}".format(i + 1,
                                          len(imagePaths)))
    # استخرج اسم الشخص
    print(imagePath)
    name = imagePath.split(os.path.sep)[-2]
    print(name)

    # نحمل  الصوره ونحولها للصيغه الاصليه لان ال CV2 تحول الصور الى BGR ولذلك وجب التحويل
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # تحديد الاحداثيات للاوجه في كل وجه داخلالصوره
    boxes = face_recognition.face_locations(rgb,model=model)

    # تحويل كل صوره لمجموعه من المعطيات للتعارف عليها
    encodings = face_recognition.face_encodings(rgb, boxes)

    # نمر على جميع المعطيات
    for encoding in encodings:
        # تحميل المعطيات واسمائها داخل القوائم
        knownEncodings.append(encoding)
        knownNames.append(name)

# عند الانتهاء حمل ال model واحفظه
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encodingsPath, "wb")
f.write(pickle.dumps(data))
f.close()
