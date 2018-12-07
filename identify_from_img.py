#Modified from https://github.com/nyoki-mtl/keras-facenet
import numpy as np
import os, time
import cv2
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model

valid = ["valid/Ariel.jpg", "valid/Hana.jpg", "valid/Janice.jpg", "valid/Jayptr.jpg", "valid/Michelle.jpg", "valid/Peter.jpg" ]
compare = ["test/IMG_0671.JPG" ] 
min_faceSzie = (160, 160)

#cascade_path = 'haarcascade_frontalface_default.xml'
cascade_path = 'haarcascade_frontalface_alt2.xml'

image_size = 160

#pretrained Keras model (trained by MS-Celeb-1M dataset)
model_path = 'model/facenet_keras.h5'
model = load_model(model_path)

#------------------------------------------------------------
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def align_image(img, margin):
    cascade = cv2.CascadeClassifier(cascade_path)

    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if(len(faces)>0):
        imgFaces = []
        bboxes = []
        for face in faces:
            (x, y, w, h) = face
            if(w>min_faceSzie[0] and h>min_faceSzie[1]):
                print("w,h=",w,h)
                faceArea = img[y:y+h, x:x+w]
                faceMargin = np.zeros((h+margin*2, w+margin*2, 3), dtype = "uint8")
                faceMargin[margin:margin+h, margin:margin+w] = faceArea

                cv2.imwrite("tmp/"+str(time.time())+".jpg", faceMargin)
                #aligned = resize(faceMargin, (image_size, image_size), mode='reflect')
                aligned = cv2.resize(faceMargin ,(image_size, image_size))
                cv2.imwrite("tmp/"+str(time.time())+"_aligned.jpg", aligned)
                imgFaces.append(aligned)
                bboxes.append((x, y, w, h))

        if(len(bboxes)>0):
            return imgFaces, bboxes
        else:
            return None

    else:
        return None

def preProcess(img):
    whitenImg = prewhiten(img)
    whitenImg = whitenImg[np.newaxis, :]
    return whitenImg

#-------------------------------------------------

names = []
nameFaces = []

for img_file in valid:
    filename, file_extension = os.path.splitext(img_file)
    username = os.path.basename(filename)
    imgValid = cv2.imread(img_file)
    aligned, _ = align_image(imgValid, 6)
    if(aligned is None):
        print("Cannot find any face in image: {}".format(imgValid))
    else:
        faceImg = preProcess(aligned[0])
        embs = l2_normalize(np.concatenate(model.predict(faceImg)))
        names.append(username)
        nameFaces.append(embs)

print("Names:", names)

for img_file in compare:
    filename, file_extension = os.path.splitext(img_file)
    imgCompared = cv2.imread(img_file)
    aligned, faceBoxes = align_image(imgCompared, 6)

    scoreList = []
    print("Image: ", filename)

    if(aligned is not None):
        scoreList = []

        for id, imgFace in enumerate(aligned):
            imgFace = preProcess(imgFace)
            embs = l2_normalize(np.concatenate(model.predict(imgFace)))

            smallist_id = 99999 
            smallist_distance = 99999

            for idUser, embsValid in enumerate(nameFaces):
                actualName = names[idUser]
                distanceNum = distance.euclidean(embsValid, embs)
                if(smallist_distance>distanceNum):
                    smallist_distance=distanceNum
                    smallist_id = idUser

            x = faceBoxes[id][0]
            y = faceBoxes[id][1]
            w = faceBoxes[id][2]
            h = faceBoxes[id][3]
            print("id:{}, bbox:({},{},{},{}, user:{})".format(id, x,y,w,h, idUser))
            fontSize = round(imgCompared.shape[0] / 576, 1)
            if(fontSize<0.35): fontSize = 0.35
            boldNum = int(imgCompared.shape[0] / 432)
            if(boldNum<1): boldNum = 1

            cv2.rectangle(imgCompared,(x,y),(x+w,y+h),(0,255,0),boldNum)
            cv2.putText(imgCompared, actualName, (x, y-(boldNum*3)), cv2.FONT_HERSHEY_COMPLEX, fontSize, (255,0,0), boldNum)

        cv2.imwrite(filename+"_"+actualName+".jpg", imgCompared)
        print("Write to "+filename+"_"+actualName+".jpg")
