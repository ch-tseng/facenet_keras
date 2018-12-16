# -*- coding: utf-8 -*-

import numpy as np
import os, time
import cv2
import imutils
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model

valid = "door_faces/valid/"
compare = "door_in_source.avi"

video_out = "door1.avi"
min_faceSize = (70, 70)
max_faceSize = (190, 190)
cascade_path = 'haarcascade_frontalface_alt2.xml'
min_score = 0.85
image_size = 160
tracker_type = "MOSSE"  #CSRT, MIL, KCF, MEDIANFLOW, MOSSE

#pretrained Keras model (trained by MS-Celeb-1M dataset)
model_path = 'model/facenet_keras.h5'
model = load_model(model_path)

#-----------------------------------------------------------------------------

def tracker(bbox, frame):
    trackerType =  cv2.Tracker_create(trackerType.upper())
    tracker = OPENCV_OBJECT_TRACKERS[trackerType]()

    ok = tracker.init(frame, bbox)

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

    faces = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=6)
    if(len(faces)>0):
        imgFaces = []
        bboxes = []
        for face in faces:
            (x, y, w, h) = face
            if(w>min_faceSize[0] and h>min_faceSize[1] and w<max_faceSize[0] and h<max_faceSize[1]):
                print("w,h=",w,h)
                faceArea = img[y:y+h, x:x+w]
                faceMargin = np.zeros((h+margin*2, w+margin*2, 3), dtype = "uint8")
                faceMargin[margin:margin+h, margin:margin+w] = faceArea

                cv2.imwrite("tmp/"+str(time.time())+".jpg", faceMargin)
                #aligned = resize(faceMargin, (image_size, image_size), mode='reflect')
                aligned = cv2.resize(faceMargin ,(image_size, image_size))
                #cv2.imwrite("tmp/"+str(time.time())+"_aligned.jpg", aligned)
                imgFaces.append(aligned)
                bboxes.append((x, y, w, h))

        if(len(bboxes)>0):
            return imgFaces, bboxes
        else:
            return None, None

    else:
        return None, None

def preProcess(img):
    whitenImg = prewhiten(img)
    whitenImg = whitenImg[np.newaxis, :]
    return whitenImg

#-------------------------------------------------

def face2name(face, faceEMBS, faceNames):
    #print(len(faceEMBS), len(faceNames))
    imgFace = preProcess(face)
    embs = l2_normalize(np.concatenate(model.predict(imgFace)))

    smallist_id = 0
    smallist_embs = 999
    for id, valid in enumerate(faceEMBS):
        distanceNum = distance.euclidean(embs, valid)
        print(distanceNum)
        if(smallist_embs>distanceNum):
            smallist_embs = distanceNum
            smallist_id = id

    return smallist_id, faceNames[smallist_id], smallist_embs

def draw_text(bbox, txt, img):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    fontSize = round(img.shape[0] / 676, 1)
    if(fontSize<0.35): fontSize = 0.35
    boldNum = int(img.shape[0] / 332)
    if(boldNum<1): boldNum = 1

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),boldNum)
    cv2.putText(img, txt, (x, y-(boldNum*3)), cv2.FONT_HERSHEY_COMPLEX, fontSize, (0,255,0), boldNum)

    return img

def displayImage(img, winName="People", width=640, write=True, waitTime=1):
    cv2.imshow(winName,imutils.resize(img, width=width))
    out.write(img)
    cv2.waitKey(waitTime)

# ----------------------------------------------------

valid_names = []
valid_embs = []

for img_file in os.listdir(valid):
    filename, file_extension = os.path.splitext(img_file)

    username = os.path.basename(filename)
    imgValid = cv2.imread(valid+img_file)
    #cv2.imshow("Valid", imgValid)
    #cv2.waitKey(0)
    aligned, _ = align_image(imgValid, 6)
    if(aligned is None):
        print("Cannot find any face in image: {}".format(img_file))
    else:
        faceImg = preProcess(aligned[0])
        embs = l2_normalize(np.concatenate(model.predict(faceImg)))
        valid_names.append(username)
        valid_embs.append(embs)

print("Valid names:", valid_names)

VIDEO_IN = cv2.VideoCapture(compare)
width = int(VIDEO_IN.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(VIDEO_IN.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(video_out,fourcc, 29.0, (int(width),int(height)))

if(len(valid_names)>0):
    i = 0
    hasFrame = True

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    trackStatus = 0

    while hasFrame:
        hasFrame, imgCompared = VIDEO_IN.read()
        frameDisplay = imgCompared.copy()
        aligned, faceBoxes = align_image(imgCompared, 6)

        if(faceBoxes is not None):
            if(len(faceBoxes)>0):
                #print("Face: " + str(len(faceBoxes)))
                for id, face in enumerate(faceBoxes):
                    #print("    face #"+str( i))
                    valid_id, valid_name, score = face2name(aligned[id], valid_embs, valid_names)
                    if(score<min_score):
                        print("Give this face the name....:" + valid_name + ","+str(score))
                        frameDisplay = draw_text(face, valid_name + "(" + str(round(score,2)) + ")", frameDisplay)

                        ok = tracker.init(imgCompared, face)
                        trackStatus = 1
                        displayImage(frameDisplay, winName="People", width=640, write=True, waitTime=1)
                        print("face area:", face)
                        while trackStatus==1 and hasFrame==True:
                            hasFrame, imgCompared = VIDEO_IN.read()
                            frameDisplay = imgCompared.copy()

                            ok, bbox = tracker.update(imgCompared)
                            print(ok, bbox)
                            if ok:
                                print("Tracking.....")
                                frameDisplay = draw_text(bbox, valid_name+ "(" + str(round(score,2)) + ")", frameDisplay)
                            else:
                                print("Lost tracking.....")
                                trackStatus = 0

                            displayImage(frameDisplay, winName="People", width=640, write=True, waitTime=1)

                    else:
                        displayImage(frameDisplay, winName="People", width=640, write=True, waitTime=1)
            else:
                displayImage(frameDisplay, winName="People", width=640, write=True, waitTime=1)

                i += 1

        displayImage(frameDisplay, winName="People", width=640, write=True, waitTime=1)


    out.release()

else:
    print("There is no any face in valid images.")
