from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from linkedin_v2 import linkedin
from array import array
import os
from PIL import Image
import sys
import time
import requests
from io import BytesIO
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils import face_utils
from threading import Thread
import imutils
import dlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode


def anonymize_face_simple(image, factor=3.0):
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1
    # apply a Gaussian blur to the input image using our computed
    # kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[9]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[7]) # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar

if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
# Add your Computer Vision endpoint to your environment variables.
if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
else:
    print("\nSet the COMPUTER_VISION_ENDPOINT environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
    
computervision_client = ComputerVisionClient("https://bikrampaul.cognitiveservices.azure.com/", CognitiveServicesCredentials("85637546b7264a5ba1dce4f4b8b35e7d"))
remote_image_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/landmark.jpg"
'''
Describe an Image - remote
This example describes the contents of an image with the confidence score.
'''
        
        
'''
Detect Faces - remote
This example detects faces in a remote image, gets their gender and age, 
and marks them with a bounding box.
'''
#print("===== Detect Faces - remote =====")
# Get an image with faces
#remote_image_url_faces = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/faces.jpg"
# Select the visual feature(s) you want.
#remote_image_features = ["faces"]
# Call the API with remote URL and features
#detect_faces_results_remote = computervision_client.analyze_image(remote_image_url_faces, remote_image_features)

# Print the results with gender, age, and bounding box
#print("Faces in the remote image: ")
#if (len(detect_faces_results_remote.faces) == 0):
#    print("No faces detected.")
#else:
#    for face in detect_faces_results_remote.faces:
#        print("'{}' of age {} at location {}, {}, {}, {}".format(face.gender, face.age, \
#        face.face_rectangle.left, face.face_rectangle.top, \
#        face.face_rectangle.left + face.face_rectangle.width, \
#        face.face_rectangle.top + face.face_rectangle.height))
        
API_KEY = '86mf11anljqsfe'
API_SECRET = 'g0rxpjazY6gsH41v'
RETURN_URL = 'https://api-university.com'

authentication = linkedin.LinkedInAuthentication(API_KEY, API_SECRET, RETURN_URL, linkedin.PERMISSIONS.enums.values())
# Optionally one can send custom "state" value that will be returned from OAuth server
# It can be used to track your user state or something else (it's up to you)
# Be aware that this value is sent to OAuth server AS IS - make sure to encode or hash it
#authorization.state = 'your_encoded_message'
print (authentication.authorization_url)  # open this url on your browser
application = linkedin.LinkedInApplication(authentication)
#bikram's authentication code
authentication.authentication_code = 'AQRiRMpG1XfjaRF4bqC9oncAVknCanLCuov-5gZgfuiNRUUbqlbI3lgP8Se34gWRMRxxiurBD6_HQ-MWdJknW2K0Vv081jwllVpAmEGzVieh-QFFbNqTgwkWxJbzulBsgCboR-54lsJnDo5YZB0zZVn2gGsQeyT1Lh9V01_pqnvERFzCX0QvDVTCisjeVg'
#hitesh's authentication code
#authentication.authentication_code = 'AQQdLTIKgEtdQoQvUBd7n90e5zhLisrLitXfrFOR11dZKEk3RiUj2ctNHgRtWczM3RwMVQLfTMjY7Jk9ROiYA2usl-oTTT3or4c8qKiEinnx2_AYJwgp3mrB-5482N2sPGm5dcUB0GUHCtmdANaPDfk2KDwzH3gcJNBpSE4FlrBlrgwbWtk1E24E5QtoKQ'
#bikram's access token
application = linkedin.LinkedInApplication(token='AQWXUgHMAwl_htHJ_NareR8thl_Z-_1cAOMg65dS9dkpikMnVxgZjiipk8Uuv2w3f0O_5awiGzA_xiq64nSQBwcmPygphqtC4-rTpVJmmnyMKTTTW1micVfnVQ4A6uT0gGrNlbR4DLYDmEgQIbMOVbjjm9tsYjBEMFj3snVTIll6FuAs-s0IW-CITfhsm2-_1VYo0hiBTz2FyZphPcAR2H7kDqeHifvrrfZ-BSpwvZXHCRUMDjx-59v9wEleM7VkOQorBpwpFoUs3WLbEeatKQbQ8kjA80W2vX1nnKzo1uy6Hi6-COoI47v0IbOCCKHT2Iu-bl6BltI0M5N-64EPcjEuj4ndQg')
#print(application)
data = application.get_profile()
#print(data)
data1 = data['profilePicture']
linkedinphoto = data1['displayImage']
print("Detecting face")
#print(linkedinphoto)
#r = requests.get('https://api.linkedin.com/v2/me?oauth2_access_token=AQWXUgHMAwl_htHJ_NareR8thl_Z-_1cAOMg65dS9dkpikMnVxgZjiipk8Uuv2w3f0O_5awiGzA_xiq64nSQBwcmPygphqtC4-rTpVJmmnyMKTTTW1micVfnVQ4A6uT0gGrNlbR4DLYDmEgQIbMOVbjjm9tsYjBEMFj3snVTIll6FuAs-s0IW-CITfhsm2-_1VYo0hiBTz2FyZphPcAR2H7kDqeHifvrrfZ-BSpwvZXHCRUMDjx-59v9wEleM7VkOQorBpwpFoUs3WLbEeatKQbQ8kjA80W2vX1nnKzo1uy6Hi6-COoI47v0IbOCCKHT2Iu-bl6BltI0M5N-64EPcjEuj4ndQg')
#getdata = r.json()
#print(getdata)
#d = requests.get('https://api.linkedin.com/v2/me?oauth2_access_token=AQWXUgHMAwl_htHJ_NareR8thl_Z-_1cAOMg65dS9dkpikMnVxgZjiipk8Uuv2w3f0O_5awiGzA_xiq64nSQBwcmPygphqtC4-rTpVJmmnyMKTTTW1micVfnVQ4A6uT0gGrNlbR4DLYDmEgQIbMOVbjjm9tsYjBEMFj3snVTIll6FuAs-s0IW-CITfhsm2-_1VYo0hiBTz2FyZphPcAR2H7kDqeHifvrrfZ-BSpwvZXHCRUMDjx-59v9wEleM7VkOQorBpwpFoUs3WLbEeatKQbQ8kjA80W2vX1nnKzo1uy6Hi6-COoI47v0IbOCCKHT2Iu-bl6BltI0M5N-64EPcjEuj4ndQg&projection=(id,firstName,lastName,profilePicture(displayImage~:playableStreams))')

#Bikram's access token
#getdata1 = requests.get('https://api.linkedin.com/v2/me?oauth2_access_token=AQWXUgHMAwl_htHJ_NareR8thl_Z-_1cAOMg65dS9dkpikMnVxgZjiipk8Uuv2w3f0O_5awiGzA_xiq64nSQBwcmPygphqtC4-rTpVJmmnyMKTTTW1micVfnVQ4A6uT0gGrNlbR4DLYDmEgQIbMOVbjjm9tsYjBEMFj3snVTIll6FuAs-s0IW-CITfhsm2-_1VYo0hiBTz2FyZphPcAR2H7kDqeHifvrrfZ-BSpwvZXHCRUMDjx-59v9wEleM7VkOQorBpwpFoUs3WLbEeatKQbQ8kjA80W2vX1nnKzo1uy6Hi6-COoI47v0IbOCCKHT2Iu-bl6BltI0M5N-64EPcjEuj4ndQg&projection=(id,firstName,lastName,profilePicture(displayImage~:playableStreams))')

#hitesh's access token
#getdata1 = requests.get('https://api.linkedin.com/v2/me?oauth2_access_token=AQVSwk-08Q6ytHmLp7vRehjPTD1ZQCsiS28TjcMyd3OqDLuIMwCwnSmivbTW83fWVuXPdjHtM8JRDUoxCVY6o5bd2pXZ-axnDODiZmn3-BmtZyBsvDFKrH55PODDfaXOKlu4k-R9pQgIAxZFevWWxIJLq_POsVNf-qQjGu-MKZKb2LQvyRWPot5M8qJLt_oVVp2O-zSDfbJSulfllVzBfzxyTT9r7LfJkm_JonrL_vrDTINY4ITXl5zaLKcf3ZMwcfvzYtwut8Bax-FDtWbszslTG6btCMFWYtq2E9YM3DXaFkySrG-FYZ3RMpEyRNNOTJhm1n1RvtfjSUA0qPVuVakn-Mu4FQ&projection=(id,firstName,lastName,profilePicture(displayImage~:playableStreams))')

#abhi's access token
getdata1 = requests.get('https://api.linkedin.com/v2/me?oauth2_access_token=AQVx0oAasQW4cLdrEtrHoDwDNUFgtoqkIeSlktEhJZIsEXPVNVY6gubIm5b-jZ0aARBubCPmvNyS7y4BZ-j4fyfLGtSbyQwJPhLA5v9DZFOAEQi3ngCCxHn_1zClK_6aL5II6jcZ6B1m4527St2UF3bfVNK6T9n8i1h0bkroWhQokpprsY1pqJLDhesoclSl1DHCx9uvEX_eb1jzzm_ECHxNz7lYksO49Dw_JN3cERcAxOiHmKaA_d4dM1S3Z9kON7QXwMA0PKwhSsMr2N_V7mXzfpKUVD7umdIGmPKE4WrztNZQZtBJvUVDQcdOqnFeUphKizgDxvU5RtlmNFiYMNTmZ6vUfw&projection=(id,firstName,lastName,profilePicture(displayImage~:playableStreams))')

getdata1 = getdata1.json()
#print(getdata1)

x = getdata1['firstName']
x = x['localized']
x = x['en_US']
y = getdata1['lastName']
y = y['localized']
y = y['en_US']
name = x + y
a=getdata1["profilePicture"]
b=a['displayImage~']
c=b['elements']
d=c[3]
#print(d)
e=d['identifiers']
f=e[0]
linkedinphoto=f['identifier']
i = d['data']
i=i["com.linkedin.digitalmedia.mediaartifact.StillImage"]
j=i['displaySize']
imagewidth=j['width']
imageheight=j['height']
remote_image_url_faces = linkedinphoto
remote_image_features = ["faces"]
image = Image.open(BytesIO(requests.get(linkedinphoto).content))
plt.imshow(image)
plt.axis("off")
_ = plt.title(name, size="x-large", y=-0.1)
plt.show()


# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

original = image
scale_percent = 95
width = int(imagewidth * scale_percent / 100)
height = int(imageheight * scale_percent / 100)
dim = (width, height)
# resize image
image=np.array(image)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)   
## define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.5

## initialize dlib's face detector (HOG-based) and then create
## the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

## grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)


if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    model.save_weights('model.h5')

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    #cap = cv2.VideoCapture(0)
    #while True:
        # Find haar cascade to draw bounding box around face
        #ret, frame = cap.read()
        #if not ret:
        #    break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detect_faces_results_remote = computervision_client.analyze_image(linkedinphoto, remote_image_features)
    print("Faces in the remote image: ")
    

    tempImg = resized.copy()
    faces = facecasc.detectMultiScale(tempImg,scaleFactor=1.3, minNeighbors=5)
    if (len(detect_faces_results_remote.faces) == 0):
        print("No faces detected.")
    else:
        for face in detect_faces_results_remote.faces:
            print(face)
            print("'{}' of age {} at location {}, {}, {}, {}".format(face.gender, face.age, \
            face.face_rectangle.left, face.face_rectangle.top, \
            face.face_rectangle.left + face.face_rectangle.width, \
            face.face_rectangle.top + face.face_rectangle.height))
        tempImg = resized.copy()
        faces = face_cascade.detectMultiScale(tempImg,scaleFactor=1.5,minNeighbors=5,minSize=(30, 30))
        maskShape = (resized.shape[0], resized.shape[1], 1)
        mask = np.full(maskShape, 0, dtype=np.uint8)
        for (x,y,w,h) in faces:
            x_center = int(round(x + (w/2)))
            y_center = int(round(y + (h/2)))
            tempImg = cv2.blur(tempImg,(50,50))
            cv2.circle(resized,(x_center,y_center),int(h/2),(255,0,0))
            cv2.circle(mask,(int((x+x+w)/2),int((y+y+h)/2)),int(w/2),(255,0,0),-1)
            mask_inv = cv2.bitwise_not(mask)
            background_img = cv2.bitwise_and(tempImg,tempImg,mask=mask_inv)
            foreground_img = cv2.bitwise_and(resized,resized,mask=mask)
            dst = cv2.add(foreground_img,background_img)
            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            cv2.rectangle(dst, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(dst, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #plt.imshow(dst)
            #plt.show()
            if ((w*h)<((resized.shape[0]*resized.shape[1])*0.5)):
                print("Face quality is too small")
            else:
                print("Face quality is okay")
            tempImg = dst.copy()
            gray = cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY)
        ## detect faces in the grayscale frame
            rects = detector(gray, 0)
            for rect in rects:
        ## determine the facial landmarks for the face region, then
        ## convert the facial landmark (x, y)-coordinates to a NumPy
        ## array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

        ## extract the mouth coordinates, then use the
        ## coordinates to compute the mouth aspect ratio
                mouth = shape[mStart:mEnd]
                mar = mouth_aspect_ratio(mouth)

        ## compute the convex hull for the mouth, then
        ## visualize the mouth
                mouthHull = cv2.convexHull(mouth)

                cv2.drawContours(dst, [mouthHull], -1, (0, 255, 0), 1)
                
        ## Draw text if mouth is open
                if mar > MOUTH_AR_THRESH:
                    cv2.putText(dst, "Teeth is visible", (30,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                    #plt.imshow(dst)
                    #plt.show()
                else:
                    print("Teeth is not visible")
            plt.imshow(dst)
            plt.axis("off")
            _ = plt.title(name, size="x-large", y=-0.1)
            plt.show()

tags_result_remote = computervision_client.tag_image(linkedinphoto)
print("Tags in the remote image")
if(len(tags_result_remote.tags)==0):
    print("No tags detected")
else:
    for tag in tags_result_remote.tags:
        print("'{}' with confidence {:.2f}%".format(tag.name, tag.confidence*100))

print("===== Describe an image - remote =====")
# Call API
description_results = computervision_client.describe_image(linkedinphoto )

# Get the captions (descriptions) from the response, with confidence level
print("Description of remote image: ")
if (len(description_results.captions) == 0):
    print("No description detected.")
else:
    for caption in description_results.captions:
        print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))
        
#codes taken from
#https://github.com/mauckc/mouth-open
#https://github.com/omar178/Emotion-recognition
#https://github.com/atulapra/Emotion-detection