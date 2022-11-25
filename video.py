import cv2
import numpy as np
import tensorflow as tf
import os

# used to detect faces in a picture
facec = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#choose font
font = cv2.FONT_HERSHEY_SIMPLEX
# load model
model = tf.keras.models.load_model("New_Model.h5")

class Model():
    
    def preprocess(img):
        return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow(img, batch_size=32, shuffle=True)
    
    def model(img):
        emotionsdict = {0:'disgusted', 1:'happy', 2:'surprised', 3:'neutral', 4:'sad', 5:'angry', 6:'fearful'}
        roi = cv2.resize(img, (227, 227))
        img = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(Model.preprocess(img))
        pred = emotionsdict[list(pred[0]).index(max(list(pred[0])))]
        return pred
        
class VideoCamera(object):

    #emotions = ['disgusted' 'happy' 'surprised' 'neutral' 'sad' 'angry' 'fearful']
    
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        

    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        _, fr = self.video.read()
        fr = cv2.flip(fr, 1)
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 3, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            # predict using model 
            pred = Model.model(fc)

            #put text and rectangles
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            #add emojis based on emotion
            emoji = cv2.imread(f"static/images/emotions/{pred}.png")
            emoji = cv2.resize(emoji, (100,100))
            fr[100:200, :100] = cv2.addWeighted(fr[100:200, :100], 0, emoji, 1, 0)
            
        #_, jpeg = cv2.imencode('.jpg', fr)
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
