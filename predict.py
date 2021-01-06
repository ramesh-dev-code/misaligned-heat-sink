'''
Created on Dec 24, 2020

@author: RAMESH
'''
import cv2 as cv
import tensorflow as tf
import numpy as np
from datetime import datetime


model = tf.keras.models.load_model('/home/test/workspace/HeatSink/src/checkpoints/best_model_06-01-2021_13:11:20.h5')
train_mean = np.array([123.68, 116.779, 103.939], dtype="float32") # Imagenet
#train_mean = np.array([110.03, 114.15, 115.04], dtype="float32") # Heat Sink- Old
#train_mean = np.array([107.69, 111.02, 112.61], dtype="float32") # Heat Sink - New
cap_frame = cv.VideoCapture("/dev/video0")
cv.namedWindow("Test")

im_name = '/home/test/workspace/HeatSink/src/test_image.jpg'
while (cap_frame.isOpened()) :
    success,frame = cap_frame.read()
    if not success:
        print('Frame read failed')
        break
    cv.imshow("Test",frame)
    k = cv.waitKey(1)
    # Press Esc to close the input image 
    if k%256 == 27:
        break
    elif k%256 == 32:                           
        #cv.imwrite(im_name,frame)
        print('Before resizing',frame.shape,type(frame))
        t1 = datetime.now()
        # Convert BGR into RGB image
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Resizing the image into 224 x 224 
        img = tf.image.resize(frame,[224,224],antialias=True).numpy()
        print('After resizing: ',img.shape,type(img))        
        # Subracting the per-channel mean from the captured frame
        for c in range(3):
            img[:,:,c] = img[:,:,c] - train_mean[c] 
        cv.imwrite(im_name,img)
        cv.imshow("Preprocessed Image",img)
        k2 = cv.waitKey(0)
        # Press Enter to close the preprocessed image
        if k2%256 == 13:
            cv.destroyWindow("Preprocessed Image")              
        img = np.expand_dims(img, axis=0)                        
        print('After data augmentation: ',img.shape,type(img))             
        yhat = model.predict(img)        
        yhat = yhat[0].tolist()
        print(yhat)
        if yhat[0] > yhat[1]:
            print('Result: FAIL')
        else:
            print('Result: PASS')       
        td = (datetime.now()-t1).total_seconds()
        print('Execution Time: {} seconds'.format(td))             

cap_frame.release()
cv.destroyAllWindows()
