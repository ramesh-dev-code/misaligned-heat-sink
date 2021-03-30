'''
Created on Dec 24, 2020

@author: RAMESH
'''
import cv2 as cv
import tensorflow as tf
import numpy as np
import pathlib
from cProfile import label
'''
class_name = "FAIL"
class_label = {'FAIL':0,'PASS':1}
gt = class_label[class_name] # Ground Truth
miss = 0 # Incorrect result
model = tf.keras.models.load_model('/home/test/workspace/HeatSink/src/checkpoints/best_model_18-03-2021_10:24:52.h5')
test_dir = list(pathlib.Path('/home/test/Documents/DeepLearning/Classification/AOI/HeatSink/Testing/Images/Raw/{}/'.format(class_name)).glob('*.jpg'))
img_count = len(test_dir)
print('Image Count: ',img_count)
imagenet_mean = np.array([123.68, 116.779, 103.939], dtype="float32") # Imagenet
res_str = ['FAIL','PASS']
k = 1
for imgPath in test_dir :
    img = cv.imread(str(imgPath))        
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)    
    img = tf.image.resize(img,[224,224],antialias=True)
    img = img.numpy()
    for c in range(3):
        img[:,:,c] = img[:,:,c] - imagenet_mean[c]
    img = np.expand_dims(img, axis=0)         
    yhat = model.predict(img)        
    yhat = yhat[0].tolist()    
    max_ind = yhat.index(max(yhat))    
    if max_ind != gt:
        print('Image:',k,',Prob:', yhat,',Result:',res_str[max_ind])
        miss = miss + 1        
    k = k + 1    
print('Incorrect count: ', miss)
accuracy = img_count/(miss + img_count) 
print('Accuracy is ', accuracy)
'''

class_label = {'FAIL':0,'PASS':1}
fp = 0 # False Positive
fn = 0 # False Negative
model = tf.keras.models.load_model('/home/test/workspace/HeatSink/src/checkpoints/best_model_18-03-2021_10:04:30.h5')
imagenet_mean = np.array([123.68, 116.779, 103.939], dtype="float32") # Imagenet
res_str = ['FAIL','PASS']
img_count = [0,0]
for label_name in res_str :
    print('Class Name: ', label_name)
    gt = class_label[label_name] # Ground Truth
    label_dir = list(pathlib.Path('/home/test/Documents/DeepLearning/Classification/AOI/HeatSink/Testing/Images/Raw/{}/'.format(label_name)).glob('*.jpg'))
    img_ind = 1
    miss = 0
    img_count[gt] = len(label_dir)
    print('Number of {} images: {}'.format(label_name,img_count[gt]))
    for imgPath in label_dir :        
        img = cv.imread(str(imgPath))        
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)    
        img = tf.image.resize(img,[224,224],antialias=True)
        img = img.numpy()
        for c in range(3):
            img[:,:,c] = img[:,:,c] - imagenet_mean[c]
        img = np.expand_dims(img, axis=0)         
        yhat = model.predict(img)        
        yhat = yhat[0].tolist()    
        max_ind = yhat.index(max(yhat))                        
        if max_ind != gt:
            print('Image:',img_ind,',Prob:', yhat,',Result:',res_str[max_ind])
            miss = miss + 1                                
        img_ind = img_ind + 1
    if gt == 0:
        fp = miss
        print("Number of FPs: ",fp)
    else:
        fn = miss
        print("Number of FNs: ",fn)                

tp = img_count[1] - fp
tn = img_count[0] - fn 
accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
precision = 100 * tp / (tp + fp)
fpr = 100 * fp / (tn + fp) 
sensitivity = 100 * tp / (tp + fn)
print('Sensitivity: {:.2f}%'.format(sensitivity))
print('Accuracy: {:.2f}%'.format(accuracy))
print('Precision: {:.2f}%'.format(precision))
print('FPR: {:.2f}%'.format(fpr))
        