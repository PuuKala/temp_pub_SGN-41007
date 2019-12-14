# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:58:27 2019

@author: ankan
"""

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import os
import numpy as np
from harkkatyo1 import read_class_names


def create_submission_scores(clf, train_gen, pred_gen):
    class_names = read_class_names("train\\train\\*\\*")
    retlist = []
    fid = int(f.split(os.sep)[-1].split('.')[0])
    X_mobile = model.predict(X[np.newaxis, ...])[0].reshape(1, -1)
    y_pred = clf.predict(X_mobile)[0]
    retlist.append({"Id": fid, "Category": class_names[y_pred]})
    df = pd.DataFrame(retlist)
    df = df.set_index("Id").sort_index()
    df.to_csv('submission_as2.csv')

#%% If main, gives error if run by itself, so this one is in it's own cell
# Train & evaluate, main 'function'
if __name__ == '__main__':

    #%% Construct classifiers cell
    convs = [MobileNet(input_shape=(224,224,3),include_top=False),
             MobileNetV2(input_shape=(224,224,3),include_top=False),
             InceptionV3(input_shape=(224,224,3),include_top=False)]
    
    clfs = []
    
    for i in convs:
        x = i.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(17, activation='softmax')(x)
        model = Model(inputs=i.input, outputs=predictions)
        for layer in i.layers:
            layer.trainable = False
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        clfs.append(model)
    
    
    #%% Image loading
    imgdatgen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2)
    
    train_gen = imgdatgen.flow_from_directory('train\\train',
                                           target_size=(224,224),
                                           batch_size=79,
                                           subset='training')
    valid_gen = imgdatgen.flow_from_directory('train\\train',
                                           target_size=(224,224),
                                           batch_size=79,
                                           subset='validation')
    
    
    #%% Classifier training with generators
    for i in clfs:
        i.fit_generator(
                train_gen,
                steps_per_epoch=800,
                epochs=4,
                validation_data=valid_gen,
                validation_steps=200)
        
        
    #%% Fit and make the submission with the best classifier
    imggen = imgdatgen.flow_from_directory('train\\train',
                                           target_size=(224,224),
                                           batch_size=79)
    pred_gen = imgdatgen.flow_from_directory('test\\',
                                             target_size=(224,224),
                                             batch_size=79,
                                             class_mode=None)
    create_submission_scores(clfs[0], imggen, pred_gen)