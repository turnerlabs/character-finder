from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam

map_characters = {0: 'Santa', 1:'NoSanta'}
pic_size = 64
num_classes = len(map_characters)
test_size = 0.15
pictures_per_class = 1000
epochs = 100    
batch_size = 32

def load_pictures(train=False,validation=False,test=False):
    pics = []
    labels = []
    if validation:
        for k,char in map_characters.items():
            pictures = [k for k in glob.glob('./validation%s/*'%char)]
            #nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
            for pic in pictures:
                a = cv2.imread(pic)
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                a = cv2.resize(a, (pic_size,pic_size))
                pics.append(a)
                labels.append(char)
                #print labels
        return np.array(pics), np.array(labels)

    if test:
        picName = []
        for k,char in map_characters.items():
            pictures = [k for k in glob.glob('./test%s/*'%char)]
            #nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
            for pic in pictures:
                a = cv2.imread(pic)
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                a = cv2.resize(a, (pic_size,pic_size))
                pics.append(a)
                labels.append(char)
                picName.append(pic)
        return np.array(pics), np.array(labels), np.array(picName)

    if train:
        for k,char in map_characters.items():
            pictures = [k for k in glob.glob('./%s/*'%char)]
            #nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
            for pic in pictures:
                a = cv2.imread(pic)
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                a = cv2.resize(a, (pic_size,pic_size))
                pics.append(a)
                labels.append(char)
                #print labels
        return np.array(pics), np.array(labels)

def get_dataset(train=False,validation=False,test=False):
    if train or validation:
        X, y = load_pictures(train,validation,test)
    elif test:
        X, y, picName = load_pictures(train,validation,test)
    y = pd.factorize(y)
    y = keras.utils.to_categorical(y[0], num_classes)
    X = X.astype('float32') / 255.
    if train or validation:
        return X,y
    else:
        return X,y,picName

def create_model_six_conv(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt

def training(model, X_train, X_validate, y_train, y_validate):
    history = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_validate, y_validate),shuffle=True)
    return model, history


if __name__ == '__main__':
    X_train, y_train = get_dataset(train=True)
    model, opt = create_model_six_conv(X_train.shape[1:])
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    X_validate, y_validate = get_dataset(validation=True)
    model, history = training(model, X_train, X_validate, y_train, y_validate)
    X_test, y_test, picName = get_dataset(test=True)
    score = model.predict(X_test)
    #print score
    #print y_test
    b = np.zeros_like(y_test)
    b[np.arange(len(y_test)), score.argmax(1)] = 1
    # print len(y_test)
    # print len(b)
    # num = (y_test==b).sum()
    # print num
    # acc = num/len(y_test)
    # print acc
    rightSanta = 0
    right = 0
    santa = 0
    colNames = ['ImageName','Predicted','Actual','Confidence']
    result = pd.DataFrame(columns=colNames)
    for i in range(0,len(y_test)):
        imgName = picName[i].rsplit("/",1)[1]
        if y_test[i,0]==1:
            actual = 'Santa'
            santa = santa + 1
        elif y_test[i,1]==1:
            actual = 'NotSanta'

        if(b[i,0]==1):
            predicted = 'Santa'
            Confidence = score[i,0]
        elif(b[i,1]==1):
            predicted = 'NotSanta'
            Confidence = score[i,1]

        if actual == predicted:
            right = right + 1

        if actual == predicted and predicted == 'Santa':
            rightSanta = rightSanta + 1

        dfToAppend = pd.DataFrame([[imgName,predicted,actual,Confidence]],columns=colNames)
        result = result.append(dfToAppend)

    y_actual = pd.Series(y_test[:,0])
    y_pred = pd.Series(b[:,0])
    y_actual.replace(1,'Santa',inplace=True)
    y_actual.replace(0,'NotSanta',inplace=True)
    y_pred.replace(1,'Santa',inplace=True)
    y_pred.replace(0,'NotSanta',inplace=True)
    df_confusion = pd.crosstab(y_actual, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    totalAcc = right/len(y_test)
    santaAcc = rightSanta/santa
    result.reset_index(inplace=True,drop=True)
    print result
    print '\n\nOverall Accuracy'
    print totalAcc*100
    print '\n\nSanta Detecting Accuracy'
    print santaAcc*100
    print '\n\nConfusion Matrix'
    print df_confusion
    print '\n\n'
