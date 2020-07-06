#import library
import numpy as np
from skimage import color, exposure, transform, io
import os
import glob
import pandas as pd
import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
#set parameters 
NUM_CLASSES = 43
IMG_SIZE = 48
model_path='vgg16_10_48_0.01.h5'
def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    return img
def test_model(model):
    test = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')
    # Load test dataset
    X_test = []
    y_test = []
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # predict and evaluate
    y_pred = model.predict_classes(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:')
    print(cnf_matrix) 
    return cnf_matrix
def load_model(model_path='vgg16_10_48_0.01.h5'):
    model=keras.models.load_model(model_path)
    return model

model = load_model()
cnf_matrix= test_model(model)
#plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
class_names = [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization',cmap=plt.cm.Blues)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',cmap=plt.cm.Blues)

plt.show()
