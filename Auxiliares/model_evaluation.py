import matplotlib.pyplot as plt

from sklearn.metrics import classification_report as cr
from sklearn.metrics import balanced_accuracy_score as ba
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay as cm_display

import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def plot_model_history(train_acc, val_acc, train_loss, val_loss):
    fig = plt.figure(figsize=(15,5))
    
    ax = plt.subplot(1, 2, 1)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title("Accuracies")
    plt.legend(['Train', 'Validation'])

    ax = plt.subplot(1, 2, 2)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title("Losses")
    plt.legend(['Train', 'Validation'])


def get_dataset_prediction(dataset, model):
    y_true = []
    y_pred = []

    for i in dataset:

        img_batch = tf.expand_dims(i[0], axis=0)
        pred = model.predict(img_batch)
        # applying classification rule
        pred = np.round(pred).astype(int)[0][0]
        true = i[1].numpy()

        y_true.append(true)
        y_pred.append(pred)
       

        print(pred, true)
    
    return y_true, y_pred


def evaluate_prediction(y_true, y_pred):
    
    print('')
    print('Balanced Accuracy:',ba(y_true, y_pred))
    print('')
    print('Regular Accuracy:', acc(y_true, y_pred))
    print('')
    print('Classification Report:')
    print(cr(y_true, y_pred))
    
    cm_matrix = cm(y_true, y_pred).astype(np.float32)
    
    disp = cm_display(cm_matrix)
    disp.plot()
    plt.savefig('confusion_matrix.png')
    plt.show()

