import pandas as pd
import numpy as np
import keras
import keras.models
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.layers import LeakyReLU
# normalize the data
from sklearn import preprocessing

#DNN

def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=29, activation=LeakyReLU(),kernel_regularizer=l2()))
    # model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(32, activation=LeakyReLU(),kernel_regularizer=l2(0.05)))
    # model.add(Dropout(0.3))
    model.add(Dense(32, activation=LeakyReLU(),kernel_regularizer=l2(0.05)))
    model.add(Dense(16, activation=LeakyReLU(),kernel_regularizer=l2(0.05)))
    # model.add(Dropout(0.3))
    
    model.add(Dense(1, activation='sigmoid',kernel_regularizer=l2(0.05)))
    adad = optimizers.Adamax(lr=0.005)
    model.compile(loss='binary_crossentropy', optimizer=adad, metrics=['accuracy'])
    return model

def plot(acc, history): 
    print('Accuracy: %.2f' % (acc*100))
    # model.save('DNN_model_1.h5')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



def main():
    # dataset = pd.read_csv("dataset/death.csv")
    # model = keras.models.load_model("model/model_1.h5")
    # data load
    raw_data = pd.read_csv("dataset/mean_train_final.csv")
    X = raw_data[['nlr', 'lactate', 'bcd', 'GENDER', 'age', 'DISCHARGE_LOCATION', 'Neoplastic_disease', 'LIVER_DISEASE', 'CONGESTIVE_HEART_FAILURE', 'OTHER_NEUROLOGICAL', 'RENAL_FAILURE', 'total', 'resp_rate', 'sys_bp', 'temp', 'hr', 'bun', 'sodium', 'glucose_blood', 'hematocrit', 'o2', 'glucose_pleural', 'pH', 'potassium', 'cre', 'wbc', 'platelets', 'alb', 'row_num']]
    y = np.ravel(raw_data[['die_in_h']])
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # pre = model.predict(X)
    # print(pre[:10], y[:10])
    batch_size = 16
    nb_epoch = 500
    np.random
    model = build_model()
    history = model.fit(x=X, y=y, epochs=nb_epoch, batch_size=batch_size, verbose=1, validation_split=0.1)
    _, accuracy = model.evaluate(X, y)   
    plot(accuracy, history)
    model.save("model/model_1.h5")


main()
