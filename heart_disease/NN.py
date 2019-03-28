import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation 
from keras.layers import LeakyReLU,BatchNormalization
import keras
from keras import regularizers

def NN_train(model,feature,label):
    # add linear layer 、 batchnormalization Dropout using relu activation softmax activation
    model.add(Dense(26,input_dim=13,activation='linear',kernel_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.3))
    model.add(Dense(10, init='uniform',activation='linear',kernel_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(alpha=.001))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(10, init='uniform',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(feature,label,batch_size=30,epochs=500,verbose = 0,validation_split=0.1) 
    # prevent overfitting : Batch normalization 、Drop out


def main():
    model = Sequential()
    feature = pd.read_csv("heart_main.csv") 
    temp_label = pd.read_csv("heart_concat.csv") 
    label = keras.utils.to_categorical(temp_label, 2)
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.33, random_state=42)
    NN_train(model, feature_train ,label_train)
    loss = model.evaluate(feature_test, label_test)
    final = model.predict(feature_test)
    final = np.array(final)
    pridict = np.argmax(final,axis=1)
    label_test = np.argmax(label_test,axis = 1)
    nls = np.equal(pridict,label_test)
    t=0
    f= 0
    for i in nls:
        if i ==1:
            f=f+1
        t=t+1
    print(f/t,loss)
    

if __name__ == '__main__':
    main()