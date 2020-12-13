import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing, metrics

def read_data():
    df_data = pd.read_csv('dataset/mean_train_final.csv')
    data_list = df_data.values.tolist()
    return df_data, data_list


def try_different_method(model,x_train,x_test,y_train,y_test, model_name):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, result)
    print('model:',model)
    print('score:',score)
    print('accuracy:',accuracy)
    print(result)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('%s\nscore: %f'%(model_name, score))
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    df_data, data_list = read_data()

    df_data.fillna(0, inplace=True)

    data_x = df_data[['nlr', 'lactate', 'bcd', 'GENDER', 'age', 'DISCHARGE_LOCATION', 'Neoplastic_disease', 'LIVER_DISEASE', 'CONGESTIVE_HEART_FAILURE', 'OTHER_NEUROLOGICAL', 'RENAL_FAILURE', 'total', 'resp_rate', 'sys_bp', 'temp', 'hr', 'bun', 'sodium', 'glucose_blood', 'hematocrit', 'o2', 'glucose_pleural', 'pH', 'potassium', 'cre', 'wbc', 'platelets', 'alb', 'row_num','day_of_die','intubation',]]
    data_y = df_data['die_in_h']

    print(df_data.head())

    
    x_train,x_test,y_train,y_test = train_test_split(data_x, data_y, test_size = 0.33)
    dec_tree = ensemble.RandomForestClassifier(n_estimators=20, max_depth=8) 
    


    # model_LinearRegression = linear_model.LinearRegression()
    # model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
    # model_DecisionTreeClassifier = tree.DecisionTreeClassifier()
    # model_SVR = svm.SVR()
    model_RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators=20, max_depth=8) 

    # try_different_method(model_LinearRegression,x_train,x_test,y_train,y_test, 'LinearRegression')
    # try_different_method(model_KNeighborsRegressor,x_train,x_test,y_train,y_test, 'KNeighborsRegressor')
    try_different_method(model_DecisionTreeClassifier,x_train,x_test,y_train,y_test, 'DecisionTreeClassifier')
    # try_different_method(model_SVR,x_train,x_test,y_train,y_test, 'SVR')
    try_different_method(model_RandomForestClassifier,x_train,x_test,y_train,y_test, 'RandomForestClassifier')

    model_RandomForestClassifier = cross_val_score(model_RandomForestClassifier,data_x,data_y,cv=10)

    superpa = []
    for i in range(50):
        rfc = ensemble.RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
        rfc_s = cross_val_score(rfc,data_x,data_y,cv=10).mean()
        superpa.append(rfc_s)
        print(i)
    print(max(superpa),superpa.index(max(superpa))+1)#打印出：最高精確度取值，max(superpa))+1指的是森林數目的數量n_estimators
    plt.figure(figsize=[20,5])
    plt.plot(range(1,51),superpa)
    plt.show()