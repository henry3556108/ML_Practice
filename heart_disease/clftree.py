import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
def ReadData():
    feature = pd.read_csv('heart_main.csv')
    label = pd.read_csv('heart_concat.csv')
    return feature,label

# def DataProccess():   
#     data = pd.concat([data,target],axis=1)
#     dic={20:0,30:0,40:0,50:0,60:0,70:0}
#     ntemp=data["age"]
#     
#     for id,i in enumerate(ntemp):
#         ls[id]=int(i/10)*10
#         dic[int(i/10)*10]=dic[int(i/10)*10]+1
#     ls = pd.DataFrame(ls,columns=["nage"])
#     ndata=pd.concat([data,target],axis=1)
#     y=ndata['target']
#     x = ndata.drop(['age','target'],axis = 1)
#     y_test= np.array(y_test)
#     total = 0
#     t=0
#     for index,i in enumerate(final):
#         # print(type(y_test[index]))
#         if int(i)== int(y_test[index])==1:
#             t = t +1
#         if y_test[index]==1:
#             total = total + 1
#     print(t/total)
#     # split=[0,.1,0,0,0,0]
#     # templs=dic.values()
    

#     ndata.to_csv("new_heart_data")

def plot(data,target):
    color= []
    for index,item in enumerate(target['target']):
        if item == 1:
            color.append('r')
        else:
            color.append('b')
    total_data = pd.concat([data,pd.DataFrame(color,columns=['color'])],axis = 1)
    temp = total_data
    temp = temp.drop(['sex','cp','fbs','restecg','exang','slope','ca'],axis=1)
    cols = temp.columns
    for index,_ in enumerate(cols):
        try:
            Sct(temp[cols[index]],temp[cols[index+1]],temp['color'])
        except:
            pass
def Sct(lx,ly,c):
    plt.scatter(x = lx,y = ly,c = c)
    plt.xlabel(lx.name)
    plt.ylabel(ly.name)
    plt.show()

def Classifier(clf,feature , label):
    clf.fit(feature,label)
    


def main():
    feature, label =ReadData()
    plot(feature, label)
    # feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.33, random_state=42)
    # clf = DecisionTreeClassifier()
    # Classifier(clf,feature_train,label_train)
    # final = clf.predict(feature_test)
    # print(len(final),len(label_test))
    # label_test = pd.Series(label_test['target'])
    # print(label_test.shape)
    # print('正確率{}'.format(np.mean(label_test==final)))
if __name__ == '__main__':
    main()