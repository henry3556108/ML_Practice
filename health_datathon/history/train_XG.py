from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
import shap
from xgboost import plot_tree   
import xgboost as xgb

def draw_tree(model):
    estimator = model.estimators_[5]
    # 將樹輸出成dot檔
    dot_data = tree.export_graphviz(estimator,filled=True,feature_names=['nlr', 'lactate', 'bcd', 'GENDER', 'age', 'DISCHARGE_LOCATION', 'Neoplastic_disease', 'LIVER_DISEASE', 'CONGESTIVE_HEART_FAILURE', 'OTHER_NEUROLOGICAL', 'RENAL_FAILURE',
             'total', 'resp_rate', 'sys_bp', 'temp', 'hr', 'bun', 'sodium', 'glucose_blood', 'hematocrit', 'o2', 'glucose_pleural', 'pH', 'potassium', 'cre', 'wbc', 'platelets', 'alb', 'row_num'])
    # 讀取dot檔並繪製出來
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("image/random_classification_tre_out.png")


mae, mse, acc, mat = [], [], [], []

# load data
dataset = pd.read_csv("dataset/non_null_data.csv")

# split data into X and Y
x = dataset[['nlr', 'lactate', 'bcd', 'GENDER', 'age', 'DISCHARGE_LOCATION', 'Neoplastic_disease', 'LIVER_DISEASE', 'CONGESTIVE_HEART_FAILURE', 'OTHER_NEUROLOGICAL', 'RENAL_FAILURE',
             'total', 'resp_rate', 'sys_bp', 'temp', 'hr', 'bun', 'sodium', 'glucose_blood', 'hematocrit', 'o2', 'glucose_pleural', 'pH', 'potassium', 'cre', 'wbc', 'platelets', 'alb', 'row_num']]
#x = dataset[['age', 'DISCHARGE_LOCATION', 'Neoplastic_disease','potassium']]
y = np.ravel(dataset[['intubation']])

# normalize the data
x = preprocessing.StandardScaler().fit(x).transform(x)

# k-fold cross validation
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

n_splits = 4
kf = KFold(n_splits=n_splits, random_state=8, shuffle=True)
kf.get_n_splits(x)
for seed in range(5):
    mat = []
    print(seed)
    for k_index,(train_index, test_index) in enumerate(kf.split(x)):

        # fit model on training data
        model = XGBClassifier()
        model.fit(x[train_index], y[train_index])
        # make predictions for test data
        y_pred = model.predict_proba(x[test_index])

        # output

        threshold = (seed+1)/10
        y_np = []
        for item in y_pred[:, 1]:
            if (item >= threshold):
                y_np.append(1)
            else:
                y_np.append(0)
        # matrix = pd.DataFrame(confusion_matrix(y[test_index],model.predict(x[test_index])))
        # mat.append(matrix)
    
        # print(classification_report(y[test_index], y_np))
    
    # plot_tree(model)
    ceate_feature_map(['nlr', 'lactate', 'bcd', 'GENDER', 'age', 'DISCHARGE_LOCATION', 'Neoplastic_disease', 'LIVER_DISEASE', 'CONGESTIVE_HEART_FAILURE', 'OTHER_NEUROLOGICAL', 'RENAL_FAILURE',
             'total', 'resp_rate', 'sys_bp', 'temp', 'hr', 'bun', 'sodium', 'glucose_blood', 'hematocrit', 'o2', 'glucose_pleural', 'pH', 'potassium', 'cre', 'wbc', 'platelets', 'alb', 'row_num'])
    
    g = xgb.to_graphviz(model,fmap='xgb.fmap')
    import codecs  

    f=codecs.open(f'xgb_tree{seed}.png', mode='wb')  
    f.write(g.pipe('png'))
    f.close()  
    
    
    # draw_tree(model)
    # mat = np.stack(mat)
    # matrix = mat.mean(axis = 0)
    # sn.heatmap(matrix, annot=True)
    # plt.title(f"k-fold_threshold{threshold}")
    # plt.savefig(f"image/k-fold_threshold{threshold}_XG_mean.png")
    # plt.close()
        

# Plot normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                    #   title='Normalized confusion matrix')


'''
import shap

# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)
shap.summary_plot(shap_values, x, plot_type="bar")'''
