import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# df = pd.read_csv("patient_final.csv", index_col=0)
# cols = df.columns
# df = df[cols[-17:]]
df = pd.read_csv("PaO2_lab.csv")
# print(df["VALUEUOM"].unique())
print(df.head())
def show_category(name):
    df[name] = df[name].astype(str)
    key = df.groupby(name).groups.keys()
    value = df[name].value_counts()   

    plt.pie(value.values, labels= key, autopct='%1.1f%%', )
    plt.title(name)
    plt.show()


def hand_show_category(name):
    # df[name] = df[name].astype(str)
    # key = df[df["lactate"].notnull()].groupby(name).groups.keys()
    # value = df[].count()
    # print(df["lactate"].notnull().sum())
    plt.pie([df["lactate"].notnull().sum(), len(df)-df["lactate"].notnull().sum()], labels= ["Not null", "null"], autopct='%1.1f%%', )
    plt.title(name)
    plt.show()


def get_mean(x):
    # print(x)
    v1, v2 = x.replace("(","").replace("]","").split(",")
    print(v1 , v2)
    return ((float(v1) + float(v2)) / 2)

def show_bar(bins, series, name):
    height = 800
    std_ls = []
    series = series.astype(float)
    for i in range(3):
        std_ls.append(series.mean()+series.std()*(i+1))
        std_ls.append(series.mean()-series.std()*(i+1))
    for i in std_ls:
        l4 = plt.vlines(i, 0, height/4, label='Mean', linestyle='solid', colors="black")
    plt.title(name)
    # plt.hist(series.values, int(bins),  density=True)
    plt.hist(series.values, int(bins))
    plt.ylim(0, height)
    plt.xlim(series.min()-0.1, series.max()+0.1)
    l1 = plt.vlines(series.mean(), 0, height, label='Mean', linestyle='--')
    l2 = plt.vlines(series.max(), 0, height, label='Max', linestyle='--', colors="r")
    l3 = plt.vlines(series.min(), 0, height, label='Min', linestyle='--', colors="b")
    plt.legend(handles=[l1, l2, l3, l4], labels=[f"Mean {series.mean():.6}", f"Max {series.max():.6}", f"Min {series.min():.6}", f"Std {series.std():.6}"],  loc='best')
    
    plt.show()



    #
    # pass

# print(df.describe()["age"])
# print(df["glucose_blood"].unique())
# print(df["age"].value_counts())

# print(df["age"].isna().sum())
# print(df["day_of_die"].notnull().sum())
# print(df["day_of_die"].describe())
# print(df.groupby("day_of_die").count())
# print()
# df["day_of_die"] = df["day_of_die"].apply(lambda x : 0 if x == -1 else x)
# # print(df.describe()["day_of_die"])

df["VALUENUM"] = np.log1p(df["VALUENUM"])
show_bar((df["VALUENUM"].max() - df["VALUENUM"].min())//0.1, df[df["VALUENUM"].notnull()]["VALUENUM"], "PaO2")

# show_category("die_in_h_long")
# hand_show_category("lactate")