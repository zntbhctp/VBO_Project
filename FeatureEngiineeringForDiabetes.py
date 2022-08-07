#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

#############################################
# BUSINESS PROBLEM
#############################################
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

#############################################
# DATASET STORY
#############################################
# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan,
# 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

#############################################
# VARIABLES
#############################################
# Pregnancies: Hamilelik sayısı
    # Tek başına hamilelik sayısının anlamlı bir öenmi bulunmamaktadır. Hamilelik öncesi ölçüm verileride elimizde olsaydı bu değişkeni daha anlamlı olacak şekilde kullanabilirdik
# Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
    # genel olarak normal kabul edilen aralığı 80-125 (değişebilir)
# Blood Pressure: Kan Basıncı (Küçük tansiyon) (mm Hg)
    # normal kabul edilen >= 90
# SkinThickness: Cilt Kalınlığı
    # Yağ oranını verebileceğinde önemli olabilir
# Insulin: 2 saatlik serum insülini (mu U/ml)

# DiabetesPedigreeFunction: Soydaki kişilere göre diyabet olma ihtimalini hesaplayan bir fonksiyon
    # genetik faktör
# BMI: Vücut kitle endeksi
    # 0 - 18,4: Zayıf
    # 18.5 - 24.9: Normal.
    # 25 ila 29.9 : Fazla Kilolu.
    # 30 ila 34.9 : Şişman. Birinci derece obez
    # 35 ila 44.9 : Şişman. İkinci derece obez
    # 45+ BMI: Aşırı Şişman. Üçüncü derece obez
# Age: Yaş (yıl)

# Outcome: Hastalığa sahip (1) ya da değil (0)

#############################################
# 1-Explorer Data Analysis
#############################################

# Importing Library and Basic Settings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Reading and Copy Dataset

df_ = pd.read_csv("/Users/zinnetbahcetepe/Desktop/VeriBilimiOkulu/pythonProject/Datasets/diabetes.csv")
df = df_.copy()

# Dataset General Overview
df.head(10)
df.tail()
df.shape
df.info()
df.isnull().sum()

# Numeric, Categorical and Cardinal Variables
# cat_cols, cat_but_car
df.columns
cat_th=10 # The limit for the number of uniqe variables we set so that it can be a categorical variable
car_th=20 # The limit for the number of uniqe variables we set so that it can be a cardinal variable

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
# categorical variables that look like numeric
num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
               df[col].dtypes != "O"]
# cardinal variables that look like categorical
cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and
               df[col].dtypes == "O"]

cat_cols = cat_cols + num_but_cat
#categorik except cardinal
cat_cols = [col for col in cat_cols if col not in cat_but_car]

# num_cols
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols = [col for col in num_cols if col not in num_but_cat]

cat_cols # ['Outcome']
cat_but_car # no categoric but cardinal variable
num_but_cat # ['Outcome']
num_cols # except ['Outcome']

#########################################
# Analysis of Numeric and Categorical Variable
#########################################
df[num_cols].head()
df[num_cols].describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
# There are observations with a min value of 0 in 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' variables.
# these may be missing observations

# Analysis of Target Variable with Numeric Variables
# Average of numeric variables to target variable
df.groupby('Outcome')[num_cols].mean()

# Count of numeric variables to target variable
df.groupby('Outcome')[num_cols].count()

#########################################
# Analysis of Outliers Parameters with IQR
#########################################
# Outliers for Target Variable with
# setting up and low limits
q1 = df["Glucose"].quantile(0.5)
q3 = df["Glucose"].quantile(0.95)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

up
low

df[(df["Glucose"] < low) | (df["Glucose"] > up)]

df[(df["Glucose"] < low) | (df["Glucose"] > up)].index


# Is There an Outlier or Not?

df[(df["Glucose"] < low) | (df["Glucose"] > up)].any(axis=None)
df[(df["Glucose"] < low)].any(axis=None)

###################
# Outlier Function
###################
# For this application, the quartile ratio was decided to be 5% and 95%.

# Step-1: setting up limit and low limit
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Step-2: Catching Outliers
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Step-3: Catching Outliers for each Variable

for col in num_cols:
    print(col, check_outlier(df, col))

# Just 'Insulin' variable has outliers

###################
# Accessing Outliers
###################
low, up = outlier_thresholds(df, "Insulin")
# conditional filtering that catches the outliers
df[(df["Insulin"] < low) | (df["Insulin"] > up)].head()

# index information
outlier_index= df[(df["Insulin"] < low) | (df["Insulin"] > up)].index
outlier_index
# there are only 2 observations we can delete them from the dataset
df.shape

# new dataset without outliers
df = df[~(df["Insulin"] < low) | (df["Insulin"] > up)]


#########################################
# Correlation of Missing Parameters
#########################################
df[num_cols].head()
corr = df[num_cols].corr()
corr

# there is a moderate correlation between 'Age' and 'Pregnancies' (corr is 0.544)

import seaborn as sns
sns.regplot(df["Glucose"], df["Insulin"])


# Heat Map
sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr, cmap='RdBu')
plt.show()


#########################################
# Local Outlier Factor
#########################################

dff = df.select_dtypes(include=['float64', 'int64'])

# LocalOutlierFactor Method
# for argument 'n_neighbors' we use default parameter value: 20
clf = LocalOutlierFactor(n_neighbors=20)
# Apply to dff:
clf.fit_predict(dff)
# Output is lof scores

dff_scores = clf.negative_outlier_factor_
dff_scores[0:5]
# Sorting of scores
np.sort(dff_scores)[0:5]

# PCA(Principal Component Analysis)
#  In order to setting threshold, we can use Elbow
scores = pd.DataFrame(np.sort(dff_scores))
scores.plot(stacked=True, xlim=[0, 100], style='.-')
plt.show()

# There is a sharp differentiation in the 10th index, The 10th index value can be set as the threshold value
th = np.sort(dff_scores)[10]

# Outliers
dff[dff_scores < th]

dff[dff_scores < th].shape
# There are only 10 observations, we can drop them

# To understand why these observations are outliers as a result of the LOF analysis,
# we can compare them using the "describe" function.
dff.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

# Analyzing
# Specially Glucose, Insulin and DiabetesPedigreeFunction variables look high, some of them 0
#  Pregnancies frequency  4 and 9
# As a result, we analyzed all the variables together.

# index of outliers
dff_lof= dff[dff_scores < th].index
dff_lof

# drop from dataset
df.head(15)
df = df.drop(index=(dff_lof), axis=1)
df.head(15) # no 13th index


#########################################
# Analysis of Missing Values
#########################################
df.head()
# is there missing parameter
df.isnull().values.any()
# number of missing parameter
df.isnull().sum()
# number of integers in variables
df.notnull().sum()

# total number of missing values in the data set
df.isnull().sum().sum()
# its looks like there are no missing values in dataset

df[num_cols].describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
# There are observations with a min value of 0 in 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' variables.
# these may be missing observations


df.columns
df[df['Glucose'] == 0].any(axis=None) # False

df[df['Insulin'] == 0].any(axis=None) # True


# Assigning the "0" value to outliers and removing them from the dataset
# For 'Insulin'
df[df['Insulin'] == 0].any(axis=None) # True
# How many?
df[df['Insulin'] == 0].shape # 368
# remove from dataframe
df = df[~(df["Insulin"] == 0)]

# For 'BloodPressure'
df[df['BloodPressure'] == 0].any(axis=None) # True
df[df['BloodPressure'] == 0].shape # 35
df.columns

# For 'BMI'#
df[df['BMI'] == 0].any(axis=None) # True
df[df['BMI'] == 0].shape # 11
# remove from dataframe
df = df[~(df["BMI"] == 0)]

missing_val_bmi = df[df['BloodPressure'] == 0].index
missing_val_ins = df[df['Insulin'] == 0].index


df[(df["Glucose"] == 0) | (df["BMI"] == 0) | (df["Insulin"] == 0)].any(axis=None) # False
################################################################################

# Since the number of missing observations is too high, we can fill in the missing observations with the KNN Method.
# or we can leave missing values because we will build a tree-based model
# We will try both methods and see the model result

####################################################
# # Feature Extraction
####################################################
df[num_cols].nunique()

# BMI: Body mass index
#     # 0 - 18.4: Weak
#     #18.5 - 24.9: Normal
#     #25 to 29.9 : Overweight
#     #30 to 34.9: Fat. first degree obese
#     #35 to 44.9: Fat. second degree obese
#     #45+ BMI: Extremely Fat. third degree obese

my_labels = ['weak', 'normal', 'owerweight', 'first degree obese', 'second degree obese', 'third degree obese']

#df["BMI_qcut"] = pd.qcut(df['BMI'], 6)

df["BMI_qcut_labels"] = pd.qcut(df['BMI'], 6, labels=my_labels)
df.head()

####################################################
# One-Hot Encoding
####################################################
# Assign ohe_cols equal to 10, less than 10 and greater than 2
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
ohe_cols


df = pd.get_dummies(df, ohe_cols, drop_first=True)
df.head()

df.head()

####################################################
# # Creating New Variables with Feature Interactions
####################################################
df.head()

df["NEW_BP_GLUCOSE"] = df["BloodPressure"] * df["Glucose"]


df.head()
df.info()
df.shape
df.value_counts()
df.nunique()

# Numeric Variable (Including New Variables)
new_num_cols = [col for col in df.columns if df[col].dtypes != "O"]

new_num_cols


#############################################
# Model
#############################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
df.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=28)

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
# success score
accuracy_score(y_pred, y_test)
# % 85,1 predict score
#  Given the variable values in the dataframe, we can predict with 85% accuracy whether a patient has diabetes or not.

#############################################
# The score we would get if we didn't do any of the above pre-processing
#############################################
# df_ we repeat the operations with the data set we backed up
df_.head()
df_.dropna(inplace=True)
y2 = df_["Outcome"]
X2 = df_.drop(["Outcome"], axis=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.30, random_state=17)
rf_model2 = RandomForestClassifier(random_state=46).fit(X_train2, y_train2)
y_pred2 = rf_model2.predict(X_test2)
accuracy_score(y_pred2, y_test2)
# Predict score %77


# Observing the effect of newly produced variables on model success

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)


# "Glucose", "NEW_BP_GLUCOSE", "Insulin" and "Age" most important variables determining diabetes disease
# As a result, we developed a model with a success score of 85%.

