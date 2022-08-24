###################################################
# PROJECT: SCOUTIUM TALENTED CLASSIFICATION MODEL
###################################################

##############################
#İş Problemi
##############################
# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre,
# oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.

##############################
# Veri Seti Hikayesi
##############################
# scoutium_attributes.csv
# task_response_id :Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id : İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id : İlgili oyuncunun id'si
# position_id: İlgili oyuncunun o maçta oynadığı pozisyonun id’si
                # 1: Kaleci
                # 2: Stoper
                # 3: Sağ bek
                # 4: Sol bek
                # 5: Defansif orta saha 6: Merkez orta saha 7: Sağ kanat
                # 8: Sol kanat
                # 9: Ofansif orta saha 10: Forvet
# analysis_id: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id:  Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value : Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)

# scoutium_potential_labels.csv
# task_response_id : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id : İlgili maçın id'si
# evaluator_id : Değerlendiricinin(scout'un) id'si
# player_id : İlgili oyuncunun id'si
# potential_label : Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)


#############################################
# LIBRARIES, FUNCTIONS AND SETTINGS
#############################################

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import get_scorer_names

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# pip install catboost
# pip install lightgbm
# conda install lightgbm
# pip install xgboost
# pip install sklearn


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


#############################################
# DATA PRE-PROCESSING (VERİ ÖN İŞLEME)
#############################################
scoutium_att = pd.read_csv("./Datasets/scoutium_attributes.csv", sep=';')
scoutium_att.columns
scoutium_att.head()
# ['task_response_id', 'match_id', 'evaluator_id', 'player_id', 'position_id', 'analysis_id', 'attribute_id', 'attribute_value']
scoutium_pot =pd.read_csv("Datasets/scoutium_potential_labels.csv", sep=';')
scoutium_pot.columns
scoutium_pot.head()
# ['task_response_id', 'match_id', 'evaluator_id', 'player_id', 'potential_label']

# Merging datasets
df = pd.merge(scoutium_att, scoutium_pot, on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'])
df.head()
df_ = df.copy()

df['position_id'].value_counts() # 1 -700

# position_id = 1 Kaleci'ye ait veri olduğundan, veri setinden çıkartılmıştır
df = df[~(df['position_id'] == 1)]

df['position_id'].value_counts() # no class 1
df.shape # (10030, 9)


# potential_label
df['potential_label'].value_counts()
# average          7922
# highlighted      1972
# below_average     136 # tüm verinin % 1

df = df[~(df['potential_label'] == 'below_average')]
df.shape # (9894, 9)


# Indekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturma

df.pivot_table(index=["player_id", "position_id", "potential_label"], columns="attribute_id", values="attribute_value")

# reset_index
df = df.reset_index()
df.head(10)

# “attribute_id” sütunlarının isimlerini stringe çevirme
df["attribute_id"] = df["attribute_id"].apply(str)
df.info()

#############################################
# EXPLORER DATA ANALYSIS (KEŞİFÇİ VERİ ANALİZİ)
#############################################
# ENCODING
# Label Encoding
df.nunique()
# label encoderdan geçireceğimiz 2 eşsiz değeri olan kolonları, aşağıdaki list comprehension ile filtreleyebiliriz
#label_cols = [[col for col in df.columns if (df[col].dtypes == "O") &  (df[col].nunique() == 2)]]
#df = le(df, label_cols) # sadece bir değişken olduğundan aşağıdaki gibi ilerledik

le = LabelEncoder()
le.fit_transform(df['potential_label'])[0:5]
df['potential_label'] = le.fit_transform(df['potential_label'])
# eğer hangisi 1 hangisi 0 bilmiyorsak inverse_transform kullanarak bu değerlerin karşılıklarını öğrenebiliriz
#le.inverse_transform([0, 1])
# 'average' = 0, 'highlighted' = 1

df.head()
df.info()


#############################################
# FEATURE SCALING (Özellik Ölçeklendirme)
#############################################
# Sayısal Değişken Listesi
num_cols = [[col for col in df.columns if (df[col].dtypes != "O")]]


num_cols = df.columns[~(df.columns.str.contains("index"))] & df.columns[~(df.columns.str.contains("potential_label"))]
num_cols
###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

for col in df[num_cols]:
    ss = StandardScaler()
    df[col] = ss.fit_transform(df[[col]])

df.head()


######################################################
# MODEL & PREDICTION
######################################################
# LR, KNN, SVC, CART, RF, Adaboost, GBM, XGBoost, LightGBM, CatBoost


y = df["potential_label"]
X = df.drop(["potential_label", "index"], axis=1)

# Tüm sınıflandırma modellerinden başarı parametrelerini alabileceğimiz fonksiyonun tanımlanması
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y, scoring="accuracy")
# accuracy: Doğru sınıflandırma oranı
# Base Models....
# accuracy: 0.6194 (LR)
# accuracy: 0.5863 (Adaboost)
# accuracy: 0.564 (GBM)
# accuracy: 0.5399 (LightGBM)
# accuracy: 0.5273 (RF)
# accuracy: 0.5235 (KNN)
# accuracy: 0.5215 (CatBoost)
# accuracy: 0.5147 (SVC)
# accuracy: 0.5024 (XGBoost)
# accuracy: 0.4948 (CART)


# En yüksek  accuracy değeerini veren %62 ile Logistic Regression, hiper paramatre optimizasyonu ile başarı skorunu arttırılmaya çalışılacaktır
# Ancak öncesinde Roc_auc, f1, precision, recall skorlarına da bakılacaktır.

base_models(X, y, scoring="roc_auc")
# Base Models....
# roc_auc: 0.6582 (LR)
# roc_auc: 0.5684 (Adaboost)
# roc_auc: 0.5222 (GBM)
# roc_auc: 0.5105 (SVC)
# roc_auc: 0.4968 (LightGBM)
# roc_auc: 0.4903 (CART)
# roc_auc: 0.4583 (RF)
# roc_auc: 0.4568 (KNN)
# roc_auc: 0.4566 (XGBoost)
# roc_auc: 0.3905 (CatBoost)

base_models(X, y, scoring="precision")
# precision: Pozitif sınıf tahminleme oranı
# Base Models....
# precision: 0.4781 (Adaboost)
# precision: 0.4315 (LightGBM)
# precision: 0.4303 (RF)
# precision: 0.4208 (GBM)
# precision: 0.212 (CatBoost)
# precision: 0.1794 (CART)
# precision: 0.1769 (XGBoost)
# precision: 0.1692 (SVC)
# precision: 0.1699 (KNN)
# precision: 0.1575 (LR)


base_models(X, y, scoring="recall")
# recall: Pozitif sınıfın doğru tahmin edilme oranı
# Base Models....
# recall: 0.4311 (XGBoost)
# recall: 0.4139 (CART)
# recall: 0.3977 (Adaboost)
# recall: 0.3947 (GBM)
# recall: 0.3916 (KNN)
# recall: 0.3916 (SVC)
# recall: 0.3881 (RF)
# recall: 0.3876 (CatBoost)
# recall: 0.3622 (LightGBM)
# recall: 0.3521 (LR)

base_models(X, y, scoring="f1")
# Base Models....
# f1: 0.2205 (XGBoost)
# f1: 0.2147 (GBM)
# f1: 0.2112 (KNN)
# f1: 0.2041 (CART)
# f1: 0.1916 (CatBoost)
# f1: 0.1893 (Adaboost)
# f1: 0.1809 (SVC)
# f1: 0.1578 (RF)
# f1: 0.1559 (LightGBM)
# f1: 0.1512 (LR)

# Tüm metrikleri değerlendirdiğimizde LR, Adaboost ve XGBoost ile devam etmeye karar veriyoruz


################################################
# XGBoost Model
################################################

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
# 'learning_rate': None, 'n_estimators': 100, max_depth': None, 'colsample_bytree': None
# CV
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.5380
cv_results['test_f1'].mean()
# 0.1677
cv_results['test_roc_auc'].mean()
# 0.4475


# XGBoost modelindeki en önemli parametreler için hatayı düşürmek için farklı değrler verip gridSearchcv modeli kuruyoruz
xgboost_params = {"learning_rate": [0.2, 0.1, 0.01],
                  "max_depth": [3, 5, 8, 10],
                  "n_estimators": [50, 100, 200, 500, 1000],
                  "colsample_bytree": [0.7, 0.8, 1, 2]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# Yukarıda denediğimiz parametrelerden en iyileri ile final modeli kuruyoruz
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7108
cv_results['test_f1'].mean()
# 0.0818
cv_results['test_roc_auc'].mean()
# 0.5249


######################################################
# Logistic Regression Model
######################################################
log_model = LogisticRegression().fit(X, y)
log_model.intercept_[0]
log_model.coef_[0]

y_pred = log_model.predict(X)
y_pred[0:10]

y[0:10]

######################################################
# Model Evaluation -- Model Başarı Değerlendirme
######################################################
# Accuracy Score tablosu için fonksiyon ısı haritası şeklinde
# precision, recall, f1 score ve support değerlerini verir
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

#bu fonksiyonun parametreleri gerçek y değerleri ile tahmin edilen y değerleridir
plot_confusion_matrix(y, y_pred)
print(classification_report(y, y_pred))

#               precision    recall  f1-score   support
#            0       0.81      0.99      0.89      7922
#            1       0.58      0.07      0.13      1972
#     accuracy                           0.80      9894
#    macro avg       0.70      0.53      0.51      9894
# weighted avg       0.77      0.80      0.74      9894




# başarı değeri 1 sınıfına göre hesaplanır

# precision: 1 olarak yaptığımız tahminlerin  %58'i başarılı
# recall: 1 olanları %7 başarı ile doğru sınıflandırmışız

# hem 1 hem 0 sınıflarını birlikte değerlendirmek için macro avg ve weighted avg değerlerine bakılmalı
# macro avg: aritmetik ortalama
# weighted avg: ağırlıklı ortalamaları


# Accuracy: 0.80
# Precision: 0.58
# Recall: 0.07
# F1-score: 0.13


# ROC AUC (1 sınıfının gerçekleşme olasılıkları)
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.69

# bu aşamaya kadar modeli train setiyle eğitip başarı metriklerinide train setiyle denedik
# model doğrulama adımı ile verisetini train test olarak ayırıp final modeli kurulacaktır

######################################################
# Model Validation: Holdout
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

log_model = LogisticRegression().fit(X_train, y_train)
# test değerlerinin tahmşn edilmesi
y_pred = log_model.predict(X_test)
# test seti için 1 sınıfının gerçekleşme olasılıkları
y_prob = log_model.predict_proba(X_test)[:, 1]

# y_test veri setinin gerçek tahmin değerleri ile, x_test ile elde ettiğimiz tahmini y değerleri
print(classification_report(y_test, y_pred))
# tüm veriseti başarı skorları
# Accuracy: 0.80
# Precision: 0.58
# Recall: 0.07
# F1-score: 0.13


# test veriseti başarı skorları
# Accuracy: 0.81
# Precision: 0.82
# Recall: 0.06
# F1-score: 0.11

# Skorların çok farklı olmadığını görüyoruz ancak precision  skorunda önemli bir farklılık var
# ROC CURVE Grafiği
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)
# 0.6845

######################################################
# Model Validation: 10-Fold Cross Validation
######################################################
# holdout yöntemini k katlı yapmak; 10 katlı dersek model 10 parçaya bölünür 9 u ile model kurulup 1 ile test yapılır hepsi için denenir
y = df["potential_label"]
X = df.drop(["potential_label", "index"], axis=1)

log_model = LogisticRegression().fit(X, y)

# 5 katlı çapraz doğrulama yöntemi kuruyoruz
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
cv_results
#
# Accuracy: 0.80
# Precision: 0.58
# Recall: 0.07
# F1-score: 0.13
# Auc: 0.6845



cv_results['test_accuracy']

cv_results['test_accuracy'].mean()
# Accuracy: 0.7725

cv_results['test_precision'].mean()
# Precision: 0.2052

cv_results['test_recall'].mean()
# Recall: 0.1401

cv_results['test_f1'].mean()
# F1-score: 0.1107

cv_results['test_roc_auc'].mean()
# AUC: 0.6380

# CV Sonrası
# Accuracy: 0.77
# Precision: 0.20
# Recall: 0.14
# F1-score: 0.11
# AUC : 0.64

# Cross Validation sonrası başarı metriklerinde düşüş gözlemlenmiştir, cv 10 katlı denenilebilir ancak pc uygun olmadığından bu çalışmada yapılmamıştır

######################################################
# Prediction for A New Observation
######################################################

X.columns

# rastgele bir kullanıcı alıyoruz
random_user = X.sample(1, random_state=1)
# ve bu user için kendi modelimizle diabet olup olmadığını kontrol ediyoruz
random_user
log_model.predict(random_user)
# 1540 indexli kullanıcı için y tahmini 0 olarak gelmiştir
df.loc[[1406, 1407], :]
# gerçek "potential_label" değeri 0

# Bu örnek için başarılı bir sonuç olduğunu söyleyebiliriz