import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from collections import Counter

#EXECUTANTO SMOTE EM TODAS AS COLUNAS

base = pd.read_csv('base_v12.csv')
#Removendo colunas pois o smote não funciona com números e categorias
#"BMI","PhysicalHealth","MentalHealth","AgeCategory","SleepTime",
colunas = ["Male","Asthma","AlcoholDrinking","Stroke",
"HeartDisease","DiffWalking","BorderlineDiabetes","Diabetes","DiabetsDuringPregnancy","PhysicalActivity","KidneyDisease",
"SkinCancer","Smoking","ExcellentHealth","FairHealth","GoodHealth","PoorHealth","VeryGoodHealth",
"AmericanIndianAlaskanNative","Asian","Black","Hispanic","OtherRace","White"]
x = base
for i in colunas:
  exec(f'y = base.{i}')

  X_train,X_test,y_train,y_test=train_test_split(x, y, train_size=0.8, stratify = y, random_state=100)
  Scaler_X = StandardScaler()
  X_train = Scaler_X.fit_transform(X_train)
  X_test = Scaler_X.transform(X_test)

  counter = Counter(y_train)
  print('Before',counter)

  smt = SMOTE()
  X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

  counter = Counter(y_train_sm)
  print('After',counter)
  print(y_train_sm)


#EXECUTANDO SMOTE APENAS NO HeartDisease
x = base
y = base.HeartDisease


base_nao_cardiacos = base[base.HeartDisease == 0]
base_cardiacos = base[base.HeartDisease == 1]

X_train,X_test,y_train,y_test=train_test_split(x, y, train_size=0.8, stratify = y, random_state=100)
Scaler_X = StandardScaler()
X_train = Scaler_X.fit_transform(X_train)
X_test = Scaler_X.transform(X_test)

counter = Counter(y_train)
print('Before',counter)

smt = SMOTE()

X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

counter = Counter(y_train_sm)
print('After',counter)
print(y_train_sm)
