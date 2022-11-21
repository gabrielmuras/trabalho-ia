import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

heart = pd.read_csv('./data/base_v12.csv')

heart.drop('Unnamed: 0', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(heart.drop('HeartDisease',axis=1), heart['HeartDisease'], test_size=0.40, random_state=101)


logmodel = LogisticRegression(solver='lbfgs',max_iter=1000)

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

conf_mat = confusion_matrix(y_test, predictions)

# print(heart.columns)
BMI=22
PhysicalHealth=30
MentalHealth=30
AgeCategory=65
SleepTime=15
Male=1
Asthma=1
AlcoholDrinking=1
Stroke=1
DiffWalking=1
BorderlineDiabetes=0
Diabetes=1
DiabetsDuringPregnancy=0
PhysicalActivity=0
KidneyDisease=1
SkinCancer=1
Smoking=1
ExcellentHealth=0
FairHealth=0
GoodHealth=0
PoorHealth=1
VeryGoodHealth=0
AmericanIndianAlaskanNative=0
Asian=0
Black=0
Hispanic=0
OtherRace=0
White=1

arrayTeste = [BMI, PhysicalHealth, MentalHealth, AgeCategory, SleepTime,
       Male, Asthma, AlcoholDrinking, Stroke,
       DiffWalking, BorderlineDiabetes, Diabetes,
       DiabetsDuringPregnancy, PhysicalActivity, KidneyDisease,
       SkinCancer, Smoking, ExcellentHealth, FairHealth,
       GoodHealth, PoorHealth, VeryGoodHealth,
       AmericanIndianAlaskanNative, Asian, Black, Hispanic,
       OtherRace, White]
print(arrayTeste)

EXEMPLO = np.array(arrayTeste).reshape((1,-1))

# ([22,20,0,24,7,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1])

print("EXEMPLO: {}".format(logmodel.predict(EXEMPLO)[0]))