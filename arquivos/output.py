import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

base = pd.read_csv('./data/base_v12.csv')
base.drop('Unnamed: 0', axis=1, inplace=True)

print("Descrição da tabela => ", base.describe()) 

print("Colunas => ", base.columns)

fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['HeartDisease'].value_counts().index
y=base['HeartDisease'].value_counts().values.tolist()
data = base.groupby("HeartDisease").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('HeartDisease', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['HeartDisease'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('HeartDisease',weight = 'bold')
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['Smoking'].value_counts().index
y=base['Smoking'].value_counts().values.tolist()
data = base.groupby("Smoking").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('Smoking', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['Smoking'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('Smoking',weight = 'bold')
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['AlcoholDrinking'].value_counts().index
y=base['AlcoholDrinking'].value_counts().values.tolist()
data = base.groupby("AlcoholDrinking").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('AlcoholDrinking', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['AlcoholDrinking'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('AlcoholDrinking',weight = 'bold')
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['Stroke'].value_counts().index
y=base['Stroke'].value_counts().values.tolist()
data = base.groupby("Stroke").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('Stroke', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['Stroke'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('Stroke',weight = 'bold')
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['DiffWalking'].value_counts().index
y=base['DiffWalking'].value_counts().values.tolist()
data = base.groupby("DiffWalking").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('DiffWalking', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['DiffWalking'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('DiffWalking',weight = 'bold')
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['Male'].value_counts().index
y=base['Male'].value_counts().values.tolist()
data = base.groupby("Male").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('Male', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['Male'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('Male',weight = 'bold')
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['AgeCategory'].value_counts().index
y=base['AgeCategory'].value_counts().values.tolist()
data = base.groupby("AgeCategory").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('AgeCategory', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['AgeCategory'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('AgeCategory',weight = 'bold')
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['Diabetes'].value_counts().index
y=base['Diabetes'].value_counts().values.tolist()
data = base.groupby("Diabetes").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('Diabetes', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['Diabetes'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('Diabetes',weight = 'bold')
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['PhysicalActivity'].value_counts().index
y=base['PhysicalActivity'].value_counts().values.tolist()
data = base.groupby("PhysicalActivity").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('PhysicalActivity', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['PhysicalActivity'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('PhysicalActivity',weight = 'bold')
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['Asthma'].value_counts().index
y=base['Asthma'].value_counts().values.tolist()
data = base.groupby("Asthma").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('Asthma', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['Asthma'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('Asthma',weight = 'bold')
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['KidneyDisease'].value_counts().index
y=base['KidneyDisease'].value_counts().values.tolist()
data = base.groupby("KidneyDisease").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('KidneyDisease', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['KidneyDisease'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('KidneyDisease',weight = 'bold')
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(15,8))
x=base['SkinCancer'].value_counts().index
y=base['SkinCancer'].value_counts().values.tolist()
data = base.groupby("SkinCancer").size()
sns.set(style="dark", color_codes=True)
pal = sns.color_palette("magma", len(data))
rank = data.argsort().argsort() 
sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
for p in ax[0].patches:
        ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
ax[0].set_xlabel('SkinCancer', weight='semibold', fontname = 'monospace')
_, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
        explode=[0.03 for i in base['SkinCancer'].value_counts().index])
for autotext in autotexts:
    autotext.set_color('white')
plt.legend(bbox_to_anchor=(1, 1))
plt.suptitle ('SkinCancer',weight = 'bold')
plt.show()


# fig, ax = plt.subplots(1, 2, figsize=(15,8))
# x=base['Black'].value_counts().index
# y=base['Black'].value_counts().values.tolist()
# data = base.groupby("Black").size()
# sns.set(style="dark", color_codes=True)
# pal = sns.color_palette("magma", len(data))
# rank = data.argsort().argsort() 
# sns.barplot(x=x,y=y,palette=np.array(pal[::-1])[rank],ax = ax[0])
# for p in ax[0].patches:
#         ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
#                     ha='center', va='bottom',
#                     color= 'black')
# ax[0].set_xlabel('Black', weight='semibold', fontname = 'monospace')
# _, _, autotexts= ax[1].pie(y, labels = x, colors = pal, autopct='%1.1f%%',
#         explode=[0.03 for i in base['Black'].value_counts().index])
# for autotext in autotexts:
#     autotext.set_color('Black')
# plt.legend(bbox_to_anchor=(1, 1))
# plt.suptitle ('Black',weight = 'bold')
# plt.show()