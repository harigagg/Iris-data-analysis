import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set(color_codes=True)

#%matplotlib inline
#%pylab inline

df = pd.read_csv('Iris.csv')
df.head()
del df['Id']
print (df.describe())
listOfColumns = df.columns
listOfNumericalColumns = []
for column in listOfColumns:
    if df[column].dtype == 'float64':
        listOfNumericalColumns.append(column)

print('listOfNumericalColumns :',listOfNumericalColumns)
spices = df['Species'].unique()
print('spices :',spices)

fig, axs = plt.subplots(nrows=len(listOfNumericalColumns),ncols=len(spices),figsize=(15,15))

for i in range(len(listOfNumericalColumns)):
    for j in range(len(spices)):
        print(listOfNumericalColumns[i]," : ",spices[j])
        axs[i,j].boxplot(df[listOfNumericalColumns[i]][df['Species']==spices[j]])
        axs[i,j].set_title(listOfNumericalColumns[i]+""+spices[j])
df.hist(figsize=(10,5))
print("HIST PLOT OF INDIVIDUAL Species")
print(spices)

for spice in spices:
        df[df['Species']==spice].hist(figsize=(10,5))

df.boxplot(by='Species',figsize=(15,15))
sns.violinplot(data=df,x='Species',y='PetalLengthCm')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
#plt.show()
from sklearn.metrics import accuracy_score

def generateClassificationReport(y_test,y_pred):
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    print('accuracy is ',accuracy_score(y_test,y_pred))

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)