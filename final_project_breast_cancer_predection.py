# import necessary librarys
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

# insert the raw data
breast_cancer_data = pd.read_csv(r"/content/cancer.csv")

"""# Data preprocessing"""

breast_cancer_data.head()

#number of row and columns
breast_cancer_data.shape

col_name=breast_cancer_data.columns


y=breast_cancer_data.diagnosis
drop_cols=["id","diagnosis",'Unnamed: 32']
x=breast_cancer_data.drop(drop_cols,axis =1)
x.head(10)

ax=sns.countplot(y,label='count',palette='Blues')
B,M=y.value_counts()
print(B,M)

x.describe()

"""DATA VISUALIZATION"""

sns.set(style="whitegrid")
breast_cancer_data=x
data_std=(breast_cancer_data-breast_cancer_data.mean())/breast_cancer_data.std()
data_1=pd.concat([y,data_std.iloc[ : ,0:10]],axis = 1)
data=pd.melt(data_1,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data,palette='Blues')
plt.xticks(rotation=45)
plt.show()

sns.set(style="whitegrid")
breast_cancer_data=x
data_std=(breast_cancer_data-breast_cancer_data.mean())/breast_cancer_data.std()
data_1=pd.concat([y,data_std.iloc[ : ,10:20]],axis = 1)
data=pd.melt(data_1,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data,palette='Purples')
plt.xticks(rotation=45)
plt.show()

sns.set(style="whitegrid")
breast_cancer_data=x
data_std=(breast_cancer_data-breast_cancer_data.mean())/breast_cancer_data.std()
data_1=pd.concat([y,data_std.iloc[ : ,20:30]],axis = 1)
data=pd.melt(data_1,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data,palette='autumn')
plt.xticks(rotation=45)
plt.show()

fig,axes=plt.subplots(figsize=(20,20))
sns.heatmap(x.corr(),annot=True,linewidth=1.0,fmt=".1f",ax=axes)
plt.show()

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature
def correlation (dataset, threshold):
  col_corr = set() #Set of all the names of correlated columns
  corr_matrix = dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff val
        colname = corr_matrix.columns[i] # getting the name of column
        col_corr.add(colname)
  return col_corr

corr_feature=correlation(x,.9)

print(len(corr_feature))

(corr_feature)

drop_list = (corr_feature)
x1 = x.drop(drop_list,axis= 1 )
x1.head()

#Split the data set into independent (X) and dependent (Y) data sets
X = x.iloc[:,2:31].values
Y = x.iloc[:,1].values

#Split the data set into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25,random_state = 0)

# Scale the data (Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform (X_test)

def models (X_train, Y_train):
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression (random_state=0)
    log.fit(X_train, Y_train)
    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
    tree.fit(X_train, Y_train)
    #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier (n_estimators = 10, criterion = 'entropy', random_state =0)
    forest.fit(X_train, Y_train)

    #Print the models accuracy on the training data
    print('[0] Logistic Regression Training Accuracy:', log.score (X_train, Y_train))
    print('[1] Decision Tree Classifier Training Accuracy:', tree.score (X_train, Y_train))
    print('[2] Random Forest Classifier Training Accuracy:', forest.score (X_train, Y_train))
    return log, tree, forest

model=models(x_train,y_train)

#Show another way to get metrics of the models
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print( classification_report (Y_test, model[0].predict(X_test)))
print( accuracy_score (Y_test, model[0].predict(X_test)))

model_test=models(x_test,y_test)