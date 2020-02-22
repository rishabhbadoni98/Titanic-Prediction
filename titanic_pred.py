import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

train['train']=1
test['train']=0


combined = pd.concat([train,test])


combined = combined.drop('Cabin' , axis=1)

num=combined.groupby('Embarked').count()

combined['Embarked'] = combined['Embarked'].fillna('S')

combined['Fare'] = combined['Fare'].fillna(combined['Fare'].dropna().median())

nullcom = combined.isnull().sum()




corr = combined.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)




combined['Age']=   combined["Age"].fillna(combined.groupby("Pclass")["Age"].transform("median"))

combined= pd.get_dummies(combined, columns=['Sex'])
combined = combined.drop('Sex_female' , axis=1)

combined = pd.get_dummies(combined, columns=['Embarked'])
combined = combined.drop('Embarked_S' , axis=1)


combined = combined.drop('Ticket', axis=1)


combined = pd.get_dummies(combined, columns=['Pclass'])
combined = combined.drop('Pclass_3' , axis=1)


combined = pd.get_dummies(combined, columns=['SibSp'])
combined = combined.drop('SibSp_8' , axis=1)


combined = pd.get_dummies(combined, columns=['Parch'])
combined = combined.drop('Parch_9' , axis=1)



newname=[]
for name in combined['Name']:
    newname.append(name.split(',')[1].split('.')[0])
    
combined['title'] = newname
combined = combined.drop('Name', axis=1)

combined = pd.get_dummies(combined, columns=['title'])
combined= combined.drop(combined.columns[40] , axis=1)



from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
combined[['Age','Fare']] = robust_scaler.fit_transform(combined[['Age','Fare']])




train_df =  combined[combined['train']==1]
y = train_df['Survived']
train_df = train_df.drop(['train','Survived'], axis=1)


test_df = combined[combined['train']==0]
test_df = test_df.drop(['train','Survived'], axis=1)



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_df,y,test_size=0.25, random_state=0)
 

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train , y_train)



from sklearn.model_selection import GridSearchCV

parameters = [ {'n_estimators':[700,600,800] , 'criterion': ['entropy'],'min_samples_leaf':[1]
                ,'max_depth':[25,35,45,55], 'min_samples_split':[10,13,15],'min_weight_fraction_leaf':[0],
                'warm_start':['True']} 
                ]

    
grid_search = GridSearchCV(estimator = classifier ,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs= -1)



grid_search = grid_search.fit(X_train,y_train)


best_accuracy = grid_search.best_score_


best_parameters = grid_search.best_params_





from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=600,max_depth=25 , criterion='entropy', random_state=0, min_samples_leaf=1 ,
                                    min_samples_split=15,min_weight_fraction_leaf=0,warm_start=True)
classifier.fit(X_train , y_train)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train , y=y_train, cv=10 )


accuracies.mean()

accuracies.std()





y_pred_test = classifier.predict(test_df)
s = pd.Series(y_pred_test).astype(int)


result = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived':s})

export_csv = result.to_csv (r'D:\Project\Titanic\finalresult.csv', index = None, header=True) 
