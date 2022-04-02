import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import  DecisionTreeClassifier , plot_tree
import pickle

data = pd.read_csv("heart1.csv")
print(data)
features =data.drop('HeartDisease'  , axis='columns')
target = data['HeartDisease']
nfeatures = pd.get_dummies( features , drop_first = True)
print(nfeatures)
print(target)
x_train , x_test , y_train , y_test = train_test_split( nfeatures , target , random_state=123)

model = DecisionTreeClassifier(criterion = 'gini')
abc = model.fit(x_train , y_train)

y_pred = model.predict(x_test)
cr = classification_report(y_test , y_pred)
print(cr)

'''
d = [[ 37 , 130 , 283 , 0 , 98 , 0, 1 , 0  ,1 , 0]]
res = model.predict(d)
print(res)
'''
with open("heart.model" , "wb") as f :
	pickle.dump(model , f)

