from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load data
dataset = loadtxt(r'E:\python\tensor\keras\pima-indians-diabetes.csv',delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#fit the model 
model=XGBClassifier()
model.fit(X_train,y_train)
print (model)

#Make Predictions

y_pred = model.predict(X_test)
predictions=[round(value) for value in y_pred]
#print(y_pred)
#Evaluate predictions
#accuracy=accuracy_score(y_test,y_pred)
accuracy=accuracy_score(y_test,predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))