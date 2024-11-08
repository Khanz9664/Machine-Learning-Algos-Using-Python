from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

''here we import the iris dataset using datasets avaliable in sklearn library and train the decision tree model on that dataset ''

iris = load_iris()
X = iris.data
Y = iris.target

''Here we are splitting the datasets using train_test_split 70% of data is used for training process and 50% of data is used for testing the model ''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

classfier = DecisionTreeClassifier()

classfier.fit(X_train, Y_train)

Y_predict = classfier.predict(X_test)

accuracy = accuracy_score(Y_test, Y_predict)*100
print("Accuracy : " , accuracy , "%")

new_predict1 = classfier.predict([[5.1, 3.0, 1.3, 0.4]])
print("predicted species: ", new_predict1)
new_predict2 = classfier.predict([[6.1, 3.0, 4.3, 1.2]])
print("predicted class: ", new_predict2)
