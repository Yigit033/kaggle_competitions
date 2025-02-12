
import pandas as pd


training_data = pd.read_csv("titanic_diseaster_project/datasets/train.csv")
testing_data = pd.read_csv("titanic_diseaster_project/datasets/test.csv")


"""men = training_data.loc[training_data["Sex"] == "male"]["Survived"] 
print(sum(men))
print(men.count())

rate_men = sum(men)/len(men)
print(rate_men)



women = training_data.loc[training_data["Sex"] == "female"]["Survived"] 
rate_women = sum(women)/len(women)
print(rate_women)"""


"""print(training_data["Ticket"].head(10))"""




from sklearn.ensemble  import RandomForestClassifier 



features = ["Pclass", "Sex", "SibSp", "Parch"]



y = training_data["Survived"]

X = pd.get_dummies(training_data[features])
print(X.head())

X_test = pd.get_dummies(testing_data[features])

print(X_test.head())


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X,y) # training the model 

predictions = model.predict(X_test)


"""print(predictions)
"""
output= pd.DataFrame({"PassengerId":testing_data["PassengerId"], "Survived": predictions})


"""print(output)
"""



"""output.to_csv("titanic_diseaster/datasets/my_submission.csv", index=False)

print("Your submission was successfully saved!")"""

y_test = testing_data["Survived"]

accuracy = model.accuracy_score(y_test, predictions)


print(accuracy)











