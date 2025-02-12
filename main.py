import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.compose import TransformedTargetRegressor

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test_ids = test["id"]

#print(train.head(20).to_string())

#print(train["Working Professional or Student"].value_counts())
#print(train["Profession"].isnull())

"""
# Explore and visualize the data
print(train.info())
print(train.describe().to_string())

print(train["Gender"].value_counts())

corr_matrix = train.corr(numeric_only=True)
print(corr_matrix.to_string())

num_cols = ["Age", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction",
            "Job Satisfaction", "Work/Study Hours", "Financial Stress"]
cat_cols = ["Gender", "Working Professional or Student", "Profession", "Sleep Duration",
            "Dietary Habits", "Degree", "Have you ever had suicidal thoughts ?",
            "Family History of Mental Illness", "Depression"]

for x in num_cols:
    train[x].hist()
    plt.title(x)
    plt.show()

for x in cat_cols:
    train[x].value_counts().plot(kind="bar")
    plt.show()
"""

def clean(data):
    data = data.drop(["id", "Name", "City", "CGPA"], axis=1)

    data.rename(columns={"Working Professional or Student": "Work/Student",
                         "Sleep Duration": "Sleep",
                         "Dietary Habits": "Diet",
                         "Have you ever had suicidal thoughts ?": "Suicidal_Thoughts",
                         "Work/Study Hours": "Hours",
                         "Family History of Mental Illness": "Family_History",
                         "Financial Stress": "Financial_Stress",
                         "Academic Pressure": "Academic_Pressure",
                         "Work Pressure": "Work_Pressure",
                         "Study Satisfaction": "Study_Satisfaction",
                         "Job Satisfaction": "Job_Satisfaction"}, inplace=True)

    data.Diet.fillna("Unknown", inplace=True)
    data.Degree.fillna("Unknown", inplace=True)
    data.Profession.fillna("Student", inplace=True)
    data.Financial_Stress.fillna(data["Financial_Stress"].median(), inplace=True)



    data.Academic_Pressure.fillna(0, inplace=True)
    data.Work_Pressure.fillna(0, inplace=True)
    data["Pressure"] = data["Academic_Pressure"] + data["Work_Pressure"]

    data.Study_Satisfaction.fillna(0, inplace=True)
    data.Job_Satisfaction.fillna(0, inplace=True)
    data["Satisfaction"] = data["Study_Satisfaction"] + data["Job_Satisfaction"]

    data = data.drop(["Academic_Pressure", "Work_Pressure",
                      "Study_Satisfaction", "Job_Satisfaction"], axis=1)

    return data

train = clean(train)
test = clean(test)

#print(train.head().to_string())

#print(train.info())

#print(train["Work/Student"].value_counts())

#print(test.head(20).to_string())

#print(test["Profession"].value_counts())


# Encode    Gender, Work/Student, Profession, Sleep, Diet,
#           Degree, Suicidal_Thoughts, Family_History


le = preprocessing.LabelEncoder()

cols = ["Gender", "Work/Student", "Profession", "Sleep", "Diet",
        "Degree", "Suicidal_Thoughts", "Family_History"]

def label_data(data):
    for col in cols:
        data[col] = le.fit_transform(data[col])
        print(le.classes_)

label_data(train)
label_data(test)

print(train.head(20).to_string())
print(test.head(20).to_string())

y = train["Depression"]
X = train.drop(["Depression"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=train["Depression"], random_state=42)

#model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

#model = RandomForestClassifier(random_state=42)
#model = model.fit(X_train, y_train)

model = GradientBoostingClassifier(random_state=42)
model = model.fit(X_train, y_train)

#model = RidgeClassifier(random_state=42)
#model = model.fit(X_train, y_train)

#model = SVC()
#model = model.fit(X_train, y_train)

#model = TransformedTargetRegressor()
#model = model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
print(recall_score(y_test, predictions))

submission_preds = model.predict(test)

df = pd.DataFrame({"id": test_ids.values,
                   "Depression": submission_preds})

df.to_csv("mental_health_kaggle_submission.csv", index=False)
