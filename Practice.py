import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier
from mlxtend.feature_selection import sequential_feature_selector
from sklearn.feature_selection import SelectKBest,chi2

date = pd.read_csv(r"C:\Users\hp\Downloads\orders.csv")
time= pd.read_csv(r"C:\Users\hp\Downloads\messages.csv")
# print(date)
# date["order_date"] = pd.to_datetime(date["order_date"])
time["date"] = pd.to_datetime(time["date"])
print(time.head(5))


# print(date.info())
# date["date_year"] = date["order_date"].dt.year
# date["date_month"] = date["order_date"].dt.month_name()
# date["date_day"] = date["order_date"].dt.day
# date["date_dayofweek"] = date["order_date"].dt.day_name()
# print(date.head(5))
# print(time.head(5))
# df = pd.read_csv(r"C:\Users\hp\Downloads\tested.csv")
# df.drop(columns=["PassengerId","Name","Ticket","Cabin"],inplace=True)
# print(df.head(5))

# x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=["Survived"]),df["Survived"],test_size=0.2,random_state=42)


# print(x_train.isnull().sum())
 
  ## IMPUTATION TRANSFORM 
# trf1 = ColumnTransformer([
#     ("impute_age",SimpleImputer(),["Age"]),
#     ("impute_fear",SimpleImputer(),["Fare"]),
# ],remainder="passthrough")

#   ## ONE-HOT-ENCODER
# trf2 = ColumnTransformer([
#     ("ohe_sex_embarked",OneHotEncoder(drop="first"),["Sex","Embarked"]),
# ],remainder="passthrough") 

#   ## SCALING
# trf3 = ColumnTransformer([
#     ("scale",MinMaxScaler(),slice(0,10))
# ])  
  
#   ## FEATURE SELECTION 
# trf4 = SelectKBest(score_func=chi2,k=8)

#   ## train the model
# trf5 = DecisionTreeClassifier()  

# trf = ColumnTransformer([
#     ("impute_age",SimpleImputer(),[2]),
#     ("impute_fear",SimpleImputer(),[5]),
#     ("ohe_sex_embarked",OneHotEncoder(),[1,6]),
#     ("scale",MinMaxScaler(),slice(0,10)),
#     ("Select_feature",SelectKBest(score_func=chi2,k=8),slice(0,10)),
#     ("model",DecisionTreeClassifier(),slice(0,10)),

# ],remainder="passthrough")

# ...existing code...

# preprocessor = ColumnTransformer([
#     ("impute_age", SimpleImputer(), ["Age"]),
#     ("impute_fare", SimpleImputer(), ["Fare"]),
#     ("ohe_sex_embarked", OneHotEncoder(), ["Sex", "Embarked"]),
#     ("scale", MinMaxScaler(), ["Pclass", "SibSp", "Parch"])
# ])

# pipe = make_pipeline(preprocessor, SelectKBest(score_func=chi2, k=8), DecisionTreeClassifier())
# print(pipe.fit(x_train, y_train))

# ...existing code...
# pipe = make_pipeline(trf)
# print(pipe.fit(x_train,y_train))
# pipe = Pipeline([
#     ('trf1',trf1),
#     ('trf2',trf2),
#     ('trf3',trf3),
#     ('trf5',trf5),
# ])
# pipe = make_pipeline(trf1,trf2,trf3,trf4,trf5)
# print(pipe.fit(x_train,y_train))



# print(x_test.isnull().sum())

# df = pd.read_csv(r"C:\Users\hp\Downloads\covid_toy.csv")
# # print(df.isnull().sum())

# x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=["has_covid"]),df["has_covid"],test_size=0.2)
# # print(x_train)

# transformer = ColumnTransformer(transformers=[
#     ("tnf1",SimpleImputer(),['fever']),
#     ("tnf2",OrdinalEncoder(categories=[['Mild','Strong']]),["cough"]),
#     ("tnf3",OneHotEncoder(drop="first",),["gender","city"]),
# ],remainder="passthrough")

# x_train_trans=pd.DataFrame( transformer.fit_transform(x_train),)   
# print(x_train_trans)       
# # print(transformer.transform(x_test))


# df = pd.read_csv(r"C:\Users\hp\Downloads\customer.csv")
# # print(df["review"].value_counts())
# # print(df["education"].value_counts())
# # print(df["gender"].value_counts())

# print(df.head(3))
# x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=["purchased"]),df["purchased"],test_size=0.2)

# transform = ColumnTransformer([
#     ("tfm1",OrdinalEncoder(categories=[["Poor","Average","Good"],["School","UG","PG"]]),["review","education"]),
#     ("tfm2",OneHotEncoder(drop="first"),["gender"]),
# ],remainder="drop")

# print(transform.fit_transform(x_train))
# # x_train_trans = pd.DataFrame(transform.fit_transform(x_train),columns=["review","education","gender","age"])
# # print(x_train_trans)

