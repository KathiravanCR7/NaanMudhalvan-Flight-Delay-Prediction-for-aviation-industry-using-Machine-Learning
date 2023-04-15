# Importing the Libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScalar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Reading the Dataset
dataset = pd.read_csv("flightdata.csv")
dataset.head()

# Handling Missing Values
dataset.info()

#skip handling the missing values step.
dataset = dataset.drop('Unnamed: 25', axis=1)
dataset.isnull().sum()

dataset = dataset[["FL_NUM", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
                   "ORIGIN", "DEST", "CRS_ARR_TIME", "DEP_DEL15", "ARR_DEL15"]]
dataset.isnull().sum()

dataset[dataset.isnull().any(axis=1)].head(10)

dataset['DEP_DEL15'].mode()

# replace the missing values with 15
datset = dataset.fillna({'ARR_DEL15': 1})
datset = dataset.fillna({'DEP_DEL15': 0})
datset.iloc[177:185]

# Handling Cateogrical Values
import math

for index, row in dataset.iterrows():
    dataset.loc[index, 'CRS_ARR_TIME'] = math.floor(row['CRS_ARR_TIME'] / 100)
datset.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['DEST'] = le.fit_transform(dataset['DEST'])
dataset['ORIGIN'] = le.fit_transform(dataset['ORIGIN'])

dataset.head(5)

dataset['ORIGIN'].unique()

dataset = pd.get_dummies(dataset, columns=['ORIGIN', 'DEST'])
dataset.head()

x = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:9].values

x

from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
z = oh.fit_transform(x[:, 4:5]).toarray()
t = oh.fit_transform(x[:, 5:6]).toarray()

z

t

x = np.delete(x[4, 5], axis=1)

# Descriptive Statistical
flight_data.describe()

# Univariate Analysis
sns.distplot(flight_data.MONTH)

# Bivariate Analysis
sns.scatterplot(x='ARR_DELAY', y='ARR_DEL15', data=flight_data)

sns.catplotlib(x='ARR_DEL15', y='ARR_DELAY', kind='bar', data=flight_data)

# Multivariate Analysis
sns.heatmap(dataset.corr())

# Splitting data into dependent and independent variables
dataset = pd.get_dummies(dataset, columns=['ORIGIN', 'DEST'])
dataset.head()

x = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:9].values

# Splitting data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

train_x, test_x, train_y, test_y = train_test_split(dataset.drop(
    'ARR_DEL15', axis=1), df['ARR_DEL15'], test_size=0.2, random_state=0)

x_test.shape

x_train.shape

y_test.shape

y_train.shape

# Scaling the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# MODEL BUILDING
# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(x_train, y_train)

DecisionTreeClassifier(random_state=0)

decisiontree = classifier.predict(x_test)

decisiontree

from sklearn.metrics import accuracy_score
desacc = accuracy_score(y_test, decisiontree)

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomFoerstClassifier(n_estimators=10, criterion='entropy')

rfc.fit(x_train, y_train)

y_predict = rfc.predict(x_test)

# ANN MODEL
# Importing the keras libraries and packages
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layer import Dense

# Creating ANN Skleton view
classification = Sequential()
classification.add(Dense(30, activation='relu'))
classification.add(Dense(128, activation='relu'))
classification.add(Dense(64, activation='relu'))
classification.add(Dense(32, activation='relu'))
classification.add(Dense(1, activation='sigmoid'))

# Compiling the ANN Model
classification.compile(optimizer='adam', loss='binary_crossentrophy', metrics=['accuracy'])

# Training the model
classification.fix(x_train, y_train, batch_size=4, validation_split=0.2, epochs=100)

# Test the model
#Decision Tree
y_pred = classifier.predict(
    [[129, 99, 1, 0, 0,  1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]])
print(y_pred)
(y_pred)

# RandomForest
y_pred = rfc.predict([[129, 99, 1, 0, 0,  1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]])
print(y_pred)
(y_pred)

classification.save('flight.h5')

# Test the model
y_pred = classification.predict(x_test)

y_pred

y_pred = (y_pred > 0.5)
y_pred


def predict_exit(sample_value):
    # Convert list to numpy array
    sample_value = np.array(sample_value)
    # Reshape because sample_value contains only one record
    sample_value = sample_value.reshape(1, -1)
    # Feature Scaling
    sample_value = sc.transform(sample_value)

    return classifier.predict(sample_value)


test = classification.predict(
    [[1, 1, 121.000000, 36.0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
if test == 1:
    print('Prediction: Chance of delay')
else:
    print('Prediction: No chance of delay.')

# Performance Testing & Hyperparameter Tuning
# Compare The Model
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

dfs = []
models = [
    ('RF', RandomForestClassifier()),
    ('DecisionTree', DecisionTreeClassifier()),
    ('ANN', MLPClassifier())
]
result = []
names = []
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'roc_auc']
target_nmes = ['no delay', 'delay']
for name, model in models:
    kfold = model_selection.kfold(n_splits=5, shuffle=True, random_state=90210)
    cv_results = model_selection.cross_validate(
        model, x_train, y_train, cv=kfold, scoring=scoring)
    clf = model.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name)
    print(classification_report(y_test, y_pred, target_names=target_names))
    results.append(cv_results)
    names.append(name)
    this_df = pd.DataFrame(cv_results)
    this_df['model'] = name
    dfs.append(this_df)

final = pd.concat(dfs, ignore_index=True)
return final

# RanomForest Accuracy
print('Training Accuracy: ', accuracy_score(y_train, y_predict_train))
print('Testing Accuracy: ', accuracy_score(y_test, y_predict))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
cm

# Accuracy score of Destination tree
from sklearn.metrics import accuracy_score
desacc = accuracy_score(y_test, decisiontree)

desacc

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, decisiontree)

cm

# Calculate the Accuracy of ANN
from sklearn.metrics import accuracy_score, classification_report
score = accuracy_score(y_pred, y_test)
print('The Accuracy for ANN model is: {}%'.format(score*100))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Comparing Model Accuracy Before & After Applying Hyperparameter Tuning
# Comare the Model
dfs = []
models = [
    ('RF', RandomForestClassifier()),
    ('DecisionTree', DecisionTreeClassifier()),
    ('ANN', MLPClassifier())
]
results = []
names = []
scoring = ['accuracy', 'precision_weighted',
           'recall_weighted', 'f1_weighted', 'roc_auc']
target_names = ['no delay', 'delay']
kfold = model_selection.kFold(n_splits=5, shuffle=True, random_state=90210)
cv_results = model_selection.cross_validate(
    model, x_train, y_train, cv=fold, scoring=scoring)
clf = model.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(name)
print(classification_report(y_tet, y_pred, target_names=target_names))
results.append(cv_results)
names.append(name)
this_df = pd.DataFrame(cv_results)
this_df['model'] = name
dfs.append(this_df)
final = pd.concat(dfs, ignore_index=True)
return final

# RandomForest Accuracy
print('Training accuracy: ', accuracy_score(y_train, y_predict_train))
print('Testing accuracy: ', accuracy_score(y_test, y_predict))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
cm

# Accuracy Score of DecisionTree
desacc = accuracy_score(y_test, decisiontree)

desacc

cm = confusion_matrix(y_test, decisiontree)

cm

# Comparing Model Accuracy Before & After Applying Hyperparameter Tuning
# Giving some parameters that can be used in randomized search cv
parameter = {
    'n_estimators': [1, 20, 30, 55, 68, 74, 90, 120, 115],
    'criterion': ['gini', 'entropy'],
    'max_features': ["auto", "sqrt", "log2"],
    'max_depth': [2, 5, 8, 10], 'verbose': [1, 2, 3, 4, 6, 8, 9, 10]
}

# Performing the randomized CV
RCV = RandomizedSearchCV(
    estimator=rf, param_distributions=parameters, cv=10, n_iter=4)

RCV.fit(x_train, y_train)

bt_params

bt_score

model = RandomForestClassifier(
    verbose=10, n_estimators=120, max_features='log2', max_depth=10, criterion='entropy')
RCV.fit(x_train, y_train)

y_predict_rf = RCV.predict(x_test)

RFC = accuracy_score(y_test, y_predict_rf)
RFC

# Model Deployment
# Save the best Model
pickle.dump(RCV, open('flight.pkl', '////////wb'))


# Importing the necessary dependencies

model = pickle.load(open('flight.pkl', 'rb'))
app = Flask(_name_)  # Initializing the app


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/prediction', methods=['POST'])
# Retrieves the value from UI:
def predict():
    name = request.form['name']
    month = request.form['month']
    dayofmonth = request.form['dayofmonth']
    dayofweek = request.form['dayofweek']
    origin = request.form['origin']
    if (origin == "map"):
        origin1, origin2, origin3, origin4, origin5 = 0, 0, 0, 0, 1
    if (origin == "dtw"):
        origin1, origin2, origin3, origin4, origin5 = 1, 0, 0, 0, 0
    if (origin == "jfk"):
        origin1, origin2, origin3, origin4, origin5 = 0, 0, 1, 0, 0
    if (origin == "sea"):
        origin1, origin2, origin3, origin4, origin5 = 0, 1, 0, 0, 0
    if (origin == "alt"):
        origin1, origin2, origin3, origin4, origin5 = 0, 0, 0, 1, 0


destination = request.form['destination']
if (destination == "map"):
    destination1, destination2, destination3, destination4, destination5 = 0, 0, 0, 0, 1
if (destination == "dtw"):
    destination1, destination2, destination3, destination4, destination5 = 1, 0, 0, 0, 0
if (destination == "jfk"):
    destination1, destination2, destination3, destination4, destination5 = 0, 0, 1, 0, 0
if (destination == "sea"):
    destination1, destination2, destination3, destination4, destination5 = 0, 1, 0, 0, 0
if (destination == "alt"):
    destination1, destination2, destination3, destination4, destination5 = 0, 0, 0, 1, 0
dept = request.form['dept']
arrtime = request.form['arrtime']
actdept = request.form['actdept']
dept15 = int(dept)-int(actdept)
total = [[name, month, dayofmonth, dayofweek, origin1, origin2, origin3, origin4,
          origin5, destination1, destinatioon2, destination3, destination4, destination5]]
# print Total
y_pred = model.predict(total)

print(y_pred)

if (y_pred == [0.]):
    ans = "The Flight will be  on time"
else:
    ans = "The Flight will be delayed"
    
return render_template("index.html", showcase=ans)

if _name_ == '_main_':
    app.run(debug=True)
