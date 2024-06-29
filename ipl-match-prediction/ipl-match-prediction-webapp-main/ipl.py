import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv('matches.csv')
new_df = df[['team1', 'team2', 'winner', 'toss_decision', 'toss_winner']]

# Drop rows with missing values
new_df.dropna(inplace=True)

# Define features and target variable
X = new_df[['team1', 'team2', 'toss_decision', 'toss_winner']]
y = new_df[['winner']]

# Encode team names
teams = LabelEncoder()
teams.fit(pd.concat([df['team1'], df['team2']]))

# Transform the team names
X.loc[:, 'team1'] = teams.transform(X['team1'])
X.loc[:, 'team2'] = teams.transform(X['team2'])
X.loc[:, 'toss_winner'] = teams.transform(X['toss_winner'])
y.loc[:, 'winner'] = teams.transform(y['winner'])

# Save the encoders
with open('teams_encoder.pkl', 'wb') as f:
    pkl.dump(teams, f)

# Encode 'toss_decision'
fb = {'field': 0, 'bat': 1}
X.loc[:, 'toss_decision'] = X['toss_decision'].map(fb)

# Convert to numpy arrays
X = np.array(X, dtype='int32')
y = np.array(y, dtype='int32').ravel()  # Flatten y to 1D array

# Balance the dataset
y_backup = y.copy()
ones, zeros = 0, 0
for i in range(len(X)):
    if y[i] == X[i][0]:
        if zeros <= 375:
            y[i] = 0
            zeros += 1
        else:
            y[i] = 1
            ones += 1
            X[i][0], X[i][1] = X[i][1], X[i][0]

    elif y[i] == X[i][1]:
        if ones <= 375:
            y[i] = 1
            ones += 1
        else:
            y[i] = 0
            zeros += 1
            X[i][0], X[i][1] = X[i][1], X[i][0]

print(np.unique(y, return_counts=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Train models
model1 = SVC().fit(X_train, y_train)
model2 = DecisionTreeClassifier().fit(X_train, y_train)
model3 = RandomForestClassifier(n_estimators=250).fit(X, y)

# Evaluate models
print(model1.score(X_test, y_test))
print(model2.score(X_test, y_test))
print(model3.score(X_test, y_test))

# Test prediction
test = np.array([2, 4, 1, 4]).reshape(1, -1)
print(model1.predict(test))
print(model2.predict(test))
print(model3.predict(test))

# Save the best model
with open('model.pkl', 'wb') as f:
    pkl.dump(model3, f)

# Load the model for testing
with open('model.pkl', 'rb') as f:
    model = pkl.load(f)

print(model.predict(test))
