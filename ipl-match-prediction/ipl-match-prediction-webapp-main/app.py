import pickle as pkl
from flask import Flask, render_template, request, url_for, redirect
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    team1 = str(request.args.get('list1'))
    team2 = str(request.args.get('list2'))
    toss_win = int(request.args.get('toss_winner'))
    choose = int(request.args.get('fb'))

    with open('teams_encoder.pkl', 'rb') as f:
        teams_encoder = pkl.load(f)

    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)

    # Transform team names using LabelEncoder
    try:
        cteam1 = teams_encoder.transform([team1])[0]
        cteam2 = teams_encoder.transform([team2])[0]
    except ValueError:
        return redirect(url_for('index'))

    if cteam1 == cteam2:
        return redirect(url_for('index'))

    lst = np.array([cteam1, cteam2, choose, toss_win], dtype='int32').reshape(1, -1)

    prediction = model.predict(lst)[0]

    if prediction == 0:
        team_win = team1
    else:
        team_win = team2

    return render_template('predict.html', data=team_win)

if __name__ == "__main__":
    app.run(debug=True)
