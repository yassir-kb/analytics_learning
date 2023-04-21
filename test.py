from flask import Flask, render_template, request

app = Flask(__name__)


# Définir la page d'accueil
@app.route('/')
def home():
    return render_template('hm.html')


# Définir la page des résultats
@app.route('/results', methods=['POST'])
def results():
    # Récupérer les données entrées par l'utilisateur
    test = request.form['test']
    sample_size = request.form['sample_size']
    group_size = request.form['group_size']
    outcome_variable = request.form['outcome_variable']
    significance = request.form['significance']

    # Appeler les fonctions appropriées pour effectuer les calculs et les tests statistiques
    # En fonction des données entrées par l'utilisateur

    # Renvoyer les résultats à la page des résultats
    return render_template('rs.html', test=test, sample_size=sample_size, group_size=group_size,
                           outcome_variable=outcome_variable, significance=significance)


if __name__ == '__main__':
    app.run(debug=True)
