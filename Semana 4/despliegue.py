import pandas as pd
import numpy as np
from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, root_mean_squared_error

#Carga de datos
train_url = 'https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2025/main/datasets/dataTrain_Spotify.csv'
data = pd.read_csv(train_url)

# ETL
data = data.dropna()

# Separacion de x & y
y = data['popularity']
cols_to_drop = ['popularity', 'track_id', 'artists', 'album_name', 'track_name', 'Unnamed: 0', 'track_genre']
X = data.drop(columns=cols_to_drop)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Seleccion de las mejores variables predictoras
selector = SelectKBest(score_func=f_regression, k=10)
X_train_top10 = selector.fit_transform(X_train, y_train)
X_test_top10 = selector.transform(X_test)
top10_features = X.columns[selector.get_support()]
print("Top 10 variables seleccionadas:")
print(top10_features)

# entrenamiento del modelo
model = LinearRegression()
model.fit(X_train_top10, y_train)

#Evaluacion del modelo
# y_pred = model.predict(X_test_top10)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))

# uso de joblib para guardar el modelo
joblib.dump(model, 'modelo_spotify.pkl')
joblib.dump(selector, 'selector_top10.pkl')

# cargar modelo y selector
modelo = joblib.load('modelo_spotify.pkl')
selector = joblib.load('selector_top10.pkl')
feature_names = X.columns[selector.get_support()]  # Obtener los nombres reales

# generacion de la api con flask
app = Flask(__name__)
api = Api(app, version='1.0', title='Spotify Popularity API', description='Predicción de popularidad de canciones')
ns = api.namespace('predict', description='Modelo de regresión de popularidad')

input_model = api.model('InputModel', {
    name: fields.Float(required=True, description=f'{name}') for name in feature_names
})

output_model = api.model('OutputModel', {
    'predicted_popularity': fields.Float
})

@ns.route('/')
class SpotifyModel(Resource):
    @ns.expect(input_model)
    @ns.marshal_with(output_model)
    def post(self):
        try:
            input_data = request.json
            features = np.array([[input_data[name] for name in feature_names]])
            pred = modelo.predict(features)[0]
            return {'predicted_popularity': round(pred, 2)}, 200
        except Exception as e:
            api.abort(400, f"Error en la predicción: {str(e)}")

# Ejecucion de la app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
