import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

bitcoin = pd.read_csv('./bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

bitcoin['Dates'] = pd.to_datetime(bitcoin['Timestamp'], unit='s')

bitcoin.dropna(inplace=True)

required_features = ['Open', 'High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
output_label = 'Close'

x_train, x_test, y_train, y_test = train_test_split(
    bitcoin[required_features],
    bitcoin[output_label],
    test_size=0.3
)
# Création du modèle de régression linéaire ainsi que celui d'entrainement
model = LinearRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(f"Score du modèle : {score}")

# Prédiction du cours à J+7
future_set = bitcoin.shift(periods=7).tail(7)
prediction = model.predict(future_set[required_features])

# Tracer la courbe d'historique ainsi que la courbe de prédiction
plt.figure(figsize=(12, 7))
plt.plot(bitcoin["Timestamp"][-400:-60], bitcoin["Weighted_Price"][-400:-60], color='goldenrod', lw=2)
plt.plot(future_set["Timestamp"], prediction, color='blue', lw=2)
plt.title("Bitcoin Price over time", size=25)
plt.xlabel("time", size=20)
plt.ylabel("price in $", size=20)
plt.show()