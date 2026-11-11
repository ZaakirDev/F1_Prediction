import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

TEAMS = [
    ["Mercedes"],
    ["Ferrari"],
    ["Red Bull Racing"],
    ["Williams"],
    ["Force India"],
    ["Renault"],
    ["Toro Rosso"],
    ["Sauber"],
    ["McLaren"],
    ["Haas F1 Team"],
    ["Manor Racing"],
    ["Racing Point"],
    ["AlphaTauri"],
    ["Alpine"],
    ["Alfa Romeo Racing"],
    ["Aston Martin"],
    ["RB"],
    ["Kick Sauber"],
    ["Racing Bulls"],
    ["Audi"],
    ["Cadillac"]
]

RACES = [
    ["Australian Grand Prix"],
    ["Bahrain Grand Prix"],
    ["Chinese Grand Prix"],
    ["Russian Grand Prix"],
    ["Spanish Grand Prix"],
    ["Monaco Grand Prix"],
    ["Canadian Grand Prix"],
    ["European Grand Prix"],
    ["Austrian Grand Prix"],
    ["British Grand Prix"],
    ["Hungarian Grand Prix"],
    ["German Grand Prix"],
    ["Belgian Grand Prix"],
    ["Italian Grand Prix"],
    ["Singapore Grand Prix"],
    ["Malaysian Grand Prix"],
    ["Japanese Grand Prix"],
    ["United States Grand Prix"],
    ["Mexican Grand Prix"],
    ["Brazilian Grand Prix"],
    ["Abu Dhabi Grand Prix"],
    ["Azerbaijan Grand Prix"],
    ["French Grand Prix"],
    ["70th Anniversary Grand Prix"],
    ["Tuscan Grand Prix"],
    ["Eifel Grand Prix"],
    ["Portuguese Grand Prix"],
    ["Styrian Grand Prix"],
    ["Sakhir Grand Prix"],
    ["Turkish Grand Prix"],
    ["Dutch Grand Prix"],
    ["Qatar Grand Prix"],
    ["Saudi Arabian Grand Prix"],
    ["Emilia Romagna Grand Prix"],
    ["Miami Grand Prix"],
    ["São Paulo Grand Prix"],
    ["Las Vegas Grand Prix"],
    ["Mexico City Grand Prix"],
    ["Japanese Grand Prix"],
    ["Australian Grand Prix"],
    ["Spanish Grand Prix"],
    ["Barcelona-Catalunya Grand Prix"]
]

DRIVERS = [
    ["ALO"],  # Fernando Alonso
    ["ALB"],  # Alexander Albon
    ["ANT"],  # Kimi Antonelli
    ["BEA"],  # Oliver Bearman
    ["BOT"],  # Valtteri Bottas
    ["BOR"],  # Gabriel Bortoleto
    ["CAM"],  # Rafael Câmara
    ["COL"],  # Franco Colapinto
    ["DEV"],  # Nyck de Vries
    ["DOO"],  # Jack Doohan
    ["ERI"],  # Marcus Ericsson
    ["FIT"],  # Pietro Fittipaldi
    ["GAS"],  # Pierre Gasly
    ["GIO"],  # Antonio Giovinazzi
    ["GRO"],  # Romain Grosjean
    ["GUT"],  # Esteban Gutiérrez
    ["HAD"],  # Isack Hadjar
    ["HAR"],  # Brendon Hartley
    ["HAM"],  # Lewis Hamilton
    ["HAY"],  # Rio Haryanto
    ["HUL"],  # Nico Hülkenberg
    ["KUB"],  # Robert Kubica
    ["KVY"],  # Daniil Kvyat
    ["LAT"],  # Nicholas Latifi
    ["LAW"],  # Liam Lawson
    ["LEC"],  # Charles Leclerc
    ["LIN"],  # Arvid Lindblad
    ["MAG"],  # Kevin Magnussen
    ["MAS"],  # Felipe Massa
    ["MSC"],  # Mick Schumacher
    ["NAS"],  # Felipe Nasr
    ["NOR"],  # Lando Norris
    ["OCO"],  # Esteban Ocon
    ["PAL"],  # Jolyon Palmer
    ["PER"],  # Sergio Perez
    ["PIA"],  # Oscar Piastri
    ["RAI"],  # Kimi Räikkönen
    ["RIC"],  # Daniel Ricciardo
    ["ROS"],  # Nico Rosberg
    ["RUS"],  # George Russell
    ["SAI"],  # Carlos Sainz
    ["SAR"],  # Logan Sargeant
    ["SIR"],  # Sergey Sirotkin
    ["STR"],  # Lance Stroll
    ["TSU"],  # Yuki Tsunoda
    ["VAN"],  # Stoffel Vandoorne
    ["VER"],  # Max Verstappen
    ["VET"],  # Sebastian Vettel
    ["WEH"],  # Pascal Wehrlein
    ["ZHO"]   # Zhou Guanyu
]

# ENCODE TEAM, DRIVER AND RACE VALUES
encoder = OneHotEncoder(sparse_output=False)
encoded_teams = encoder.fit_transform(TEAMS)
encoded_drivers = encoder.fit_transform(DRIVERS)
encoded_races = encoder.fit_transform(RACES)

def encode(val, type):
    global DRIVERS, RACES, TEAMS, encoded_drivers, encoded_races, encoded_teams

    if type == 'D':
        index = 0
        while index <= len(DRIVERS) - 1:
            if DRIVERS[index][0] == val:
                return encoded_drivers[index]
            index += 1
    elif type == 'R':
        index = 0
        while index <= len(RACES) - 1:
            if RACES[index][0] == val:
                return encoded_races[index]
            index += 1
    elif type == 'T':
        index = 0
        while index <= len(TEAMS) - 1:
            if TEAMS[index][0] == val:
                return encoded_teams[index]
            index += 1
    
    return -1

# -------------------------------
# 1. Prepare training data
# -------------------------------
X_train = np.array()  # Input features
y_train = np.array()  # One-hot encoded labels

trainingDf = pd.read_csv('training.csv')
for row in trainingDf.itertuples():
    encodedDriver = encode(row.Driver, 'D')
    qualiPos = row.QualifyingPosition # NOTE: NEED TO PREPROCESS
    encodedTeam = encode(row.Team, 'T')
    encodedRace = encode(row.Race, 'R')
    year = row.Year # NOTE: NEED TO PREPROCESS
    airTemps = row.AirTemps # NOTE: NEED TO PREPROCESS
    trackTemps = row.TrackTemps # NOTE: NEED TO PREPROCESS
    rain = row.Rain
    qualiPace = row.QualifyingPace # NOTE: NEED TO PREPROCESS
    result = row.RaceResult


# -------------------------------
# 2. Create neural network
# -------------------------------

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),

    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),

    layers.Dense(y_train.shape[1], activation='softmax')
])


# -------------------------------
# 3. Compile model
# -------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# -------------------------------
# 4. Train model
# -------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# -------------------------------
# 5. Evaluate model
# -------------------------------

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


# -------------------------------
# 6. Make predictions
# -------------------------------

predictions = model.predict(X_test)

predicted_classes = np.argmax(predictions, axis=1)


# -------------------------------
# 7. Save model
# -------------------------------

model.save("model.keras")