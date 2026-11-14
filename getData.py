import fastf1
import numpy as np
import pandas as pd
from time import sleep
from sklearn.preprocessing import OneHotEncoder

OLDEST_YEAR = 2018
LATEST_YEAR = 2027

Drivers = []
Quali_Positions = []
TeamList = []
Races = []
AirTemps = []
TrackTemps = []
RainData = []
Quali_Pace_Data = []
Years = []
Results = []

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

def getDriverPos(result_grid, driver):
    index = 0
    while index <= len(result_grid) - 1:
        if result_grid[index] == driver:
            return index + 1
        index += 1
    return -1

def addData():
    global Drivers, Quali_Positions, TeamList, Races, Years, AirTemps, TrackTemps, RainData, Quali_Pace_Data, Results

    trainingData = {
        "Driver": Drivers,
        "QualifyingPosition": Quali_Positions,
        "Team": TeamList,
        "Race": Races,
        "Year": Years,
        "AirTemps": AirTemps,
        "TrackTemps": TrackTemps,
        "Rain": RainData,
        "QualifyingPace": Quali_Pace_Data,
        "RaceResult": Results
    }

    training_df = pd.DataFrame(trainingData)
    # training_df.to_csv('training.csv', index=False, mode='a')
    training_df.to_csv('training.csv', index=False)
    print(training_df)

    Drivers = []
    Quali_Positions = []
    TeamList = []
    Races = []
    AirTemps = []
    TrackTemps = []
    RainData = []
    Quali_Pace_Data = []
    Years = []
    Results = []

for year in range(OLDEST_YEAR, LATEST_YEAR):

    schedule = fastf1.get_event_schedule(year)

    # Exclude pre-season testing events
    schedule = schedule[
        ~schedule["EventName"].str.contains(
            "testing", case=False, na=False
        )
    ]

    schedule = schedule["EventName"].to_numpy()
    if year == 2026:
        schedule = schedule[:19] # Shortens the list so its only uptil brazil
    
    normalizedYear = (year - 2016)/(2026 - 2016)

    for gp in schedule:
        print(f"Now processing {gp} {year}")

        qualiSession = fastf1.get_session(year, gp, 'Q')
        qualiSession.load(weather=True)

        # GET WEATHER DATA
        track = qualiSession.weather_data['TrackTemp'].to_numpy()
        avgTrackTemp = np.average(track)

        air = qualiSession.weather_data['AirTemp'].to_numpy()
        avgAirTemp = np.average(air)

        rainFall = qualiSession.weather_data['Rainfall'].to_numpy()[0]
        if rainFall == True:
            rainFall = 1
        else:
            rainFall = 0

        grid = qualiSession.results["Abbreviation"].to_numpy()

        RaceSession = fastf1.get_session(year, gp, 'R')
        RaceSession.load()
        resultGrid = RaceSession.results['Abbreviation'].to_numpy()
        num_drivers = len(qualiSession.drivers)

        for i in range(len(grid)):
            try:
                # getData(year, gp, grid[i], i)
                driver = grid[i]

                # GET TEAM NAME
                Teams = qualiSession.results['TeamName'].to_numpy()

                # GET QUALI PACE
                driverLaps = qualiSession.laps.pick_drivers(driver)
                race_pace_laps = driverLaps.pick_quicklaps().pick_wo_box()
                lap_times = race_pace_laps['LapTime'].dt.total_seconds()

                avg_pace = lap_times.mean()

                # GET RACE RESULT
                result = getDriverPos(resultGrid, driver)
                if result == -1:
                    print(f"ERROR: DRIVER POSITION COULD NOT BE FOUND. RACE: {gp}, YEAR: {year}, DRIVER: {driver}")
                else:
                    print(f"RACE RESULT: {result}")
                    result = abs((result - num_drivers)/(1 - num_drivers))

                # FORMAT DATA
                Drivers.append(encode(driver, 'D'))
                Quali_Positions.append(abs(((i + 1) - num_drivers)/(1 - num_drivers)))
                TeamList.append(encode(Teams[i], 'T'))
                Races.append(encode(gp, 'R'))
                AirTemps.append(avgAirTemp)
                TrackTemps.append(avgTrackTemp)
                RainData.append(rainFall)
                Quali_Pace_Data.append(avg_pace)
                Years.append(normalizedYear)
                Results.append(result)

                print("Data formatted successfully")
            
            except fastf1.exceptions.RateLimitExceededError:
                print("Rate Limit Exceeded. The program will sleep for an hour until APIs can be called again.")
                sleep(3600 + 300) # Extra 5 mins just to be safe

                # getData(year, gp, grid[i], i)
                driver = grid[i]

                # GET TEAM NAME
                Teams = qualiSession.results['TeamName'].to_numpy()

                # GET QUALI PACE
                driverLaps = qualiSession.laps.pick_drivers(driver)
                race_pace_laps = driverLaps.pick_quicklaps().pick_wo_box()
                lap_times = race_pace_laps['LapTime'].dt.total_seconds()

                avg_pace = lap_times.mean()

                # GET RACE RESULT
                result = getDriverPos(resultGrid, driver)
                if result == -1:
                    print(f"ERROR: DRIVER POSITION COULD NOT BE FOUND. RACE: {gp}, YEAR: {year}, DRIVER: {driver}")
                else:
                    print(f"RACE RESULT: {result}")
                    result = abs((result - num_drivers)/(1 - num_drivers))

                # FORMAT DATA
                Drivers.append(driver)
                Quali_Positions.append(abs(((i + 1) - num_drivers)/(1 - num_drivers)))
                TeamList.append(encode(Teams[i], 'T'))
                Races.append(encode(gp, 'R'))
                AirTemps.append(avgAirTemp)
                TrackTemps.append(avgTrackTemp)
                RainData.append(rainFall)
                Quali_Pace_Data.append(avg_pace)
                Years.append(year)
                Results.append(result)

                print("Data formatted successfully")

    addData()
    print("Data added successfully")
