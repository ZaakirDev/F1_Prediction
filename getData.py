import fastf1
import numpy as np
import pandas as pd

OLDEST_YEAR = 2023
LATEST_YEAR = 2024

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

def getDriverPos(result_grid, driver):
    index = 0
    while index <= len(result_grid) - 1:
        if result_grid[index] == driver:
            return index + 1
        index += 1
    return -1

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

        grid = qualiSession.results["Abbreviation"].to_numpy()

        RaceSession = fastf1.get_session(year, gp, 'R')
        RaceSession.load()
        resultGrid = RaceSession.results['Abbreviation'].to_numpy()

        for i in range(len(grid)):
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

            # FORMAT DATA
            Drivers.append(driver)
            Quali_Positions.append(i + 1)
            TeamList.append(Teams[i])
            Races.append(gp)
            AirTemps.append(avgAirTemp)
            TrackTemps.append(avgTrackTemp)
            RainData.append(rainFall)
            Quali_Pace_Data.append(avg_pace)
            Years.append(year)
            Results.append(result)

            print("Data formatted successfully")

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
training_df.to_csv('training.csv', index=False, mode='a')
print(training_df)
