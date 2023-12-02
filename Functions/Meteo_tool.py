# map_tools.py>
# function
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from bs4 import BeautifulSoup
import requests
import re
from datetime import datetime
import pytz

def MeteoDataLive():

    tz = pytz.timezone('Europe/Paris')
    now = datetime.now(tz)
    month = now.month
    if month < 10:
        month = '0'+str(month)

    day = now.day
    if day < 10:
        day = '0'+str(day)    
    todayDate = str(now.year)+'-'+str(month)+'-'+str(day)
    HourNow = now.hour
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 43.5722,
        "longitude": 2.0111,
        "current": ["temperature_2m", "relative_humidity_2m", "is_day", "precipitation", "cloud_cover", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "timezone": "auto",
        "start_date": todayDate,
        "end_date": todayDate
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Current values. The order of variables needs to be the same as requested.
    current = response.Current()
    
    Temperature = current.Variables(0).Value()
    Humidite = current.Variables(1).Value()
    IsDay = current.Variables(2).Value()
    Precipitation = current.Variables(3).Value()
    Pression = current.Variables(5).Value()
    V_vents = current.Variables(6).Value()
    Dir_vents = current.Variables(7).Value()
    Raf_vents = current.Variables(8).Value()

    url_weather = "https://fr.tutiempo.net/radiation-solaire/puylaurens.html"
    response = requests.get(url_weather)
    HTML_content = BeautifulSoup(response.content, 'html.parser')
    Content = HTML_content.text
    start_point=Content.find("Aujourd'hui")
    Content = Content[start_point:len(Content)]
    end_point = [i.start() for i in re.finditer("Totale", Content)]
    Content= Content[0:end_point[1]]

    start_point = [i.start() for i in re.finditer(r"\d\d:\d\d", Content)]
    end_point = [i.start() for i in re.finditer(" w/m2", Content)]
    for i in range(len(start_point)):
        if float(Content[start_point[i]:start_point[i]+2])==HourNow:
            SolarPower = float(Content[start_point[i]+5:end_point[i]])
            break
        else : 
            SolarPower = 0
    return Temperature,Humidite,IsDay,Precipitation,SolarPower,Pression,V_vents,Dir_vents,Raf_vents

def MeteoDataFuture(ChosenDate):

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 43.5722,
        "longitude": 2.0111,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "timezone": "auto",
        "start_date": ChosenDate,
        "end_date": ChosenDate
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    Temperature = hourly.Variables(0).ValuesAsNumpy()
    Humidite = hourly.Variables(1).ValuesAsNumpy()
    Precipitation = hourly.Variables(2).ValuesAsNumpy()
    Pression = hourly.Variables(3).ValuesAsNumpy()
    V_vents = hourly.Variables(5).ValuesAsNumpy()
    Dir_vents = hourly.Variables(6).ValuesAsNumpy()
    Raf_vents = hourly.Variables(7).ValuesAsNumpy()

    DataMeteo = pd.DataFrame( columns=['Heure','Temperature','Humidité','Pression','Vitesse_vent', 'Direction_vent', 'Rafales_vents' ,'Precipitation','Expo_Solaire'])

    DataMeteo.Heure             = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    DataMeteo.Temperature       = Temperature
    DataMeteo.Humidité          = Humidite
    DataMeteo.Pression          = Pression
    DataMeteo.Vitesse_vent      = V_vents
    DataMeteo.Direction_vent    = Dir_vents
    DataMeteo.Rafales_vents     = Raf_vents
    DataMeteo.Precipitation     = Precipitation

    TableMois = ['Janvier','Février','Mars','Avril','Mai','Juin','Juillet','Août','Septembre','Octobre','Novembre','Décembre'] #peu robuste attentoin mois aout
    url_weather = "https://fr.tutiempo.net/radiation-solaire/puylaurens.html"
    response = requests.get(url_weather)
    HTML_content = BeautifulSoup(response.content, 'html.parser')
    Content = HTML_content.text
    
    JourNum = int(ChosenDate[8:10])
    MoisNum = int(ChosenDate[5:7])
    JourLettre = str(JourNum) + ' ' + TableMois[MoisNum-1]
    start_point = Content.find(JourLettre)
    Content = Content[start_point:len(Content)]
    end_point = [i.start() for i in re.finditer("Totale", Content)]
    Content= Content[0:end_point[1]]

    start_point = [i.start() for i in re.finditer(r"\d\d:\d\d", Content)]
    end_point = [i.start() for i in re.finditer(" w/m2", Content)]

    TableSolarPower = pd.DataFrame(columns=['Heure', 'SolarPower'])

    for i in range(len(start_point)):
        Hour = int(Content[start_point[i]:start_point[i]+2])
        SolPow = int(Content[start_point[i]+5:end_point[i]])
        NvelleLigne = {"Heure": Hour,"SolarPower":SolPow}
        TableSolarPower = pd.concat([TableSolarPower, pd.DataFrame([NvelleLigne])], ignore_index=True)

    for i in range(len(DataMeteo.Heure)):
        for j in range(len(start_point)):
            if DataMeteo.Heure[i]==TableSolarPower.Heure[j]:
                DataMeteo.Expo_Solaire[i] = TableSolarPower.SolarPower[j]
                break
            else:
                DataMeteo.Expo_Solaire[i] = 0
    return DataMeteo

