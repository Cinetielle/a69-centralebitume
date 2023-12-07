# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
import numpy
import streamlit as st
from streamlit.hello.utils import show_code
import pandas as pd
from datetime import  datetime, timedelta
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from PIL import Image
import Functions.map_tool as map_tool
import Functions.Meteo_tool as Meteo_tool
import pytz

LegendMap = Image.open('./im/mapLegend.png')

def data_explore() -> None:
    DataGPS = pd.read_csv('./DATA/BATIMENTS/BatimentsInteret.csv', sep=';')

    DataGPS = DataGPS.astype({"Lat":"float"})
    DataGPS = DataGPS.astype({"Longi":"float"})
    CoordCentraleLat = DataGPS.Lat[0]
    CoordCentraleLon = DataGPS.Longi[0]
    
    Adresse_lettre = st.text_input('',value="", type="default", placeholder="Votre Adresse", disabled=False, label_visibility="visible")
    if Adresse_lettre=='':
        st.write("En attente de la saisie d'une adresse")
    else: 
        app = Nominatim(user_agent="tutorial")
        location = app.geocode(Adresse_lettre).raw
        
        Distance = map_tool.DistanceAB_Earth(CoordCentraleLat,location['lat'],CoordCentraleLon,location['lon'])
        Distance = round (Distance*1000)
        if Distance < 1000:
            st.write('Vous êtes à ', Distance , ' m de la centrale à bitume de Puylaurens')
        else:
            Distance = Distance/1000
            st.write('Vous êtes à ', Distance , ' km de la centrale à bitume de Puylaurens')
            
        delta_longi=abs(CoordCentraleLon-float(location['lon']))
        delta_lat=abs(CoordCentraleLat-float(location['lat']))
        max_delta=max(delta_longi,delta_lat)

        zoom_lvl = map_tool.ZoomLvl(max_delta)
 
        MapTiles = map_tool.SwitchMapStyle()

        m = folium.Map(location=[location['lat'],location['lon']], zoom_start=zoom_lvl, tiles="OpenStreetMap")
        m._children['openstreetmap'].tiles=MapTiles

        IconCentrale = folium.Icon(icon="house", icon_color="black", color="black", prefix="fa")
        folium.Marker([location['lat'],location['lon']], popup="Domicile", tooltip="Domicile").add_to(m)
        folium.Marker([CoordCentraleLat,CoordCentraleLon], popup="Centrale à bitume", tooltip="Centrale à bitume", icon=IconCentrale).add_to(m)
        st_data = st_folium(m, width=725)

    # set time series
    TimeVision = st.sidebar.selectbox('Quelles données voulez-vous consulter?',('Historique', 'Temps réel', 'Prévisions'))
    if TimeVision == 'Historique':
        # set time series
        print('in development')
        #meteo = pd.read_csv('./DATA/METEO/meteo_puylaurens.csv', sep=';', skiprows=3)
        #date = pd.to_datetime(meteo.iloc[:, 0], format="%d/%m/%y")
        #start_date = st.sidebar.date_input('Début de période', date[0]+datetime.timedelta(days=5))
        #end_date = st.sidebar.date_input('Fin de période', date[len(date)-1])
    elif TimeVision == 'Temps réel':
        Temperature, Humidite, IsDay, Precipitation, SolarPower, Pression, V_vents, Dir_vents, Raf_vents = Meteo_tool.MeteoDataLive()

        Temperature = round(Temperature*100)/100
        Pression    = round(Pression*100)/100
        V_vents     = round(V_vents*100)/100
        Dir_vents   = round(Dir_vents*100)/100
        Raf_vents   = round(Raf_vents*100)/100

        if IsDay==1: JourStatus ='jour'
        else: JourStatus ='nuit'

        TableParticule = pd.DataFrame(
        [
            {"Donnée": "Température",         "Valeur":Temperature,        "Unité":'°C'},
            {"Donnée": "Humidité",            "Valeur":Humidite,           "Unité":'%'},
            {"Donnée": "Jour / nuit",         "Valeur":JourStatus,         "Unité":''},
            {"Donnée": "Précipitation",       "Valeur":Precipitation,      "Unité":'mm'},
            {"Donnée": "Exposition Solaire",  "Valeur":SolarPower,         "Unité":'W/m²'},
            {"Donnée": "Pression",            "Valeur":Pression,           "Unité":'hPa'},
            {"Donnée": "Vitesse vents",       "Valeur":V_vents,            "Unité":'km/h'},
            {"Donnée": "Direction vents",     "Valeur":Dir_vents,          "Unité":'°'},
            {"Donnée": "Vitesse Rafales",     "Valeur":Raf_vents,          "Unité":'km/h'},

        ]
        )

        st.sidebar.write(TableParticule.to_html(escape=False, index=False), unsafe_allow_html=True)

    elif TimeVision == 'Prévisions':
        tz = pytz.timezone('Europe/Paris')
        now = datetime.now(tz)
        month = now.month
        if month < 10:
            month = '0'+str(month)
        day = now.day
        if day < 10:
            day = '0'+str(day)    
        today = str(now.year)+'-'+str(month)+'-'+str(day)
        today = datetime.strptime(today, '%Y-%m-%d').date()
        DayMenu = [today + timedelta(days = i) for i in range(8)]
        ChosenDay = st.sidebar.selectbox('Choisissez le jour de prévisions',DayMenu[1:len(DayMenu)])

        MeteoData = Meteo_tool.MeteoDataFuture(str(ChosenDay))
        

    st.title('Informations complémentaires')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
    with col2:
        st.image(LegendMap, caption="Légende des différentes routes affichées",width=350)
    with col3:
        st.write("")
        

        
st.set_page_config(page_title="Ma situation", page_icon="")

st.markdown("# Impacts de la centrale à bitume sur une adresse précise")
st.sidebar.header("Paramètres")
st.markdown(
    """
    Cette page permet de visualiser les impacts de la centrale à bitume de Puylaurens, pour une adresse spécifique.
    
    """
)
data_explore()
