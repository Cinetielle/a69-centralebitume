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

<<<<<<< HEAD:pages/1_Autour de la centrale.py
from datetime import  datetime, timedelta
=======
from typing import Any
import numpy as np
import streamlit as st
from streamlit.hello.utils import show_code
import pandas as pd
import folium
from streamlit_folium import st_folium
>>>>>>> ac53ce1 (B):pages/2_Autour de la centrale.py
from PIL import Image
from typing import Any
from streamlit_folium import st_folium
from streamlit.hello.utils import show_code

import folium
import Functions.map_tool as map_tool
import Functions.Meteo_tool as Meteo_tool
<<<<<<< HEAD:pages/1_Autour de la centrale.py
import numpy
import pandas as pd
import streamlit as st

LegendMap = Image.open('./im/mapLegend.png')

def main() -> None:
    Earth_rad = 6371
    Distance = st.sidebar.slider(r"Choisir le rayon d'impact de la centrale [km]", value=5.0, min_value=0.0, max_value=10.0, step=0.01)

    MeteoData = Meteo_tool.MeteoByTimeChoice()
=======
from datetime import  datetime, timedelta
import pytz
from pyproj import Transformer
from utils import Δh_Briggs, stability_pasquill, sigma


LegendMap = Image.open('./im/mapLegend.png')

def surelevation(meteo):
    global Vs, v, d, Ts, Ta, Pa, Qh, RSI, HR, vVent
    debut_jour = st.sidebar.date_input("Choisir le jour de début des émissions :", pd.to_datetime('2021/03/06'), key='start')
    if (meteo.index[-1]-pd.to_datetime(st.session_state.start)).days < 120:
        increment = st.sidebar.slider("Choisir la durée des émissions (en jours):", value=1, min_value=0, max_value=(meteo.index[-1]-pd.to_datetime(st.session_state.start)).days, step=1)
    else:
        increment = st.sidebar.slider("Choisir la durée des émissions (en jours):", value=120, min_value=0, max_value=(meteo.index[-1]-pd.to_datetime(st.session_state.start)).days, step=1)

    fin_jour = pd.to_datetime(st.session_state.start)+timedelta(days=increment)

    filtre = (meteo.index >= pd.to_datetime(st.session_state.start)) & (meteo.index <= fin_jour)
    meteo_slice = meteo.iloc[filtre, [5, 6, 7, 8, 10, 11, 12, 13, 14]]
    Vs = st.sidebar.slider(r"Choisir la vitesse ($m.s^{-1}$) des gaz en sortie de cheminée ", value=13.9, min_value=8., max_value=23.4, step=0.1)
    d = 1.35
    Ts = st.sidebar.slider(r"Choisir la température en sortie de cheminée", value=110, min_value=80, max_value=150, step=1)

    v = meteo_slice.iloc[:, 3].mean()/3.6 # vitesse du vent en m/s
    Pa = meteo_slice.iloc[:, 4].mean()  # pression atmosphérique en Pa
    Ta = meteo_slice.iloc[:, 0].mean() # température de l'air en °C
    RSI = meteo_slice.iloc[:, 7].mean()  # insolation solaire moyenne sur 24H
    HR = meteo_slice.iloc[:, 2].mean() # Humidité moyenne sur 24H

    #vecteur vent
    vdir = meteo_slice.iloc[:, 4].to_numpy()
    vVent = np.asarray([np.sin(vdir*np.pi/180), np.cos(vdir*np.pi/180)]).T*-1
    
    vVent = (meteo_slice.iloc[:, 3].to_numpy()[:, np.newaxis]/3.6)*vVent
    
def Historique():
    RGF93_to_WGS84 = Transformer.from_crs('2154', '4326', always_xy=True)
    WGS84_to_RGF93 = Transformer.from_crs('4326', '2154', always_xy=True)
    #coordonnée de la sortie de cheminée
    #-5 mètre pour intégrer le décaissement
    x0, y0, z0 = 623208.070, 6273468.332, 230-5+19

    alt = pd.read_csv('./DATA/TOPOGRAPHIE/BDALT.csv', header=None)
    filtre = (alt.loc[:, 0] < 640000) & (alt.loc[:, 1] > 6.255*1E6)
    alt = alt.loc[filtre, :]
    vx, vy = np.unique(alt.loc[:, 0]), np.unique(alt.loc[:, 1])
    nx, ny = len(vx), len(vy)
    Z = np.zeros((ny, nx))
    idY, idX = (alt.loc[:, 1]-vy.min())/75, (alt.loc[:, 0]-vx.min())/75
    Z[idY.to_numpy(dtype=int), idX.to_numpy(dtype=int)] = alt.loc[:, 2]
    X, Y = np.meshgrid(vx, vy)
    X_, Y_, Z_ = X-x0, Y-y0, Z-z0
    dist_XY = np.sqrt(X_**2+Y_**2)
    extent = [X[dist_XY < xmax*1E3].min(), X[dist_XY < xmax*1E3].max(), Y[dist_XY < xmax*1E3].min(), Y[dist_XY < xmax*1E3].max()]
    Z[dist_XY < xmax*1E3] = np.nan
    
    meteo = pd.read_csv('./DATA/METEO/Donnees_meteo_Puylaurens.csv', sep=';', encoding='UTF-8')
    meteo.index = pd.to_datetime(meteo.iloc[:, :5])
    
    surelevation(meteo)
    
def Tps_réel():
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
        {"Donnée": "Exposition nuageuse", "Valeur":SolarPower,         "Unité":'W/m²'},
        {"Donnée": "Pression",            "Valeur":Pression,           "Unité":'hPa'},
        {"Donnée": "Vitesse vents",       "Valeur":V_vents,            "Unité":'km/h'},
        {"Donnée": "Direction vents",     "Valeur":Dir_vents,          "Unité":'°'},
        {"Donnée": "Vitesse Rafales",     "Valeur":Raf_vents,          "Unité":'km/h'},

    ]
    )

    st.sidebar.write(TableParticule.to_html(escape=False, index=False), unsafe_allow_html=True)

def Prévisions():
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
>>>>>>> ac53ce1 (B):pages/2_Autour de la centrale.py

def Carte():
    Earth_rad = 6371 #km
    DataGPS = pd.read_csv('./DATA/BATIMENTS/BatimentsInteret.csv', sep=';')

    DataGPS = DataGPS.astype({"Lat":"float"})
    DataGPS = DataGPS.astype({"Longi":"float"})
    DataGPS = DataGPS.astype({"Effectif":"float"})
    interestingRow = DataGPS[DataGPS["Batiment"] == "Centrale à bitume"]
    CoordCentraleLat = interestingRow["Lat"]
    CoordCentraleLon = interestingRow["Longi"]

    max_delta =  xmax/ (np.pi * Earth_rad*2 / 360)
    zoom_lvl = map_tool.ZoomLvl(max_delta)

    MapTiles = map_tool.SwitchMapStyle()

    m = folium.Map(location=[CoordCentraleLat,CoordCentraleLon], zoom_start=zoom_lvl, tiles="OpenStreetMap")
    m._children['openstreetmap'].tiles=MapTiles
    folium.Circle(location=[CoordCentraleLat,CoordCentraleLon], fill_color='#000', radius=xmax*1000, weight=2, color="#000").add_to(m)

    lgd_txt = '<span style="color: {col};">{txt}</span>'

    #Creation des différents groupes
    Ecole_Groupe = folium.FeatureGroup(name= lgd_txt.format( txt='Ecole', col= '#37A7DA'))
    EHPAD_Groupe = folium.FeatureGroup(name= lgd_txt.format( txt='EHPAD / maison de retraite', col= '#C84EB0'))
    Creche_Groupe= folium.FeatureGroup(name= lgd_txt.format( txt='Crèches', col= '#EC912E'))

    Ecole_effectif = 0
    EHPAD_effectif = 0
    Creche_effectif = 0
    
    #Affichage des markers
    for i in range(len(DataGPS)):
        Type=DataGPS.Type[i]
        DistanceBat = map_tool.DistanceAB_Earth(CoordCentraleLat,DataGPS.Lat[i],CoordCentraleLon,DataGPS.Longi[i])
        
        if float(DistanceBat)<=float(xmax):    
            Message=DataGPS.Batiment[i]+" - "+DataGPS.Ville[i]
            Effectif_CurBat =float( DataGPS.Effectif[i])
            if Type==0:
                IconColor='black'
                Marker=folium.Marker([DataGPS.Lat[i],DataGPS.Longi[i]], tooltip=Message, icon=folium.Icon(color=IconColor)).add_to(m)
            elif Type==1:
                IconColor='blue'
                Ecole_effectif = Ecole_effectif + Effectif_CurBat
                Marker=folium.Marker([DataGPS.Lat[i],DataGPS.Longi[i]], tooltip=Message, icon=folium.Icon(color=IconColor))
                Ecole_Groupe.add_child(Marker)
                m.add_child(Ecole_Groupe)
            elif Type==2:
                IconColor='purple'
                EHPAD_effectif = EHPAD_effectif + Effectif_CurBat
                Marker=folium.Marker([DataGPS.Lat[i],DataGPS.Longi[i]], tooltip=Message, icon=folium.Icon(color=IconColor))
                EHPAD_Groupe.add_child(Marker)
                m.add_child(EHPAD_Groupe)
            elif Type==3:
                IconColor='orange'
                Marker=folium.Marker([DataGPS.Lat[i],DataGPS.Longi[i]], tooltip=Message, icon=folium.Icon(color=IconColor))
                Creche_Groupe.add_child(Marker)
                m.add_child(Creche_Groupe)
            else: #normalement aucun marker
                IconColor='red'

    folium.map.LayerControl('topleft', collapsed= False).add_to(m) 
    st_data = st_folium(m, width=725, height=725)

    TablePopulation = pd.DataFrame(columns=['Population concernée', 'Effectif'])
     
    if Ecole_effectif>0:#condition to rework
        NvelleLigne = {"Population concernée": "Elèves","Effectif":Ecole_effectif}
        TablePopulation = pd.concat([TablePopulation, pd.DataFrame([NvelleLigne])], ignore_index=True)

    if EHPAD_effectif>0:
        NvelleLigne = {"Population concernée": "Résidents EHPAD","Effectif":EHPAD_effectif}
        TablePopulation = pd.concat([TablePopulation, pd.DataFrame([NvelleLigne])], ignore_index=True)

    st.write(TablePopulation.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.write('')

    st.title('Informations complémentaires')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
    with col2:
        st.image(LegendMap, caption="Légende des différentes routes affichées",width=350)
    with col3:
        st.write("")
        
def data_explore():
    global xmax
    xmax = st.sidebar.slider(r"Choisir le rayon d'impact de la centrale [km]", value=5.0, min_value=0.0, max_value=20.0, step=0.01)
    TimeVision = st.sidebar.selectbox('Quelles données voulez-vous consulter?',('Historique', 'Temps réel', 'Prévisions'))
    if TimeVision == 'Historique':
        Historique()

    elif TimeVision == 'Temps réel':
        Tps_réel()

    elif TimeVision == 'Prévisions':
        Prévisions()


st.set_page_config(page_title="Autour de la centrale", page_icon="")
st.markdown("# Impacts de la centrale à bitume pour un rayon précis")
st.sidebar.header("Paramètres")
st.markdown(
    """
    Cette page permet de visualiser les impacts de la centrale à bitume de Puylaurens.
    
    """
)
main()
