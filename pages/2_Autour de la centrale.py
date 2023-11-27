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
import datetime
import folium
from streamlit_folium import st_folium
from PIL import Image

LegendMap = Image.open('./im/mapLegend.png')

def data_explore() -> None:
    Earth_rad = 6371
    TimeVision = st.sidebar.selectbox('Quelles données voulez-vous consulter?',('Historique', 'Temps réel', 'Prévisions'))
    if TimeVision == 'Historique':
        meteo = pd.read_csv('./DATA/METEO/meteo_puylaurens.csv', sep=';', skiprows=3)
        date = pd.to_datetime(meteo.iloc[:, 0], format="%d/%m/%y")
        start_date = st.sidebar.date_input('Début de période', date[0]+datetime.timedelta(days=5))
        end_date = st.sidebar.date_input('Fin de période', date[len(date)-1])
    elif TimeVision == 'Temps réel':
        st.write(TimeVision)
    elif TimeVision == 'Prévisions':
        st.write(TimeVision)

    DataGPS = pd.read_csv('./DATA/BATIMENTS/BatimentsInteret.csv', sep=';')

    DataGPS = DataGPS.astype({"Lat":"float"})
    DataGPS = DataGPS.astype({"Longi":"float"})
    DataGPS = DataGPS.astype({"Effectif":"float"})
    interestingRow = DataGPS[DataGPS["Batiment"] == "Centrale à bitume"]
    CoordCentraleLat = interestingRow["Lat"]
    CoordCentraleLon = interestingRow["Longi"]

    Distance = st.sidebar.slider(r"Choisir le rayon d'impact de la centrale [km]", value=5.0, min_value=0.0, max_value=10.0, step=0.01)
    max_delta =  Distance/ (numpy.pi * Earth_rad*2 / 360)

    if max_delta<0.005: zoom_lvl=16
    elif max_delta<0.011:zoom_lvl=15
    elif max_delta<0.022:zoom_lvl=14
    elif max_delta<0.044:zoom_lvl=13
    elif max_delta<0.088:zoom_lvl=12
    elif max_delta<0.176:zoom_lvl=11
    elif max_delta<0.352:zoom_lvl=10
    elif max_delta<0.703:zoom_lvl=9
    elif max_delta<1.406:zoom_lvl=8
    elif max_delta<2.813:zoom_lvl=7
    elif max_delta<5.625:zoom_lvl=6
    elif max_delta<11.25:zoom_lvl=5
    elif max_delta<22.5:zoom_lvl=4
    elif max_delta<45:zoom_lvl=3
    elif max_delta<90:zoom_lvl=2
    elif max_delta<180:zoom_lvl=1
    else: zoom_lvl=0

    ToggleSat = st.toggle('Vue carte / Vue satellite')
    if ToggleSat:
        MapTiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'"
    else:
        MapTiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png'

    
    m = folium.Map(location=[CoordCentraleLat,CoordCentraleLon], zoom_start=zoom_lvl, tiles="OpenStreetMap")
    m._children['openstreetmap'].tiles=MapTiles
    folium.Circle(location=[CoordCentraleLat,CoordCentraleLon], fill_color='#000', radius=Distance*1000, weight=2, color="#000").add_to(m)

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
        DistanceBat = numpy.arccos(numpy.sin(numpy.radians(CoordCentraleLat))*numpy.sin(numpy.radians(DataGPS.Lat[i]))+
                                   numpy.cos(numpy.radians(CoordCentraleLat))*numpy.cos(numpy.radians(DataGPS.Lat[i]))*
                                   numpy.cos(numpy.radians(CoordCentraleLon-DataGPS.Longi[i])))*Earth_rad 
        
        if float(DistanceBat)<=float(Distance):    
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

    TablePopulation = pd.DataFrame([{"Population concernée":'',"Effectif":''}])
    
    if Ecole_Groupe.show==1:#condition to rework
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

    

    

st.set_page_config(page_title="Autour de la centrale", page_icon="")
st.markdown("# Impacts de la centrale à bitume pour un rayon précis")
st.sidebar.header("Paramètres")
st.markdown(
    """
    Cette page permet de visualiser les impacts de la centrale à bitume de Puylaurens.
    
    """
)
data_explore()
