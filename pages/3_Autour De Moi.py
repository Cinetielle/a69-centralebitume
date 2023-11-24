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
import numpy as math
import streamlit as st
from streamlit.hello.utils import show_code
import pandas as pd
import datetime
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

def data_explore() -> None:
    CoordCentraleLat=43.556417403968226;    CoordCentraleLon= 2.050626971845913

    Adresse_lettre = st.text_input('',value="", type="default", placeholder="Votre Adresse", disabled=False, label_visibility="visible")
    if Adresse_lettre=='':
        st.write("En attente de la saisie d'une adresse")
    else: 
        app = Nominatim(user_agent="tutorial")
        location = app.geocode(Adresse_lettre).raw

        Distance = math.arccos(math.sin(math.radians(CoordCentraleLat))*math.sin(math.radians(float(location['lat'])))+
                               math.cos(math.radians(CoordCentraleLat))*math.cos(math.radians(float(location['lat'])))*
                               math.cos(math.radians(CoordCentraleLon-float(location['lon']))))*6371
        
        Distance = round (Distance*1000)
        if Distance < 1000:
            st.write('Vous êtes à ', Distance , ' m de la centrale à bitume de Puylaurens')
        else:
            Distance = Distance/1000
            st.write('Vous êtes à ', Distance , ' km de la centrale à bitume de Puylaurens')
            
        delta_longi=abs(CoordCentraleLon-float(location['lon']))
        delta_lat=abs(CoordCentraleLat-float(location['lat']))
        max_delta=max(delta_longi,delta_lat)

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
 
        m = folium.Map(location=[location['lat'],location['lon']], zoom_start=zoom_lvl)

        IconCentrale = folium.Icon(icon="house", icon_color="black", color="black", prefix="fa")
        folium.Marker([location['lat'],location['lon']], popup="Domicile", tooltip="Domicile").add_to(m)
        folium.Marker([CoordCentraleLat,CoordCentraleLon], popup="Centrale à bitume", tooltip="Centrale à bitume", icon=IconCentrale).add_to(m)
        st_data = st_folium(m, width=725)

    # set time series
    TimeVision = st.sidebar.selectbox('Quelles données voulez-vous consulter?',('Historique', 'Temps réel', 'Prévisions'))
    if TimeVision == 'Historique':
        # set time series
        meteo = pd.read_csv('./DATA/METEO/meteo_puylaurens.csv', sep=';', skiprows=3)
        date = pd.to_datetime(meteo.iloc[:, 0], format="%d/%m/%y")
        start_date = st.sidebar.date_input('Début de période', date[0]+datetime.timedelta(days=5))
        end_date = st.sidebar.date_input('Fin de période', date[len(date)-1])
    elif TimeVision == 'Temps réel':
        st.write(TimeVision)
    elif TimeVision == 'Prévisions':
        st.write(TimeVision)
        
st.set_page_config(page_title="Ma situation", page_icon="")
st.markdown("# Impacts de la centrale à bitume sur une adresse précise")
st.sidebar.header("Paramètres")
st.markdown(
    """
    Cette page permet de visualiser les impacts de la centrale à bitume de Puylaurens, pour une adresse spécifique.
    
    """
)
data_explore()
