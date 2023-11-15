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
import math
import numpy as np
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

        Distance = math.acos(math.sin(math.radians(CoordCentraleLat))*math.sin(math.radians(float(location['lat'])))+
                             math.cos(math.radians(CoordCentraleLat))*math.cos(math.radians(float(location['lat'])))*
                             math.cos(math.radians(CoordCentraleLon-float(location['lon']))))*6371
        
        Distance = round (Distance*1000)
        if Distance < 1000:
            st.write('Vous êtes à ', Distance , ' m de la centrale à bitume')
        else:
            Distance = Distance/1000
            st.write('Vous êtes à ', Distance , ' km de la centrale à bitume')
            
        m = folium.Map(location=[location['lat'],location['lon']], zoom_start=16)

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
        
    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


st.set_page_config(page_title="Les données", page_icon="")
st.markdown("# Impacts de la centrale à bitume sur une adresse précise")
st.sidebar.header("Paramètres")
st.markdown(
    """
    Cette page permet de visualiser les impacts de la centrale à bitume de Puylaurens, pour une adresse spécifique.
    
    """
)
data_explore()
