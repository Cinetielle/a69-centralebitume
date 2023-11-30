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
from geopy.geocoders import Nominatim
from PIL import Image
from bs4 import BeautifulSoup
import requests

LegendMap = Image.open('./im/mapLegend.png')


def data_explore() -> None:
    Earth_rad = 6371
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
        
        Distance = numpy.arccos(numpy.sin(numpy.radians(CoordCentraleLat))*numpy.sin(numpy.radians(float(location['lat'])))+
                                numpy.cos(numpy.radians(CoordCentraleLat))*numpy.cos(numpy.radians(float(location['lat'])))*
                                numpy.cos(numpy.radians(CoordCentraleLon-float(location['lon']))))*Earth_rad
        
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
 
        ToggleSat = st.toggle('Vue carte / Vue satellite')
        if ToggleSat:
            MapTiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'"
        else:
            MapTiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png'

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
        meteo = pd.read_csv('./DATA/METEO/meteo_puylaurens.csv', sep=';', skiprows=3)
        date = pd.to_datetime(meteo.iloc[:, 0], format="%d/%m/%y")
        start_date = st.sidebar.date_input('Début de période', date[0]+datetime.timedelta(days=5))
        end_date = st.sidebar.date_input('Fin de période', date[len(date)-1])
    elif TimeVision == 'Temps réel':
        st.write(TimeVision)
    elif TimeVision == 'Prévisions':
        
        url_weather = "http://www.infoclimat.fr/public-api/gfs/xml?_ll=43.57202,2.01227&_auth=VE4FElEvUnBWe1BnBXMGL1c%2FU2YAdlJ1C3dQMw5gXiNTNVU2UTABY1U4B3oOIQQyByoGZlxiCTMKawpqD30AfFQ1BWhROlI0VjBQNwU3Bi1Xe1MuAD5SdQt3UDYOY140Uy5VMFE7AX1VOwdsDjcELgc8BmZcfAkuCmgKag9iAGJUNQVkUTVSMVY8UDEFKgYtV2JTNgA5Um0LaVBhDjBePFMxVTdRYQFgVTgHYw4gBDYHPQZjXGEJMApgCmUPZQB8VCgFGFFBUi1WeVBwBWAGdFd5U2YAYVI%2B&_c=2154afe7f27ed3dd1101e78462166290"
        response = requests.get(url_weather)
        HTML_content = BeautifulSoup(response.content, 'html.parser')
        Table = HTML_content.findAll('echeance')

        DayMenu=[i for i in range(70)]; nb_day =0
        DataMeteo = pd.DataFrame( columns=['Jour','Temperature [°C]','Humidité [%]','Pression [Hpa]','Vitesse vent [km/h]', 'Direction vent [°]', 'Nébulosité [%]'])

        for cell in Table:
            print (str(cell))
            #jour - heure
            Day = str(cell)
            PatternDbt=' timestamp="'
            result_dbt = Day.find(PatternDbt)
            result_fin = Day.find('UTC">')
            DayMenu[nb_day] = Day[result_dbt+len(PatternDbt):result_fin-10]
            Date = Day[result_dbt+len(PatternDbt):result_fin-1]
            #Température
            PatternDbt='<temperature><level val="2m">'
            result_dbt = Day.find(PatternDbt)
            result_fin = Day.find('</level><level val="sol">')
            Temp = float(Day[result_dbt+len(PatternDbt):result_fin])-273.15
            #Pression
            PatternDbt='<level val="niveau_de_la_mer">'
            result_dbt = Day.find(PatternDbt)
            result_fin = Day.find('</level></pression>')
            Pression = float(Day[result_dbt+len(PatternDbt):result_fin])/100
            #Humidité
            PatternDbt='<humidite><level val="2m">'
            result_dbt = Day.find(PatternDbt)
            result_fin = Day.find('</level></humidite>')
            Humid = float(Day[result_dbt+len(PatternDbt):result_fin])
            #V_Vent
            PatternDbt='<vent_moyen><level val="10m">'
            result_dbt = Day.find(PatternDbt)
            result_fin = Day.find('</level></vent_moyen>')
            V_vent = float(Day[result_dbt+len(PatternDbt):result_fin])
            #Dir_Vent
            PatternDbt='</vent_rafales><vent_direction><level val="10m">'
            result_dbt = Day.find(PatternDbt)
            result_fin = Day.find('</level></vent_direction>')
            Dir_Vent = float(Day[result_dbt+len(PatternDbt):result_fin])%360
            #Nebulosité
            PatternDbt='</level><level val="totale">'
            result_dbt = Day.find(PatternDbt)
            result_fin = Day.find('</level></nebulosite>')
            Nebul = float(Day[result_dbt+len(PatternDbt):result_fin])
            
            NvelleLigne =  {'Jour':Date,
                            'Temperature [°C]':Temp,
                            'Humidité [%]':Humid,
                            'Pression [Hpa]':Pression,
                            'Vitesse vent [km/h]':V_vent,
                            'Direction vent [°]':Dir_Vent,
                            'Nébulosité [%]':Nebul}
            
            DataMeteo = pd.concat([DataMeteo, pd.DataFrame([NvelleLigne])], ignore_index=True)

            nb_day=nb_day+1;

        for i in range(len(DayMenu)):
            if i>=nb_day:   
                DayMenu.pop(len(DayMenu)-1)

        DayMenuset = set(DayMenu)
        DayMenu=list(DayMenuset)
        DayMenu.sort()

        st.sidebar.selectbox('Choisissez le jour de prévisions',DayMenu)
        st.sidebar.write('Prévisions météo issues du site: https://www.infoclimat.fr/')


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
