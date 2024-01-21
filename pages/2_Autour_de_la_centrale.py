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
import numpy as np
import streamlit as st
from streamlit.hello.utils import show_code
import pandas as pd
import folium
from streamlit_folium import st_folium
from PIL import Image
import Functions.map_tool as map_tool
import Functions.Meteo_tool as Meteo_tool
from datetime import  datetime, timedelta
import pytz
from pyproj import Transformer
from utils import Δh_Briggs, stability_pasquill, sigma
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd

from DATA.EMISSIONS_CENTRALES import emission_ATOSCA, emission_CAREPS_moy #en g/m3
from DATA.VTR_ERU import Composés
from utils import normalize

villes = {'Revel': [619399.490,6262672.707],
         'Puylaurens':[620266.862,6275241.681],
         'St Germain-des-prés': [624500.311,6274095.737],
         'Soual':[628528.524,6273191.962],
         'Lempaut':[624346.252,6270432.622],
         'Sémalens':[628217.241,6277447.042],
         'Vielmur-Sur-Agout':[626542.116,6280461.738],
         'St Paul-Cap-de-Joux':[617341.329,6283743.400],
         'Villeneuve-lès-lavaur':[602022.138,6278485.487],
         'Lavaur':[604890.514,6289623.373],
         'Saix':[634096.773,6276222.376],
         'Dourgne':[630321.279,6265464.647],
         'Sorèze':[624423.786,6261910.745],
         'Lautrec':[630493.276,6289921.993],
         'Graulhet':[618445.042,6296272.167],
         'Blan':[619758.761,6270229.387]}

routes = gpd.read_file('./DATA/ROUTES/route_CBPuylau.shp')
batiments = gpd.read_file('./DATA/BATIMENTS/batiments_CBPuylaurens.shp')

LegendMap = Image.open('./im/mapLegend.png')

def surelevation(meteo):
    global Vs, v, d, Ts, Ta, Pa, Qh, RSI, HR, vVent, increment
    debut_jour = st.sidebar.date_input("Choisir le jour de début des émissions :", pd.to_datetime('2021/03/06'), key='start')
    if (meteo.index[-1]-pd.to_datetime(st.session_state.start)).days < 30*6:
        increment = st.sidebar.slider("Choisir la durée des émissions (en jours):", value=1, min_value=0, max_value=(meteo.index[-1]-pd.to_datetime(st.session_state.start)).days, step=1)
    else:
        increment = st.sidebar.slider("Choisir la durée d'exposition (en jours):", value=30*6, min_value=0, max_value=(meteo.index[-1]-pd.to_datetime(st.session_state.start)).days, step=1)

    fin_jour = pd.to_datetime(st.session_state.start)+timedelta(days=increment)

    filtre = (meteo.index >= pd.to_datetime(st.session_state.start)) & (meteo.index <= fin_jour)
    meteo_slice = meteo.iloc[filtre, [5, 6, 7, 8, 10, 11, 12, 13, 14]]
    Vs = st.sidebar.slider(r"Choisir la vitesse ($m.s^{-1}$) des gaz en sortie de cheminée ", value=13.9, min_value=8., max_value=23.4, step=0.1)
    d = 1.35
    Ts = st.sidebar.slider(r"Choisir la température en sortie de cheminée", value=110, min_value=80, max_value=150, step=1)

    v = meteo_slice.iloc[:, 3].resample('d').mean()/3.6 # vitesse du vent en m/s
    Pa = meteo_slice.iloc[:, 4].resample('d').mean()  # pression atmosphérique en Pa
    Ta = meteo_slice.iloc[:, 0].resample('d').mean() # température de l'air en °C
    RSI = meteo_slice.iloc[:, 7].resample('d').mean()  # insolation solaire moyenne sur 24H
    HR = meteo_slice.iloc[:, 2].resample('d').mean() # Humidité moyenne sur 24H

    #vecteur vent
    meteo_slice['vdir_cos']=np.cos(meteo_slice.iloc[:, 4]*np.pi/180)
    meteo_slice['vdir_sin']=np.sin(meteo_slice.iloc[:, 4]*np.pi/180)
    
    vVent = np.asarray([meteo_slice['vdir_sin'].resample('d').mean().to_numpy(), meteo_slice['vdir_cos'].resample('d').mean().to_numpy()]).T*-1
    
    vVent = (meteo_slice.iloc[:, 3].resample('d').mean().to_numpy()[:, np.newaxis]/3.6)*vVent

def compute(ny, nx, n_slice, dist_XY, X_, Y_, Z, z0):
    C = np.zeros((n_slice, ny, nx))
    for i in range(n_slice):
        Δh = Δh_Briggs(dist_XY, Vs, v[i], d, Ts, Ta[i])
        dot_product=X_*vVent[i, 0]+Y_*vVent[i, 1]
        magnitudes=v[i]*dist_XY
        # angle entre la direction du vent et le point (x,y)
        subtended=np.arccos(dot_product/(magnitudes+1e-15));
        # distance le long de la direction du vent jusqu'à une ligne perpendiculaire qui intersecte x,y
        downwind=np.cos(subtended)*dist_XY
        filtre = np.where(downwind > 0)
        crosswind=np.sin(subtended)*dist_XY
        SA_climatique = stability_pasquill(v[i], RSI[i], HR[i], mode='24H')
        sr = sigma(SA_climatique, downwind)
        σy =sr[1, 0, filtre[0],filtre[1]]
        σz =sr[1, 1, filtre[0],filtre[1]]
        C[i, filtre[0], filtre[1]] = (np.exp(-crosswind[filtre[0],filtre[1]]**2./(2.*σy**2.))* np.exp(-(Z[filtre[0],filtre[1]] -z0- Δh[filtre[0],filtre[1]])**2./(2.*σz**2.)))/(2.*np.pi*v[i]*σy*σz)
    return C

def plot_composés(Z, x0, y0, extent, C, Cmax, Cmean, cc, titre, contour_aigue, contour_aigue_color, contour_chronique, contour_chronique_color, VTR, ERU=None, contour_ERU=[1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3], contour_ERU_color=["indigo", "navy", 'teal', "lightgreen", 'orange', "fuchsia"], background=None):
    st.markdown(f"""
                    
            ## {titre}""", unsafe_allow_html=True)
            

    st.markdown("""<p>Pour les toxiques à seuil, il existe des valeurs toxicologiques de référence (VTR), en dessous desquelles l'exposition est réputée sans risque. Ces valeurs toxicologiques de référence, basées sur les connaissances scientifiques, sont fournies, pour chaque voie d'exposition, dans des bases de données réalisées par différents organismes internationaux. (extrait de l'étude du CAREPS) </p>
                <p> Nous pouvons ajouter que les seuils dépendent de la durée d'exposition; il existe donc des seuils pour des expositions aigues (intense sur une courte période) et chronique (moins intense mais sur une période plus longue voir continue) </p>
                <p> La méthode utilisée ne permet qu'une description journalière. Pour une analyse plus fine des expositions aigu, un autre type de modélisation doit etre envisagé (dans un périmètre proche de la centrale notamment).</p>
                <p style="color:red">On peut considérer qu'il existe un risque sanitaire si l'indice de risque individuel dépasse 1.</p>
                """, unsafe_allow_html=True)
    
    st.markdown(""" ### Exposition Aigu""", unsafe_allow_html=True)
    st.markdown(""" La concentration moyenne journalière maximum est de :  """, unsafe_allow_html=True)
    st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.nanmax(Cmax)*cc*1E6} µg.m<sup style="color:blue; font-size: 30px;">-3</sup> </p>', unsafe_allow_html=True)
    
    if background is None:
        st.markdown("""La concentration moyenne du fond de l'air est inconnue.""", unsafe_allow_html=True)
        fond_air = np.nan
    else:
        st.markdown("""La concentration du fond de l'air est de :""", unsafe_allow_html=True)
        st.table(background)
        fond_air = background.iloc[:, 1].mean()
    
    st.markdown("""Les seuils sanitaires pour une exposition aigu sont les suivants :""", unsafe_allow_html=True)
    maxmin=1E6
    for vtr in VTR:
        if len(vtr[0].iloc[vtr[2], :]) > 0:
            st.write(f'Source: {vtr[1]}')
            st.table(vtr[0].iloc[vtr[2], :])
            if len(vtr[0].iloc[vtr[2], 1]) > 0:
                maxmin = np.nanmin([np.nanmin(vtr[0].iloc[vtr[2], 1]), maxmin])
            
    if (maxmin != 1E6):
        st.markdown("""L'indice de risque individuel (Concentration/VTR) maximum lié à la seule centrale est de:""", unsafe_allow_html=True)
        st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.nanmax(Cmax)*cc*1E6/maxmin} </p>', unsafe_allow_html=True)
        
        if ~np.isnan(fond_air):
            st.markdown("""L'indice de risque individuel (Concentration/VTR) maximum total est de:""", unsafe_allow_html=True)
            st.markdown(f'<p style="color:blue; font-size: 30px;"> {(np.nanmax(Cmax)*cc*1E6+fond_air)/maxmin} </p>', unsafe_allow_html=True)
        
    fig, ax = plt.subplots(figsize=(10, 10))
    #ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
    im = ax.contourf(Cmax*cc*1E6, contour_aigue, extent=extent, colors=contour_aigue_color, origin='lower', zorder=1) 
    batiments.plot(ax=ax, color='k')
    routes.plot(ax=ax, color='gray')
    ax.scatter(x0, y0, c='crimson', zorder=3)
    divider = make_axes_locatable(ax)
    ax.set_xlim(x0-xmax*1E3, x0+xmax*1E3)
    ax.set_ylim(y0-xmax*1E3, y0+xmax*1E3)
    cax = divider.append_axes("top", size="7%", pad="10%")
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(r"Concentration moyenne journalière maximale dans l'air sur la période choisie ")
    st.pyplot(fig)
    
    st.markdown(""" ### Exposition Chronique """, unsafe_allow_html=True)

    st.markdown(""" La concentration moyenne maximum sur la période sélectionnée est de :  """, unsafe_allow_html=True)
    st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.nanmax(Cmean)*cc*1E6}  µg.m<sup style="color:blue; font-size: 30px;">-3</sup></p>', unsafe_allow_html=True)
    
    if background is None:
        st.markdown("""La concentration moyenne du fond de l'air est inconnue.""", unsafe_allow_html=True)
        fond_air = np.nan
    else:
        st.markdown("""La concentration du fond de l'air est de :""", unsafe_allow_html=True)
        st.table(background)
        fond_air = background.iloc[:, 1].mean()
    
    st.markdown("""Les seuils sanitaires pour une exposition chronique sont les suivants :""", unsafe_allow_html=True)
    maxmin=1E6
    for vtr in VTR:
        if len(vtr[0].iloc[vtr[3], :]) > 0:
            st.write(f'Source: {vtr[1]}')
            st.table(vtr[0].iloc[vtr[3], :])
            if len(vtr[0].iloc[vtr[3], 1]) > 0:
                maxmin = np.nanmin([np.nanmin(vtr[0].iloc[vtr[3], 1]), maxmin])
            
    if (maxmin != 1E6):
        st.markdown("""L'indice de risque individuel (Concentration/VTR) maximum lié à la seule centrale est de:""", unsafe_allow_html=True)
        st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.nanmax(Cmean)*cc*1E6/maxmin} </p>', unsafe_allow_html=True)
        
        if ~np.isnan(fond_air):
            st.markdown("""L'indice de risque individuel (Concentration/VTR) maximum total est de:""", unsafe_allow_html=True)
            st.markdown(f'<p style="color:blue; font-size: 30px;"> {(np.nanmax(Cmean)*cc*1E6+fond_air)/maxmin} </p>', unsafe_allow_html=True)
        
    fig, ax = plt.subplots(figsize=(10, 10))
    #ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
    im = ax.contourf(Cmean*cc*1E6, contour_chronique, extent=extent, colors=contour_chronique_color, origin='lower', zorder=1) 
    batiments.plot(ax=ax, color='k')
    routes.plot(ax=ax, color='gray')
    ax.scatter(x0, y0, c='crimson', zorder=3)
    divider = make_axes_locatable(ax)
    ax.set_xlim(x0-xmax*1E3, x0+xmax*1E3)
    ax.set_ylim(y0-xmax*1E3, y0+xmax*1E3)
    cax = divider.append_axes("top", size="7%", pad="10%")
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(r"Concentration moyenne dans l'air sur la période choisie ($µg.m^{-3}$)")
    st.pyplot(fig)
    
    if ERU is not None:
        
        st.markdown("""
                    ### Exposition Chronique sans seuil 
                    <p> Pour les toxiques sans seuil, les mêmes instances internationales ont défini pour certains composés chimiques la probabilité, pour un individu, de développer un cancer lié à une exposition égale, en moyenne sur sa durée de vie, à une unité de dose (1 μg.m-3 pour l’inhalation) de la substance toxique. Ces probabilités sont exprimées, pour la plupart des organismes, par un excès de risque unitaire (ERU). Un ERU à 10-5 signifie qu’une personne exposée en moyenne durant sa vie à une unité de dose, aurait une probabilité supplémentaire de 0,00001, par rapport au risque de base, de contracter un cancer lié à cette exposition. Le CIRC, l'EPA et l’Union Européenne ont par ailleurs classé la plupart des composés chimiques en fonction de leur cancérogénicité.(extrait de l'étude du CAREPS)</p>""", unsafe_allow_html=True)
        for eru in ERU:
            st.write(f'Source: {eru[1]}')
            st.table(eru[0])
            val = eru[0].iloc[:, 1].to_numpy()
            fig, ax = plt.subplots(figsize=(10, 10))
            #ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
            fraction_de_vie = (increment/365)/85.3#durée de vie moyenne : 80ans
            im = ax.contourf(np.nanmean(C*cc*1E6, axis=0)*fraction_de_vie*np.min(val), contour_ERU, extent=extent, colors=contour_ERU_color, origin='lower', zorder=1) 
            batiments.plot(ax=ax, color='k')
            routes.plot(ax=ax, color='gray')
            ax.scatter(x0, y0, c='crimson', zorder=3)
            divider = make_axes_locatable(ax)
            ax.set_xlim(x0-xmax*1E3, x0+xmax*1E3)
            ax.set_ylim(y0-xmax*1E3, y0+xmax*1E3)
            cax = divider.append_axes("top", size="7%", pad="15%")
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_label(r"Probabilité accrue de développer un cancer sur la période choisie"+" \n échelle de couleur (de gauche à droite) : \n 1/1 milliard, 1/100 millions, ... 1/10 000, 1/1000")
            st.pyplot(fig)
        
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
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    
    meteo = pd.read_csv('./DATA/METEO/Donnees_meteo_Puylaurens.csv', sep=';', encoding='UTF-8')
    meteo.index = pd.to_datetime(meteo.iloc[:, :5])
    
    surelevation(meteo)
    
    C = compute(ny, nx, len(v), dist_XY, X_, Y_, Z, z0)
    Cmax = np.nanmax(C, axis=0)
    Cmean = np.nanmean(C, axis=0)
    ind_max = np.where(Cmean == np.nanmax(Cmean))
    
    st.markdown("""
        <p> Pour les calculs suivant nous utilisons les modèles de Briggs et de Pasquill & Grifford (mode 2). Pour plus d'information se reporter au chapitre <a href='Comprendre_Le_Calcul_Et_Ses_Options' target='_self'>détaillant le calcul et ses options</a>. A noter que le calcul peut-etre long et que lorsqu'il s'execute à nouveau, les figures sont semi-transparentes.
        """, unsafe_allow_html=True)
    
    
    e = st.selectbox("Choisir un composé :",
                     ["SOx", "NOx", "CO", "Formaldéhyde", "Benzene (71-43-2)", "Chrome (Cr)", "Acroléine", "Arsenic (As)", "Nickel (Ni)", "Acide acrylique",
                       "Acétaldéhyde", "Cobalt (Co)", "Phénol", "Cadmium (Cd)",], index=0)
    
    c_mode = st.selectbox("Choisir une concentration à la source :",
                          ["Mesure ATOSCA, pièce E6", "Moyenne des centrales, d'après l'étude du CAREPS", "Maximum des centrales, d'après l'étude du CAREPS", 'Seuil DREAL'], index=0)
    
    concentration = {"Mesure ATOSCA, pièce E6":emission_ATOSCA,
                     "Moyenne des centrales, d'après l'étude du CAREPS":emission_CAREPS_moy,
                     "Maximum des centrales, d'après l'étude du CAREPS":None,
                     'Seuil DREAL':None}
    #actualise la concentration en fonction du débit de la cheminée
    cc0 = concentration[c_mode][e] #en  g/s
    cc = (Vs*(d/2)**2*np.pi*cc0)/(13.9*(d/2)**2*np.pi)
    #en g/s
    plot_composés(Z, x0, y0, extent,
                  C, Cmax, Cmean,
                  cc,
                  Composés[e]["titre"], Composés[e]["contour_aigue"], Composés[e]["contour_aigue color"],
                  Composés[e]["contour_chronique"], Composés[e]["contour_chronique color"],
                  Composés[e]["VTR"], ERU=Composés[e]["ERU"], background=Composés[e]["Background"])




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
    xmax = st.sidebar.slider(r"Choisir le rayon d'impact de la centrale [km]", value=1.5, min_value=0.5, max_value=15., step=0.1)
    TimeVision = st.sidebar.selectbox('Quelles données voulez-vous consulter?',('Historique', 'Temps réel', 'Prévisions'))
    if TimeVision == 'Historique':
        Historique()

    elif TimeVision == 'Temps réel':
        Tps_réel()

    elif TimeVision == 'Prévisions':
        Prévisions()


st.set_page_config(page_title="Autour de la centrale", page_icon="")
st.markdown("# Impacts de la centrale à bitume")
st.sidebar.header("Paramètres")
st.markdown(
    """
    Cette page permet de visualiser les impacts de la centrale à bitume de Puylaurens.
    
    Les impacts dans les conditions météorologiques suivantes sont mal évalué par cette méthode de calcul :  inversion atmosphérique, brume, turbulence.
    
     <p>Les calculs réalisés ici sont des prévisions.</p> <p style="color:red">La réalité peut diverger de ces calculs.</p> <p>Des mesures in-situ sont donc indispensables.</p> <p>Ces prévisions permettent notamment d'optimiser les dispositifs de mesure.</p>
    """, unsafe_allow_html=True
)
data_explore()
