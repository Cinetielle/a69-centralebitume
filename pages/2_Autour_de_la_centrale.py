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

from DATA.EMISSIONS_CENTRALES import emission_ATOSCA, emission_CAREPS_moy, emission_DREAL #en g/m3
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

@st.cache_data
def load_data():
    x0, y0, z0 = 623208.070, 6273468.332, 230-5+19
    routes = gpd.read_file('./DATA/ROUTES/route_CBPuylau.shp')
    batiments = gpd.read_file('./DATA/BATIMENTS/batiments_CBPuylaurens.shp')
    
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
    return routes, batiments, Z, extent, X, Y, X_, Y_, dist_XY, meteo, vx, vy, nx, ny

LegendMap = Image.open('./im/mapLegend.png')

def meteo_slice(fin_jour, debut_jour):
    filtre = (meteo.index >= pd.to_datetime('2021/03/06')) & (meteo.index <= pd.to_datetime('2021/03/06')+timedelta(days=30*6))
    filtre = (meteo.index >= pd.to_datetime(debut_jour)) & (meteo.index <= fin_jour)
    meteo_slice = meteo.iloc[filtre, [5, 6, 7, 8, 10, 11, 12, 13, 14]]

    v = meteo_slice.iloc[:, 3].resample('d').mean()/3.6 # vitesse du vent en m/s
    Pa = meteo_slice.iloc[:, 6].resample('d').mean()*1E2  # pression atmosphérique en Pa
    Ta = meteo_slice.iloc[:, 0].resample('d').mean() # température de l'air en °C
    RSI = meteo_slice.iloc[:, 7].resample('d').mean()  # insolation solaire moyenne sur 24H
    HR = meteo_slice.iloc[:, 2].resample('d').mean() # Humidité moyenne sur 24H

    #vecteur vent
    meteo_slice['vdir_cos']=np.cos(meteo_slice.iloc[:, 4]*np.pi/180)
    meteo_slice['vdir_sin']=np.sin(meteo_slice.iloc[:, 4]*np.pi/180)
    
    vVent = np.asarray([meteo_slice['vdir_sin'].resample('d').mean().to_numpy(), meteo_slice['vdir_cos'].resample('d').mean().to_numpy()]).T*-1
    
    vVent = (meteo_slice.iloc[:, 3].resample('d').mean().to_numpy()[:, np.newaxis]/3.6)*vVent
    return v, Ta, Pa, RSI, HR, vVent 

def surelevation(meteo):
    global Vs, v, d, Ts, Ta, Pa, Qh, RSI, HR, vVent, increment
    debut_jour = st.sidebar.date_input("Choisir le jour de début des émissions :", pd.to_datetime('2021/03/06'), key='start')
    increment = st.sidebar.slider("Choisir la durée d'exposition (en jours):", value=30, min_value=0, max_value=60, step=1)

    fin_jour = pd.to_datetime(st.session_state.start)+timedelta(days=increment)
    Vs = st.sidebar.slider(r"Choisir la vitesse ($m.s^{-1}$) des gaz en sortie de cheminée ", value=13.9, min_value=8., max_value=23.4, step=0.1)
    d = 1.35
    Ts = st.sidebar.slider(r"Choisir la température en sortie de cheminée", value=110, min_value=45, max_value=155, step=5)
    v, Ta, Pa, RSI, HR, vVent = meteo_slice(fin_jour, st.session_state.start)


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

@st.cache_data
def plot_composés(Z, x0, y0, extent, C, Cmax, Cmean, Flux, titre, contour_aigue, contour_aigue_color, contour_chronique, contour_chronique_color, VTR, ERU=None, contour_ERU=[1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3], contour_ERU_color=["indigo", "navy", 'teal', "lightgreen", 'orange', "fuchsia"], background=None):
    st.markdown(f"""
                    
            ## {titre}""", unsafe_allow_html=True)
            

    st.markdown("""<p>Pour les toxiques à seuil, il existe des valeurs toxicologiques de référence (VTR), en dessous desquelles l'exposition est réputée sans risque. Ces valeurs toxicologiques de référence, basées sur les connaissances scientifiques, sont fournies, pour chaque voie d'exposition, dans des bases de données réalisées par différents organismes internationaux. (extrait de l'étude du CAREPS) </p>
                <p> Nous pouvons ajouter que les seuils dépendent de la durée d'exposition; il existe donc des seuils pour des expositions aigues (intense sur une courte période) et chronique (moins intense mais sur une période plus longue voir continue) </p>
                <p> La méthode utilisée ne permet qu'une description journalière. Pour une analyse plus fine des expositions aigu, un autre type de modélisation doit etre envisagé (dans un périmètre proche de la centrale notamment).</p>
                <p style="color:red">On peut considérer qu'il existe un risque sanitaire si l'indice de risque individuel dépasse 1.</p>
                """, unsafe_allow_html=True)
    
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.hist(Flux.flatten(), bins=10, color='k')
    # ax.set_label('Distribution des débits massiques (/s)')
    # st.pyplot(fig)
    
    st.markdown(""" ### Exposition Aigu""", unsafe_allow_html=True)
    st.markdown(""" La concentration moyenne journalière maximum est de :  """, unsafe_allow_html=True)
    st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.round(np.nanmax(Cmax)*1E6, 4)} µg.m<sup style="color:blue; font-size: 30px;">-3</sup> </p>', unsafe_allow_html=True)
    
    if background is None:
        st.markdown("""La concentration moyenne du fond de l'air est inconnue.""", unsafe_allow_html=True)
        fond_air = 0
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
        st.markdown("""La valeur toxicologique de référence pour une exposition aigu retenu est de:""", unsafe_allow_html=True)
        st.markdown(f'<p style="color:blue; font-size: 30px;"> {maxmin} µg.m<sup style="color:blue; font-size: 20px;">-3</sup></p>', unsafe_allow_html=True)
        
        st.markdown("""L'indice de risque individuel (Concentration/VTR) maximum lié à la seule centrale est de:""", unsafe_allow_html=True)
        st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.round(np.nanmax(Cmax)*1E6/maxmin, 4)} </p>', unsafe_allow_html=True)
        
        if fond_air != 0:
            st.markdown("""L'indice de risque individuel (Concentration/VTR) maximum total est de:""", unsafe_allow_html=True)
            st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.round((np.nanmax(Cmax)*1E6+fond_air)/maxmin, 4)} </p>', unsafe_allow_html=True)
        
    fig, ax = plt.subplots(figsize=(10, 10))
    #ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
    im = ax.contourf((Cmax*1E6+fond_air)/maxmin, [1E-3, 1E-2, 1E-1, 5E-1, 1E0, 2E0, 5E0, 1E1, 1E2], extent=extent, colors=["indigo", "navy", 'teal', "lightgreen", "gold", "orangered", "crimson", "fuchsia"], origin='lower', zorder=1) 
    batiments.plot(ax=ax, color='k')
    routes.plot(ax=ax, color='gray')
    ax.scatter(x0, y0, c='crimson', zorder=3)
    divider = make_axes_locatable(ax)
    ax.set_xlim(x0-xmax*1E3, x0+xmax*1E3)
    ax.set_ylim(y0-xmax*1E3, y0+xmax*1E3)
    cax = divider.append_axes("top", size="7%", pad="10%")
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Indice de risque individuel sur la période choisie')
    st.pyplot(fig)
    
    st.markdown(""" ### Exposition Chronique """, unsafe_allow_html=True)

    st.markdown(""" La concentration moyenne maximum sur la période sélectionnée est de :  """, unsafe_allow_html=True)
    st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.round(np.nanmax(Cmean)*1E6, 3)}  µg.m<sup style="color:blue; font-size: 30px;">-3</sup></p>', unsafe_allow_html=True)
    
    if background is None:
        st.markdown("""La concentration moyenne du fond de l'air est inconnue.""", unsafe_allow_html=True)
        fond_air = 0
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
        st.markdown("""La valeur toxicologique de référence pour une exposition chronique retenu est de:""", unsafe_allow_html=True)
        st.markdown(f'<p style="color:blue; font-size: 30px;"> {maxmin} µg.m<sup style="color:blue; font-size: 20px;">-3</sup></p>', unsafe_allow_html=True)
        
        st.markdown("""L'indice de risque individuel (Concentration/VTR) maximum lié à la seule centrale est de:""", unsafe_allow_html=True)
        st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.round(np.nanmax(Cmean)*1E6/maxmin, 4)} </p>', unsafe_allow_html=True)
        
        if fond_air != 0:
            st.markdown("""L'indice de risque individuel (Concentration/VTR) maximum total est de:""", unsafe_allow_html=True)
            st.markdown(f'<p style="color:blue; font-size: 30px;"> {np.round((np.nanmax(Cmean)*1E6+fond_air)/maxmin, 4)} </p>', unsafe_allow_html=True)
        
    fig, ax = plt.subplots(figsize=(10, 10))
    #ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
    im = ax.contourf((Cmean*1E6+fond_air)/maxmin, [1E-3, 1E-2, 1E-1, 5E-1, 1E0, 2E0, 5E0, 1E1, 1E2], extent=extent, colors=["indigo", "navy", 'teal', "lightgreen", "gold", "orangered", "crimson", "fuchsia"], origin='lower', zorder=1) 
    batiments.plot(ax=ax, color='k')
    routes.plot(ax=ax, color='gray')
    ax.scatter(x0, y0, c='crimson', zorder=3)
    divider = make_axes_locatable(ax)
    ax.set_xlim(x0-xmax*1E3, x0+xmax*1E3)
    ax.set_ylim(y0-xmax*1E3, y0+xmax*1E3)
    cax = divider.append_axes("top", size="7%", pad="10%")
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label("Indice de risque individuel sur la période choisie")
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
            im = ax.contourf(np.nanmean(C*1E6, axis=0)*fraction_de_vie*np.min(val), contour_ERU, extent=extent, colors=contour_ERU_color, origin='lower', zorder=1) 
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
    #RGF93_to_WGS84 = Transformer.from_crs('2154', '4326', always_xy=True)
    #WGS84_to_RGF93 = Transformer.from_crs('4326', '2154', always_xy=True)
    #coordonnée de la sortie de cheminée
    #-5 mètre pour intégrer le décaissement
    x0, y0, z0 = 623208.070, 6273468.332, 230-5+19

    surelevation(meteo)
    
    C = compute(ny, nx, len(v), dist_XY, X_, Y_, Z, z0)

    
    st.markdown("""
        <p> Pour les calculs suivant nous utilisons les modèles de Briggs et de Pasquill & Grifford (mode 2). Pour plus d'information se reporter au chapitre <a href='Comprendre_Le_Calcul_Et_Ses_Options' target='_self'>détaillant le calcul et ses options</a>. A noter que le calcul peut-etre long et que lorsqu'il s'execute à nouveau, les figures sont semi-transparentes.
        
        Deux types de concentration à l'émission peuvent-être choisis: 
        <ol>
        <li>Les seuils imposés par la DREAL. Dans ce cas, pour certains métaux, la concentration maximum imposée est celle d'un ensemble de composés. Dans ce cas, on suppose que seul le métal sélectionné est émis, ce qui est une hypothèse très majorante.</li>
        <li>Les émissions rapportées par ATOSCA</li>
        </ol>
        
        Concernant la liste des composés, nous avons selectionné ceux pour lesquels nous avons trouvé des VTR et qui semblait majeurs au vu de leur concentration. La liste n'est donc pas exhaustive.
                """, unsafe_allow_html=True)
    
    
    e = st.selectbox("Choisir un composé :",
                     ["SOx", "NOx", "CO", "Formaldéhyde", "Benzene (71-43-2)", "Chrome (Cr)", "Acroléine", "Arsenic (As)", "Nickel (Ni)", "Acide acrylique",
                       "Acétaldéhyde", "Cobalt (Co)", "Phénol", "Cadmium (Cd)",], index=0)
    
    c_mode = st.selectbox("Choisir une concentration à la source :",
                          ["Mesure ATOSCA, pièce E6", 'Seuil DREAL'], index=1)
    
    concentration = {"Mesure ATOSCA, pièce E6":emission_ATOSCA,
                     'Seuil DREAL':emission_DREAL}
    
    #actualise le flux en fonction du débit de la cheminée
    cc0 = concentration[c_mode][e] #en  (g/Nm3)h_17%O2
    centrale = normalize(O2_ref=17, T0=273, P0=101.3E3)
    S=np.pi*(d/2)**2
    O2 = st.sidebar.slider("Choisir la teneur en oxygène (%):", value=14.4, min_value=7., max_value=20., step=0.1)
    H = st.sidebar.slider("Choisir l'humidité relative des émissions (%):", value=13.7, min_value=5., max_value=30., step=0.1)
    centrale.get_Q_norm(Vs, Ta, O2, H, Pa, S=S, output=False)
    
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.hist(centrale.Q0.to_numpy(), bins=10, color='k')
    # ax.set_label('Distribution des débits volumiques  humides Normalisés à 17% O2 (Nm3/h)')
    # st.pyplot(fig)
    
    Flux = centrale.get_DébitMassique(cc0) #en g/s
    Flux = Flux.to_numpy()[:, np.newaxis, np.newaxis]
    Cmax = np.nanmax(C*Flux, axis=0)
    Cmean = np.nanmean(C*Flux, axis=0)
    #ind_max = np.where(Cmean == np.nanmax(Cmean))
    
    plot_composés(Z, x0, y0, extent,
                  C*Flux, Cmax, Cmean,
                  Flux,
                  Composés[e]["titre"], Composés[e]["contour_aigue"], Composés[e]["contour_aigue color"],
                  Composés[e]["contour_chronique"], Composés[e]["contour_chronique color"],
                  Composés[e]["VTR"], ERU=Composés[e]["ERU"], background=Composés[e]["Background"])






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

routes, batiments, Z, extent, X, Y, X_, Y_, dist_XY, meteo, vx, vy, nx, ny = load_data()
xmax = st.sidebar.slider(r"Choisir le rayon d'impact de la centrale [km]", value=1.5, min_value=0.5, max_value=15., step=0.1)
Historique()
