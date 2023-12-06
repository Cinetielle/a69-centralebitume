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

import datetime

import numpy as np
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code

import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import branca.colormap as cmp
import folium
from streamlit_folium import st_folium
from pyproj import Transformer

image_DP = Image.open('./im/E9F7Q18WEAc7P8_.jpeg')
image_DP2 = Image.open('./im/Gaussian_Plume_fr.png')
image_DP3 = Image.open('./im/Turner1970.png')

RGF93_to_WGS84 = Transformer.from_crs('2154', '4326', always_xy=True)
WGS84_to_RGF93 = Transformer.from_crs('4326', '2154', always_xy=True)

#coordonnée de la sortie de cheminée
#-5 mètre pour intégrer le décaissement
x0, y0, z0 = 623208.070, 6273468.332, 230-5+19

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

def topographie():
    fig, ax = plt.subplots()
    ax.scatter(x0, y0, c='red', label='Usine à bitume RF500')
    for n, l in villes.items():
        plt.scatter(l[0], l[1], c='w', s=10, zorder=2)
        plt.text(l[0]-600, l[1]+600, n, fontsize=6, zorder=3)
    f = ax.imshow(Z, origin='lower', extent=extent, cmap='terrain')
    plt.colorbar(f, ax=ax).set_label('Altitude (m)')
    ax.legend()
    st.pyplot(fig)

def topographie_zoom():
    idxs, idxe = 250, 350
    idys, idye = 225, 285
    extent = [X.min()+75*idxs, X.max()-75*(len(vx)-idxe), Y.min()+75*idys, Y.max()-75*(len(vy)-idye)]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(x0, y0, c='red', label='Usine à bitume RF500')
    ax.scatter(621669.837, 6274592.660, c='k', label='Station météo')
    for n, l in villes.items():
        if n in ['Puylaurens','St Germain-des-prés']:
            plt.scatter(l[0], l[1], c='w', s=10, zorder=2)
            plt.text(l[0]-500, l[1]+100, n, c='w', fontsize=6, zorder=3)
    f = ax.imshow(Z[idys:idye, idxs:idxe], origin='lower', extent=extent, cmap='terrain')
    plt.colorbar(f, ax=ax, orientation='horizontal').set_label('Altitude (m)')
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    ax.legend(loc='lower left')
    st.pyplot(fig)

def Δh_Holland(Vs, v, d, Pa, Ts, Ta):
    """

    Parameters
    ----------
    Pa : TYPE
        pression atmospherique en mbar
    Vs : TYPE
        vitesse des gaz en sortie de cheminée en m/s
    v : TYPE
        vitesse du vent à la hauteur de la cheminée en m/s
    d : TYPE
        Diamètre de la cheminée en m
    Ts : TYPE
        Température en sortie de cheminée en °K
    Ta : TYPE
        Temperature ambiante en °K

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return Vs*d*(1.5+0.00268*Pa*d*((Ts-Ta)/(Ts+273.15)))/v

def Δh_CarsonAndMoses(Vs, v, d, Qh):
    """
    Parameters
    ----------
    Vs : TYPE
        vitesse des gaz en sortie de cheminée en m/s
    v : TYPE
        vitesse du vent à la hauteur de la cheminée en m/s
    d : TYPE
        Diamètre de la cheminée en m
    Qh : TYPE
        Débit de Chaleur en kJ/s

    Returns
    -------
    TYPE
        surélévation du panache
    """
    return -0.029*Vs*d/v + 2.62*np.sqrt(Qh)*d/v

def Δh_Concawes(v, d, Qh):
    """
    Parameters
    ----------
    v : TYPE
        vitesse du vent à la hauteur de la cheminée
    d : TYPE
        Diamètre de la cheminée
    Qh : TYPE
        Débit de Chaleur en kJ/s

    Returns
    -------
    TYPE
        surélévation du panache
    """
    return 2.71*np.sqrt(Qh)*d/v**(3/4)

def Δh_Briggs(x, Vs, v, d, Ts, Ta):
    """
    Formule la plus utilisée

    Parameters
    ----------
    x : TYPE
        distance XY à la source
    Vs : TYPE
        vitesse des gaz en sortie de cheminée en m/s
    v : TYPE
        vitesse du vent à la hauteur de la cheminée en m/s
    d : TYPE
        Diamètre de la cheminée en m
    Ts : TYPE
        Température en sortie de cheminée en °C
    Ta : TYPE
        Temperature ambiante en °C

    Returns
    -------
    res : TYPE
        surélévation d’un panache émis par une source ponctuelle

    """
    #effet dynamique en m4/s2
    Fm = (Ta * Vs**2 * (d/2)**2)/(Ts*4)
    #effet de flottabilité en m4/s3
    g = 9.81
    Fb = (g*Vs*(d/2)**2*(Ts-Ta))/(Ts*4)
    if Fb < 55:
        #limite d'application de la surélévation
        xf = 49*Fb**(5/8)
    else:
        xf = 119*Fb**(2/5)
    res = (3*Fm*x/(0.36*v**2) + 4.17*Fb*x**2/v**3)**(1/3)
    res[x > xf] = (3/(2*0.6**2))**(1/3) * (Fb**(1/3)*x[x>xf]**(2/3))/v
    return res
                                           
def surelevation():
    global Vs, v, d, Ts, Ta, xmax, Qh, RSI, HR, vVent
    td = meteo.index[-1]-meteo.index[0]
    date_meteo_increment =  st.sidebar.slider("Choisir rapidement une nouvelle journée après le 6 mars 2021", value=0, min_value=0, max_value=td.days, step=1)
    date_meteo = st.sidebar.date_input("Choisir la météo d'une journée particulière", pd.to_datetime('2021/03/06')+datetime.timedelta(days=date_meteo_increment))
    debut_jour = pd.to_datetime(date_meteo)
    fin_jour = pd.to_datetime(date_meteo)+datetime.timedelta(days=1)

    filtre = (meteo.index >= debut_jour) & (meteo.index <= fin_jour)
    meteo_slice = meteo.iloc[filtre, [5, 6, 7, 8, 10, 11, 12, 13, 14]]
    xmax = st.sidebar.slider(r"Choisir la distance maximale où évaluer les impacts", value=5000, min_value=1000, max_value=50000, step=10)
    Vs = st.sidebar.slider(r"Choisir la vitesse $m.s^{-1}$) des gaz en sortie de cheminée ", value=13.9, min_value=8., max_value=23.4, step=0.1)
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

    Hair = 1940 # enthalpie de l'air à 100% d'humidité relative et 83°C en kJ/kg
    debit_masse_air =(53400*0.94)/3600 #kg/s tel que donné dans le document SPIE
    Qh = Hair*debit_masse_air  #débit de chaleur en kJ/s

    x = np.arange(0, xmax, 10)
    briggs = Δh_Briggs(x, Vs, v, d, Ts, Ta)
    Concawes = Δh_Concawes(v, d, Qh)
    CarsonAndMoses = Δh_CarsonAndMoses(Vs, v, d, Qh)
    Holland = Δh_Holland(Vs, v, d, Pa, Ts, Ta)

    fig, ax = plt.subplots()
    ax.plot(x, briggs, label='Briggs')
    ax.plot([0, xmax], [Holland, Holland], label='Holland')
    ax.plot([0, xmax], [Concawes, Concawes], label='Concawes')
    ax.plot([0, xmax], [CarsonAndMoses, CarsonAndMoses], label='Carson & Moses')
    ax.set_ylabel('Hauteur au dessus de la cheminée (m)')
    ax.set_xlabel('Distance à la cheminée (m)')
    ax.legend()
    ax.set_title("Hauteur du centre du panache dans la direction du vent \n selon différents modèles")
    st.pyplot(fig)

def  stability_pasquill(v, RSI, HR, mode='24H'):
    """
    Définis les conditions suivantes et renvoi la stabilité de Pasquill en conséquence
    'JOUR_RSI_fort', 'JOUR_RSI_modéré', 'JOUR_RSI_faible', 'NUIT_NEBULOSITE_4/8-7/8', 'NUIT_NEBULOSITE_<3/8'

    Parameters
    ----------
    v : float
        wind speed at 10m , m/s
    RSI : float
        RSI= rayonnement solaire incident moyen
    HR : float
        humidité relative
        on fait l'hypothèse que si l'air au sol est sec (< 60 %) la nébuolisté équivaut à 'NUIT_NEBULOSITE_<3/8'
        sinon la nébulosité équivaut à 'NUIT_NEBULOSITE_4/8-7/8'

    mode : str, optional
        '24H', '1H', '10M'
        

    Returns
    -------
    Pasquill stability
    A : très instable
    B: instable
    C: peu instable
    D: neutre
    E:stable
    F:très stable
    """
    if mode == "24H":
        if RSI > 264:
            RSI='JOUR_RSI_fort'
        elif (RSI < 90):
            RSI='JOUR_RSI_faible'
        else:
            RSI='JOUR_RSI_modéré'
    else:
        if RSI > 472:
            RSI='JOUR_RSI_fort'
        elif (RSI < 199) & (RSI > 0):
            RSI='JOUR_RSI_faible'
        elif (HR >= 60) & (RSI == 0):
            RSI='NUIT_NEBULOSITE_<3/8'
        elif (HR < 60) & (RSI == 0):
            RSI='NUIT_NEBULOSITE_4/8-7/8'            
        else:
            RSI='JOUR_RSI_modéré'
    
    if v < 2:
        if RSI=='JOUR_RSI_fort':
            return 'A'
        elif RSI=='JOUR_RSI_modéré':
            return 'A-B'
        elif RSI=='JOUR_RSI_faible':
            return 'B'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'F'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'F'
    elif (v >= 2) & (v < 3):
        if RSI=='JOUR_RSI_fort':
            return 'A-B'
        elif RSI=='JOUR_RSI_modéré':
            return 'B'
        elif RSI=='JOUR_RSI_faible':
            return 'C'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'E'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'F'
    elif (v >= 3) & (v < 5):
        if RSI=='JOUR_RSI_fort':
            return 'B'
        elif RSI=='JOUR_RSI_modéré':
            return 'B-C'
        elif RSI=='JOUR_RSI_faible':
            return 'C'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'D'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'E'
    elif (v >= 5) & (v < 6):
        if RSI=='JOUR_RSI_fort':
            return 'C'
        elif RSI=='JOUR_RSI_modéré':
            return 'C-D'
        elif RSI=='JOUR_RSI_faible':
            return 'D'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'D'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'D'
    elif (v >= 6):
        if RSI=='JOUR_RSI_fort':
            return 'C'
        elif RSI=='JOUR_RSI_modéré':
            return 'D'
        elif RSI=='JOUR_RSI_faible':
            return 'D'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'D'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'D'
    else:
        print('Pasquill Stability error')
        return 'D'
    
def sigma(stability, x):
    """
    application restricted to downwind distance < 10km
    Parameters
    ----------
    stability : TYPE
        pasquill stability
    x : TYPE
        downwind distance

    Returns
    -------
    TYPE
        return σy and σz
        [[pasquill-gifford σy mode 1, np.nan], [pasquill-gifford σy mode 2, pasquill-gifford σz mode 2], [ASME79 σy mode 1, ASME79 σz mode 1],[Klug69 σy mode 1, Klug69 σz mode 1]]
    """
    empty = np.ones(x.shape)
    empty[:, :] = np.nan

    if stability == 'A':
        A = np.asarray([[0.443*x**0.894, empty],
                        [np.exp(-1.104+0.9878*np.log(x)-0.0076*np.log(x)**2) , np.exp(4.679-1.172*np.log(x)+0.227*np.log(x)**2)],
                        [0.4*x**0.91, 0.4*x**0.91],
                        [0.469*x**0.903, 0.017*x**1.38]])
        return A
    elif stability == 'A-B':
        A = np.asarray([[0.443*x**0.894, empty],
                        [np.exp(-1.104+0.9878*np.log(x)-0.0076*np.log(x)**2) , np.exp(4.679-1.172*np.log(x)+0.227*np.log(x)**2)],
                        [0.4*x**0.91, 0.4*x**0.91],
                        [0.469*x**0.903, 0.017*x**1.38]])
        B = np.asarray([[0.324*x**0.894, empty],
                        [np.exp(-1.634+1.035*np.log(x)-0.0096*np.log(x)**2) , np.exp(-1.999+0.8752*np.log(x)+0.0136*np.log(x)**2)],
                        [0.36*x**0.86, 0.33*x**0.86],
                        [0.306*x**0.885, 0.072*x**1.021]])
        return (A+B)/2
    elif stability == 'B':
        B = np.asarray([[0.324*x**0.894, empty],
                        [np.exp(-1.634+1.035*np.log(x)-0.0096*np.log(x)**2) , np.exp(-1.999+0.8752*np.log(x)+0.0136*np.log(x)**2)],
                        [0.36*x**0.86, 0.33*x**0.86],
                        [0.306*x**0.885, 0.072*x**1.021]])
        return B
    elif stability == 'B-C':
        C = np.asarray([[0.216*x**0.894, empty],
                        [np.exp(-2.054+1.0231*np.log(x)-0.0076*np.log(x)**2) , np.exp(-2.341+0.9477*np.log(x)-0.002*np.log(x)**2)],
                        [empty, empty],
                        [0.23*x**0.855, 0.076*x**0.879]])
        B = np.asarray([[0.324*x**0.894, empty],
                        [np.exp(-1.634+1.035*np.log(x)-0.0096*np.log(x)**2) , np.exp(-1.999+0.8752*np.log(x)+0.0136*np.log(x)**2)],
                        [0.36*x**0.86, 0.33*x**0.86],
                        [0.306*x**0.885, 0.072*x**1.021]])
        return (C+B)/2
    elif stability == 'C':
        C = np.asarray([[0.216*x**0.894, empty],
                        [np.exp(-2.054+1.0231*np.log(x)-0.0076*np.log(x)**2) , np.exp(-2.341+0.9477*np.log(x)-0.002*np.log(x)**2)],
                        [empty, empty],
                        [0.23*x**0.855, 0.076*x**0.879]])
        return C
    elif stability == 'C-D':
        C = np.asarray([[0.216*x**0.894, empty],
                        [np.exp(-2.054+1.0231*np.log(x)-0.0076*np.log(x)**2) , np.exp(-2.341+0.9477*np.log(x)-0.002*np.log(x)**2)],
                        [empty, empty],
                        [0.23*x**0.855, 0.076*x**0.879]])
        D = np.asarray([[0.141*x**0.894, empty],
                        [np.exp(-2.555+1.0423*np.log(x)-0.0087*np.log(x)**2) , np.exp(-3.186+1.1737*np.log(x)-0.0316*np.log(x)**2)],
                        [0.32*x**0.78, 0.22*x**0.78],
                        [0.219*x**0.764, 0.140*x**0.727]])
        return (C+D)/2
    elif stability == 'D':
        D = np.asarray([[0.141*x**0.894, empty],
                        [np.exp(-2.555+1.0423*np.log(x)-0.0087*np.log(x)**2) , np.exp(-3.186+1.1737*np.log(x)-0.0316*np.log(x)**2)],
                        [0.32*x**0.78, 0.22*x**0.78],
                        [0.219*x**0.764, 0.140*x**0.727]])
        return D
    elif stability == 'E':
        E = np.asarray([[0.105*x**0.894, empty],
                        [np.exp(-2.754+1.0106*np.log(x)-0.0064*np.log(x)**2) , np.exp(-3.783+1.301*np.log(x)-0.045*np.log(x)**2)],
                        [empty, empty],
                        [0.237*x**0.691, 0.217*x**0.61]])
        return E
    elif stability == 'F':
        F = np.asarray([[0.071*x**0.894, empty],
                        [np.exp(-3.143+1.0148*np.log(x)-0.007*np.log(x)**2) , np.exp(-4.49+1.4024*np.log(x)-0.054*np.log(x)**2)],
                        [0.31*x**0.71, 0.06*x**0.71],
                        [0.273*x**0.594, 0.262*x**0.5]])
        return F
    else:
        D = np.asarray([[0.141*x**0.894, empty],
                        [np.exp(-2.555+1.0423*np.log(x)-0.0087*np.log(x)**2) , np.exp(-3.186+1.1737*np.log(x)-0.0316*np.log(x)**2)],
                        [0.32*x**0.78, 0.22*x**0.78],
                        [0.219*x**0.764, 0.140*x**0.727]])
        return D

def plot_dispersion():
    global x, PG1, PG2, ASME79, Klug1969
    x = np.linspace(100, xmax, 1000)
    x = x[:, np.newaxis]
    A = sigma('A', x)
    AB = sigma('A-B', x)
    B = sigma('B', x)
    BC = sigma('B-C', x)
    C = sigma('C', x)
    CD = sigma('C-D', x)
    D = sigma('D', x)
    E = sigma('E', x)
    F = sigma('F', x)

    PG1 = st.checkbox("Pasquill & Gifford, mode 1", False)
    PG2 = st.checkbox("Pasquill & Gifford, mode 2", True)
    ASME79 = st.checkbox("ASME 1979, mode 1", False)
    Klug1969 = st.checkbox("Klug 1969, mode 1", False)

    #sigma y
    fig, ax = plt.subplots()
    if PG2:
        ax.plot(x, A[1, 0, :, 0], c='purple', label='A')
        ax.plot(x, B[1, 0, :, 0], c='navy', label='B')
        ax.plot(x, C[1, 0, :, 0], c='dodgerblue', label='C')
        ax.plot(x, D[1, 0, :, 0], c='lightseagreen', label='D')
        ax.plot(x, E[1, 0, :, 0], c='lightgreen', label='E')
        ax.plot(x, F[1, 0, :, 0], c='yellowgreen', label='F')
    if PG1:
        ax.plot(x, A[0, 0, :, 0], '--', c='purple')
        ax.plot(x, B[0, 0, :, 0], '--', c='navy')
        ax.plot(x, C[0, 0, :, 0], '--', c='dodgerblue')
        ax.plot(x, D[0, 0, :, 0], '--', c='lightseagreen')
        ax.plot(x, E[0, 0, :, 0], '--', c='lightgreen')
        ax.plot(x, F[0, 0, :, 0], '--', c='yellowgreen')
    if ASME79:
        ax.plot(x, A[2, 0, :, 0], '-.', c='purple')
        ax.plot(x, B[2, 0, :, 0], '-.', c='navy')
        ax.plot(x, C[2, 0, :, 0], '-.', c='dodgerblue')
        ax.plot(x, D[2, 0, :, 0], '-.', c='lightseagreen')
        ax.plot(x, E[2, 0, :, 0], '-.', c='lightgreen')
        ax.plot(x, F[2, 0, :, 0], '-.', c='yellowgreen')
    if Klug1969:
        ax.plot(x, A[3, 0, :, 0], ':', c='purple')
        ax.plot(x, B[3, 0, :, 0], ':', c='navy')
        ax.plot(x, C[3, 0, :, 0], ':', c='dodgerblue')
        ax.plot(x, D[3, 0, :, 0], ':', c='lightseagreen')
        ax.plot(x, E[3, 0, :, 0], ':', c='lightgreen')
        ax.plot(x, F[3, 0, :, 0], ':', c='yellowgreen')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    ax.legend()
    ax.set_xlabel("Distance à la cheminée (m)")
    ax.set_ylabel("Coefficient de dispersion dans le plan horizontal \n" + r"et perpendiculairement à la direction du vent ($\sigma _y$, m).")
    st.pyplot(fig)

    #sigma z
    fig2, ax2 = plt.subplots()
    if PG2:
        ax2.plot(x, A[1, 1, :, 0], c='purple', label='A')
        ax2.plot(x, B[1, 1, :, 0], c='navy', label='B')
        ax2.plot(x, C[1, 1, :, 0], c='dodgerblue', label='C')
        ax2.plot(x, D[1, 1, :, 0], c='lightseagreen', label='D')
        ax2.plot(x, E[1, 1, :, 0], c='lightgreen', label='E')
        ax2.plot(x, F[1, 1, :, 0], c='yellowgreen', label='F')
    if ASME79:
        ax2.plot(x, A[2, 1, :, 0], '-.', c='purple')
        ax2.plot(x, B[2, 1, :, 0], '-.', c='navy')
        ax2.plot(x, C[2, 1, :, 0], '-.', c='dodgerblue')
        ax2.plot(x, D[2, 1, :, 0], '-.', c='lightseagreen')
        ax2.plot(x, E[2, 1, :, 0], '-.', c='lightgreen')
        ax2.plot(x, F[2, 1, :, 0], '-.', c='yellowgreen')
    if Klug1969:
        ax2.plot(x, A[3, 1, :, 0], ':', c='purple')
        ax2.plot(x, B[3, 1, :, 0], ':', c='navy')
        ax2.plot(x, C[3, 1, :, 0], ':', c='dodgerblue')
        ax2.plot(x, D[3, 1, :, 0], ':', c='lightseagreen')
        ax2.plot(x, E[3, 1, :, 0], ':', c='lightgreen')
        ax2.plot(x, F[3, 1, :, 0], ':', c='yellowgreen')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(1, 10000)
    ax2.grid()
    ax2.legend()
    ax2.set_xlabel("Distance à la cheminée (m)")
    ax2.set_ylabel("Coefficient de dispersion dans le plan vertical \n" + r"et perpendiculairement à la direction du vent ($\sigma _z$, m).")
    st.pyplot(fig2)

def coupe_vertical():
    global SA, SA_climatique
    zmax = st.slider("Choisir l'altitude maximum à représenter :", value=1000, min_value=100, max_value=20000, step=100)
    z = np.linspace(0, zmax, 500)
    MCD = st.selectbox("Définir un modèle de coefficient de dispersion", ["Pasquill & Gifford, mode 2", "ASME 1979, mode 1", "Klug 1969, mode 1"], index=0)
    if MCD =="Pasquill & Gifford, mode 2":
        i=1
    elif MCD =="ASME 1979, mode 1":
        i = 2
    elif MCD =="Klug 1969, mode 1":
        i=3
    Xy = st.slider("Choisir la distance à la source de la coupe verticale perpendiculaire à la direction du vent", value=1000, min_value=100, max_value=10000, step=100)
    SA_climatique = stability_pasquill(v, RSI, HR, mode='24H')
    liste_SA = ['A', 'A-B', 'B', 'B-C', 'C', 'C-D', 'D', 'E', 'F']
    SA = st.selectbox("Redéfinir la condition de stabilité atmosphérique (la valeur par défault dépend des conditions météorologiques de la journée) : ", liste_SA, index=liste_SA.index(SA_climatique))
    X, Zx = np.meshgrid(x[:, 0], z)
    Y = 0
    surelevation = Δh_Briggs(X, Vs, v, d, Ts, Ta)
    sigma_val = sigma(SA, X)
    sigmay = sigma_val[i, 0, 0, :][np.newaxis, :]
    sigmaz = sigma_val[i, 1, 0, :][np.newaxis, :]
    newZ = Zx-19-surelevation
    C = (np.exp(-Y**2/(2*sigmay**2))*np.exp(-(newZ)**2/(2*sigmaz**2)))/(v*sigmay*sigmaz*2*np.pi)
    fig, ax = plt.subplots()
    f =ax.imshow(np.log10(C), extent=[X.min(), X.max(), Zx.min(), Zx.max()], origin='lower', vmin=-15, vmax=0, cmap='nipy_spectral', aspect=X.max()/(2*Zx.max()))
    plt.colorbar(f, ax=ax, orientation='horizontal').set_label(r'Facteur de dilution en $log_{10}$')
    ax.set_xlabel("Distance à la cheminée (m)")
    ax.set_ylabel("Altitude parallèlement \n à la direction du vent")
   

    y = np.arange(-2000, 2000, 4)
    Y, Zy = np.meshgrid(y, z)
    ax.plot([Xy, Xy], [Zy.min(), Zy.max()], c='w')
    Xy = np.asarray([[Xy]])
    surelevation = Δh_Briggs(Xy, Vs, v, d, Ts, Ta)
    sigma_val = sigma(SA, Xy)
    sigmay = sigma_val[i, 0, 0, :][np.newaxis, :]
    sigmaz = sigma_val[i, 1, 0, :][np.newaxis, :]
    newZ = Zy-19-surelevation
    C = (np.exp(-Y**2/(2*sigmay**2))*np.exp(-(newZ)**2/(2*sigmaz**2)))/(sigmay*sigmaz*v*2*np.pi)
    fig2,ax2 = plt.subplots()
    f2 = ax2.imshow(np.log10(C), extent=[Y.min(), Y.max(), Zy.min(), Zy.max()], origin='lower', vmin=-15, vmax=0, cmap='nipy_spectral', aspect=Y.max()/Zy.max())
    plt.colorbar(f2, ax=ax2, orientation='horizontal').set_label(r'Facteur de dilution en $log_{10}$')
    ax2.set_xlabel(f"Distance au centre du panache (m) à {Xy[0, 0]/1000} km du centre d'émission")
    ax2.set_ylabel("Altitude perpendiculairement \n à la direction du vent.")
    st.pyplot(fig)
    st.pyplot(fig2)

def collec_to_gdf(collec_poly):
    """Transform a `matplotlib.contour.QuadContourSet` to a GeoDataFrame"""
    polygons = []
    for i, path in enumerate(collec_poly._paths):
        mpoly = []
        path.should_simplify = False
        poly = path.to_polygons()
        # Each polygon should contain an exterior ring + maybe hole(s):
        exterior, holes = [], []
        if len(poly) > 0 and len(poly[0]) > 3:
            # The first of the list is the exterior ring :
            exterior = poly[0]
            # Other(s) are hole(s):
            if len(poly) > 1:
                holes = [h for h in poly[1:] if len(h) > 3]
        mpoly.append(Polygon(exterior, holes))
        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
    gpfile  =gpd.GeoDataFrame(geometry=polygons, crs='2154')
    return gpfile
    


def carte_stationnaire():
    vVent_mean = np.nanmean(vVent, axis=0)
    v = np.sqrt(np.sum(vVent_mean**2))

    #fig, ax = plt.subplots()
    #ax.scatter(vVent[:, 0], vVent[:, 1], c=np.sqrt(np.sum(vVent**2, axis=1)), cmap='jet')
    #ax.scatter(vVent_mean[0], vVent_mean[1], c='k')
    #ax.set_xlim(-6, 6)
    #ax.set_ylim(-6, 6)
    #st.pyplot(fig)

    C = np.zeros((ny, nx))
    Δh = Δh_Briggs(dist_XY, Vs, v, d, Ts, Ta)
    dot_product=X_*vVent_mean[0]+Y_*vVent_mean[1]
    magnitudes=v*dist_XY
    # angle entre la direction du vent et le point (x,y)
    subtended=np.arccos(dot_product/(magnitudes+1e-15));
    # distance le long de la direction du vent jusqu'à une ligne perpendiculaire qui intersecte x,y
    downwind=np.cos(subtended)*dist_XY
    filtre = np.where(downwind > 0)
    crosswind=np.sin(subtended)*dist_XY
    sr = sigma(SA_climatique, downwind)
    σy =sr[1, 0, filtre[0],filtre[1]]
    σz =sr[1, 1, filtre[0],filtre[1]]
    C[filtre[0], filtre[1]] = (np.exp(-crosswind[filtre[0],filtre[1]]**2./(2.*σy**2.))* np.exp(-(Z[filtre[0],filtre[1]] + Δh[filtre[0],filtre[1]])**2./(2.*σz**2.)))/(2.*np.pi*v*σy*σz)
    fig, ax = plt.subplots(figsize=(10, 10))
    contour_log10 = np.arange(-15.,-2.)
    contour = [10**i for i in contour_log10]
    ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
    im = ax.contour(C, contour, extent=extent, cmap='nipy_spectral', vmin=1E-15, vmax=1E-3, origin='lower', norm='log', zorder=1)
    ax.scatter(x0, y0, c='crimson', zorder=3)
    for n, l in villes.items():
        ax.scatter(l[0], l[1], c='w', zorder=2)
        ax.text(l[0], l[1], n, zorder=3, fontsize=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad="10%")
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Facteur de dilution de la concentration ; \n le code couleur ne représente pas des seuils sanitaires')
    st.pyplot(fig)
    #folium map 
    cmap = mpl.colormaps['nipy_spectral']
    norm = mpl.colors.LogNorm(vmin=1E-15, vmax=1E-2)
    cmap_list = [cmap(norm(i)) for i in contour]
    cmap_folium = cmp.LinearColormap(cmap_list, vmin=-15, vmax=-2, index=contour, caption='PCD')
    print(cmap_folium)
    gdfcontour = collec_to_gdf(im)
    gdfcontour['data'] = contour_log10
    gdfcontour = gdfcontour.to_crs('epsg:4326')  
    
    lon, lat = RGF93_to_WGS84.transform(xx=x0, yy=y0)        
    m = folium.Map(location=[lat, lon])
    IconCentrale = folium.Icon(icon="house", icon_color="black", color="black", prefix="fa")
    folium.Marker([lat, lon], popup="Centrale à bitume", tooltip="Centrale à bitume", icon=IconCentrale).add_to(m)
    filtre = [i.is_empty is False for i in gdfcontour.geometry]
    id_filtre = np.where(filtre)[0]
    folium.Choropleth(geo_data=gdfcontour.loc[id_filtre, 'geometry'],
                      data=gdfcontour.loc[id_filtre, 'data'],
                      line_weight=0.3,
                      fill_color='Paired',
                      fill_opacity = 0.1,
                      line_color='back').add_to(m)
    st_map = st_folium(m, use_container_width=True)
    






def carte_bouffee():
    vVent_mean = np.nanmean(vVent, axis=0)
    v = np.sqrt(np.sum(vVent_mean**2))

    #fig, ax = plt.subplots()
    #ax.scatter(vVent[:, 0], vVent[:, 1], c=np.sqrt(np.sum(vVent**2, axis=1)), cmap='jet')
    #ax.scatter(vVent_mean[0], vVent_mean[1], c='k')
    #ax.set_xlim(-6, 6)
    #ax.set_ylim(-6, 6)
    #st.pyplot(fig)

    C = np.zeros((ny, nx, 24))
    Δh = Δh_Briggs(dist_XY, Vs, v, d, Ts, Ta)
    dot_product=X_*vVent_mean[0]+Y_*vVent_mean[1]
    magnitudes=v*dist_XY
    # angle entre la direction du vent et le point (x,y)
    subtended=np.arccos(dot_product/(magnitudes+1e-15));
    # distance le long de la direction du vent jusqu'à une ligne perpendiculaire qui intersecte x,y
    downwind=np.cos(subtended)*dist_XY
    filtre = np.where(downwind > 0)
    crosswind=np.sin(subtended)*dist_XY
    sr = sigma(SA_climatique, downwind)
    σy =sr[1, 0, filtre[0],filtre[1]]
    σz =sr[1, 1, filtre[0],filtre[1]]
    C[filtre[0], filtre[1]] = (np.exp(-crosswind[filtre[0],filtre[1]]**2./(2.*σy**2.))* np.exp(-(Z[filtre[0],filtre[1]] + Δh[filtre[0],filtre[1]])**2./(2.*σz**2.)))/(2.*np.pi*v*σy*σz)
    fig, ax = plt.subplots(figsize=(10, 10))
    contour = np.arange(-15.,-2.)
    contour = [10**i for i in contour]
    ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
    im = ax.contour(C, contour, extent=extent, cmap='nipy_spectral', vmin=1E-15, vmax=1E-3, origin='lower', norm='log', zorder=1)
    ax.scatter(x0, y0, c='crimson', zorder=3)
    for n, l in villes.items():
        ax.scatter(l[0], l[1], c='w', zorder=2)
        ax.text(l[0], l[1], n, zorder=3, fontsize=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad="10%")
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Facteur de dilution de la concentration ; \n le code couleur ne représente pas des seuils sanitaires')
    st.pyplot(fig)

st.set_page_config(page_title="Le calcul et ses options", page_icon=":waning_gibbous_moon:")
st.markdown("# Le calcul et ses options")
st.sidebar.header("Paramètres")
st.markdown(
    """
    ## Principe du code partagé ici.

    <div style="text-align: justify;">
    Le principe de l'algorithme empirique utilisé ici et celui de la dispersion atmosphérique des panaches gaussiens.
    </div>
    
    """, unsafe_allow_html=True
)
st.image(image_DP, caption="Illustration d'une dispersion atmosphérique d'un panache de fumée.")
st.markdown(
    """  
    <div style="text-align: justify;">
    <p>
    Il a été construit sur la base du support de cours du professeur <a href="http://www.christianseigneur.fr/Accueil/">Christian Seigneur</a> du <a href="https://www.cerea-lab.fr">CEREA</a>, auquel il est possible de se référer pour plus d'informations (<a href="http://cerea.enpc.fr/fich/support_cours/SGE_M2_modelisation_2010-2011/SGE-Modelisation-Introduction.pdf">Introduction</a>, <a href="http://cerea.enpc.fr/fich/support_cours/SGE_M2_modelisation/SGE-Modelisation-Dynamique.pdf">Dynamique</a> et <a href="http://cerea.enpc.fr/fich/support_cours/SGE_M2_modelisation/SGE-Modelisation-Dispersion.pdf">Dispersion</a>). La page <a href="https://fr.wikipedia.org/wiki/Mod%C3%A9lisation_de_la_dispersion_atmosph%C3%A9rique">wikipedia</a> pose également quelques éléments de réponses instructives. Le lecteur intéressé (et anglophone) trouvera également la lecture de <a href="https://drive.google.com/file/d/1_LbkRy5sfpjzgBUT1e8S5dCJ5hFHkWdL/view?usp=sharing">cet ouvrage</a> intéressant.
    </p>

    <p>
    Des modèles HYSPLIT <a href="https://www.ready.noaa.gov/hypub-bin/dispasrc.pl">peuvent-être lancés en ligne</a> (site en anglais) pour comparaison. 
    </p>

    <p>
    En résumé, l'idée est d'évaluer la dispersion/dilution d'un volume volatil émis par une cheminée en fonction des conditions atmosphériques (voir l'image ci-dessous). Plus le volume s'éloigne de sa source d'émission et plus ce volume est dilué. La morphologie de cette dilution répond le plus souvent à une <a href="https://fr.wikipedia.org/wiki/Fonction_gaussienne">loi gaussienne</a>, dont les paramètres (écart-types dans des directions perpendiculaires au vent, horizontale -y-  et verticale -z-) sont définis par les conditions météorologiques.
    </p>
    </div>

    """, unsafe_allow_html=True
)

st.image(image_DP2, caption="Diagramme représentant une dispersion gaussienne d'un panache de fumée. Wikipedia common.")

st.markdown(
    """
    On distingue souvent :
    <ol>
    <li>les modèles de panache gaussien stationnaire</li>
    <li>les modèles de bouffées (qui peuvent-être non stationnaire)</li>

    ## Les éléments structurants.

    ### Généralités et paramètres clefs.
    <p>
    Nous allons exposés ci-après les principaux éléments de connaissances nécessaires à la résolution de ce type d'équation empirique (c'est à dire, contrainte par des observations récurrentes):
    </p>
    <ol>
    <li>des données météorologiques : 
        <ul> 
        <li> température, </li>
        <li> vitesse du vent, </li>
        <li> direction du vent </li>
        <li> stabilité atmosphérique </li>
        <li> pression (optionnel) </li>
        </ul>
    </li>
    <li>des données topographiques (forme du terrain)</li>
    <li>les caractéristiques de la source d'émission :
        <ul> 
        <li> température à l'émission</li>
        <li> diamètre de la cheminée</li>
        <li> vitesse d'émission à la cheminée</li>
        <li> concentration des molécules et poussières</li>
        </ul>
    </li>
    </ol>

    <p> En plus de la quantification des écart-types, ces données permettent également de calculer une hauteur de soulèvement du panache au dessus de l'évent de la cheminée.
    
    ### Les limites de la méthode utilisée
    <p>
    Dans certains cas, le modèle gaussien possède une capacité de prédiction plus ou moins limité:
    </p>
    <ol>
    <li>dans des contextes géomorphologiques particuliers :
        <ul> 
        <li> montagnes </li>
        <li> canyons </li>
        <li> ... </li>
        </ul>
        </li>
    <li>dans des contextes de forte rugosité :
        <ul> 
        <li> forte variation de végétation </li>
        <li> constructions humaines </li>
        <li> ... </li>
        </ul>
    </li>
    <li>dans des conditions météorologiques particulières :
        <ul> 
        <li> inversion atmosphérique </li>
        <li> brume </li>
        <li> turbulence </li>
        <li> ... </li>
        </ul>
    </li>
    <li>dans des contextes d'albédo (part du rayonnement solaire renvoyé dans l'atmosphère) particulier</li>
    <li>si l'on souhaite évaluer le comportement d'une molécule: la réactivité et le poids de la molécule en question</li>
    </ol>
    <p> Pour palier ces limites, des modifications plus ou moins complexes doivent être mise en place. Le code que nous présentons ici ne permet pas de résoudre ces complexités.</p>
    
    <p> Nous n'utilisons pas à ce stade les réflexions sur le sol et les réflexions sous une couche d'inversion de température.

    ### Description des paramètres clefs

    #### Les données météorologiques
    <p>
    Les données météorologiques passées sont présentées dans une <a href='StationMeteoPuylaurens' target='_self'>page dédiée</a>.
    </p>
    <p>
    Il est également prévu l'intégration des prévisions météorologiques de Puylaurens.
    </p>

    #### Les données topographiques
    Les données topographiques utilisées sont celles de la BD ALTI au pas de 75 m. Elles sont représentées sur la figure suivante :

    """, unsafe_allow_html=True
)

topographie()

st.markdown(
    """
    Si l'on zoom dans un périmètre restreint autour de la centrale, voici le rendu :

    """
)

topographie_zoom()

st.markdown("""
            
    ### Les données sur l'émission de la centrale à bitume
    <div style="text-align: justify;">
    <p>
    Les données sur les produits émis par la centrale à bitume sont présentées dans une <a href='EmissionsCentrale' target='_self'>page dédiée</a>.
    </p>
    <p>
    Nous retiendrons ici les paramètres suivants pour la centrale de Puylaurens (RF500), tel que mentionné dans les documents techniques (pièce E6) :
    </p>
    </div>
    <ol>
    <li> Le diamètre de la cheminée est le 1.35 m.</li>
    <li> La température à la sortie de la cheminée est de 110°C.</li>
    <li> La vitesse réglementaire minimum en sortie de cheminée est de 8 m.s<sup>-1</sup> soit un débit minimum de 41 224 m<sup>3</sup>.h<sup>-1</sup> </li>
    <li> Le débit technique maximum est de 120 633 m<sup>3</sup>.h<sup>-1</sup>, soit une vitesse réglementaire maximum en sortie de cheminée de 23.4 m.s<sup>-1</sup></li>
    <li> Un débit moyen de 71 433 +/- 8 067 m<sup>3</sup>.h<sup>-1</sup>, et une vitesse moyenne en sortie de cheminée de 13.9 +/- 1.4 m.s<sup>-1</sup> sont indiquées dans <a href="https://drive.google.com/file/d/10J062gaUUuA9CHmDnayOKdIv6I0haijt/view?usp=sharing">les caractéristiques fournies par la SPIE Batignolles-Malet via ATOSCA</a>.</li>
    </ol>
    """
    , unsafe_allow_html=True
)

st.markdown("""
        ## Analyse des effets du modèle de surélévation du panache et de la dispersion en fonction de la stabilité atmosphérique.
        
        ### Les modèles de surélévation du panache
        <div style="text-align: justify;">    
        La surélévation du panache correspond à l'écart entre l'altitude de la bouche d'émission et le centre du panache. Nous avons mis en oeuvre les quatres modèles présentés par Christian Seigneur, à savoir:
            <ol>
            <li> Le modèle de Holland </li>
            <li> Le modèle de Carson et Moses </li>
            <li> Le modèle de Concawes </li>
            <li> Le modèle de Briggs </li>
            </ol>
        Il est possible de voir la réponse des différents modèles ci-après, en fonction des conditions météo et des paramètres d'émission.
         </div>
            """
    , unsafe_allow_html=True
)       

surelevation()

st.markdown("""
        <div style="text-align: justify;">    
        D'après de professeur Seigneur, le modèle de Briggs est le plus utilisé dans les modèles de panaches gaussiens. C'est également celui-ci que nous utiliserons. A noter que c'est le seul à intégrer une variation en fonction de la distance à la cheminée. 
        </div>
            
        ### Les modèles de dispersion en fonction des conditions atmosphériques.
        <div style="text-align: justify;">
        Les coefficients de dispersion correspondent à l'écart-type de la <a href="https://fr.wikipedia.org/wiki/Fonction_gaussienne">loi gaussienne</a>. Ces coefficients sont principalement déterminés par la distance à la source et par les conditions de stabilité atmosphérique, eux même déterminé par l'insolation, la vitesse du vent et la nébulosité (voir la classification de Pasquill ci-dessous).
        </div>    
        """
    , unsafe_allow_html=True
)

st.image(image_DP3, caption="Stabilité atmosphérique : classification de Pasquill. D'après Turner, 1970. \n A: très instable ; B : instable ; C : peu instable ; D: neutre ; E : stable ; F : très stable.")

st.markdown("""
        <div style="text-align: justify;">     
        Sur la base des informations climatiques, nous définissons les classes de Pasquill comme suit:
        </div>  
        <ol>
            <li>Si nous cherchons à voir le comportement moyen sur 24 heures : </li>
            <ul>
            <li>
            Nous considérons que l'insolation solaire est faible ('DAY'>'SLIGHT') si sa moyenne est inférieure à 90 W.m<sup>2</sup>
            </li>
            <li>
            Nous considérons que l'insolation solaire est forte ('DAY'>'STRONG') si sa moyenne est supérieur à 264 W.m<sup>2</sup>
            </li>
            <li>
            Nous considérons que l'insolation solaire est moyenne ('DAY'>'MODERATE') sinon
            </li>
            </ul>
            <li>Si nous cherchons à voir le comportement précis sur une journée (pas inférieur à l'heure), afin de considéré précisément les effets climatiques sur les périodes de fonctionnement de la centrale: </li>
            <ul>
            <li>
            Nous considérons que l'insolation solaire est faible ('DAY'>'SLIGHT') si sa moyenne est inférieure à 199 W.m<sup>2</sup>
            </li>
            <li>
            Nous considérons que l'insolation solaire est forte ('DAY'>'STRONG') si sa moyenne est supérieur à 472 W.m<sup>2</sup>
            </li>
            <li>
            Nous considérons que la nébulosité (couverture nuageuse) est forte ('NIGHT'>' =< 3/8 CLOUD') si l'humidité relative moyenne est supérieur à 60%
            </li>
            <li>
            Nous considérons que la nébulosité est faible ('NIGHT'>'Thinly Overcast or >= 4/8 LOW CLOUD') si l'humidité relative moyenne est inférieure à 60%
            </li>
            <li>
            Nous considérons que l'insolation solaire est moyenne ('DAY'>'MODERATE') sinon
            </li>
            </ul>            
        </ol>
        <div style="text-align: justify;">     
        Si nous n'avons pas les informations nécessaires nous utilisons la classe 'D', tel que conseillé par Turner, 1970.
        </div>        
         
        <p>
        </p>
        """
    , unsafe_allow_html=True
)



st.markdown("""
        <div style="text-align: justify;">     
        Il existe différentes équations pour décrire l'évolution de ces coefficients de dispersion en fonction des facteurs de contrôle. Quatres de ces équations sont présentées ci-après.
        </div>   
        <p>
        </p>
        """
    , unsafe_allow_html=True
)

plot_dispersion()

st.markdown("""
        <div style="text-align: justify;">    
        Afin de bien comprendre ce qu'implique ces coefficients, nous allons représenter la dilution du volume émis en fonction de la distance à la source. Pour cela, nous ferons une représentation de ce volume 3D à travers deux coupes: 
        <ol>
            <li> une coupe verticale, parallèle à la direction du vent </li>
            <li> une coupe verticale, perpendiculaire à la direction du vent</li>
        </ol>
        </div>   
        <p>
        </p>
        """
    , unsafe_allow_html=True
)

coupe_vertical()

st.markdown("""
            <p>
            Voici quelques clefs de lecture pour appréhender ces graphiques :
            <ol>  
            <li>Dans les modèles ci dessus, l'altitude 0 est celle du pied de la cheminée d'émission. A ce stade, aucune topographie n'y est représenté. </li>
            <li>De même, il ne figure aucune variation en fonction du temps. Les variables (température, vitesse et direction du vent...) sont supposés stationnaire à l'échelle d'une journée. </li>
            <li>Le code colorimétrique reflète le degrés de dilution de ce qui a été émis. Ainsi une couleur où le log10 de la concentration vaut:
            <ul>
            <li> -3 (rouge brique) signifique que dans 1 m<sup>3</sup> de cette zone, il y a un millième de la masse émise à la source (ie par la cheminée) en une seconde. </li>
            <li> -6 (vert) signifique que dans 1 m<sup>3</sup> de cette zone, il y a un millionième de la masse émise à la source en une seconde. </li>
            <li> -9 (bleu) signifique que dans 1 m<sup>3</sup> de cette zone, il y a un milliardième de la masse émise à la source en une seconde. </li>
            </ul>
            </li>
            </ol>
            </p>
            """
    , unsafe_allow_html=True
)

st.markdown("""
        ## Calcul stationnaire ou par bouffée ?
    
        <div style="text-align: justify;">    
        Comme nous l'avons déjà mentionné, deux manières de calculer sont possibles:
            <li> En utilisant un modèle de panache gaussien stationnaire </li>
            <li> En utilisant un modèle par bouffées </li>

        Dans cette partie nous allons comparer les deux résultats sur la journée sélectionnée. Cette comparaison n'est possible que grace aux données météorologiques aux pas de 5 minutes que nous avons pu récupérer.
        </div>   
            
        ### Le calcul stationnaire
            
        Le calcul stationnaire repose sur les valeurs météorologiques moyennes de la journée. Elle suppose leur variation faible.
        """
    , unsafe_allow_html=True
)

carte_stationnaire()

st.markdown("""
        ### Le calcul par bouffée
            
        Le calcul stationnaire repose sur les valeurs météorologiques moyennes de la journée. Elle suppose leur variation faible.
        """
    , unsafe_allow_html=True
)