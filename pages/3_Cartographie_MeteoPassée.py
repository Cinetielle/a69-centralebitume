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

import time

import numpy as np
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code

import matplotlib.pyplot as plt
from PIL import Image

#coordonnée de la sortie de cheminée
#-5 mètre pour intégrer le décaissement
x0, y0, z0 = 623208.070, 6273468.332, 230-5+18.56

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
dist_XYZ = np.sqrt(X_**2+Y_**2+Z_**2)
extent = [X.min(), X.max(), Y.min(), Y.max()]

meteo = pd.read_csv('./DATA/METEO/Donnees_meteo_Puylaurens.csv', sep=';', encoding='UTF-8')
meteo.index = pd.to_datetime(meteo.iloc[:, :5])


# pour moyenner des valeurs circulaires (ie 359 ° est plus proche de 1° que de 270°)
def vonmises_pdf(x, mu=0, kappa=10):
    return np.exp(kappa * np.cos(x - mu)) / (2. * np.pi * scipy.special.i0(kappa))
 
def vonmises_fft_kde(data, mu=0, bmin=-np.pi, bmax=np.pi, kappa=50, n_bins=360):
    bins = np.linspace(bmin, bmax, n_bins + 1, endpoint=True)
    hist_n, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = np.mean([bin_edges[1:], bin_edges[:-1]], axis=0)
    kernel = vonmises_pdf(x=bin_centers, mu=mu, kappa=kappa)
    kde = np.fft.fftshift(np.fft.irfft(np.fft.rfft(kernel) * np.fft.rfft(hist_n)))
    kde /= np.trapz(kde, x=bin_centers)
    kde /= np.sum(kde)
    if np.any(np.isnan(kde)):
        print('error 1')
    return [bin_centers, kde]

def mean_orientation(orientation):
    orientation = orientation[~np.isnan(orientation)]
    orientation[orientation > 180] = 360-orientation[orientation > 180]
    oradians = np.radians(orientation)
    bins, kde = vonmises_fft_kde(oradians)
    return bins*kde

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
    global Vs, v, d, Ts, Ta, xmax, Qh, RSI, HR
    date_meteo_increment =  st.sidebar.slider("Choisir rapidement une nouvelle journée après le 6 mars 2021", value=0, min_value=0, max_value=768, step=1)
    date_meteo = st.sidebar.date_input("Choisir la météo d'une journée particulière", pd.to_datetime('2021/03/06')+datetime.timedelta(days=date_meteo_increment))
    debut_jour = pd.to_datetime(date_meteo)+datetime.timedelta(days=date_meteo_increment)
    fin_jour = pd.to_datetime(date_meteo)+datetime.timedelta(days=date_meteo_increment+1)

    filtre = (meteo.index >= debut_jour) & (meteo.index <= fin_jour)
    meteo_slice = meteo.iloc[filtre, [5, 6, 7, 8, 10, 11, 12, 13, 14]]
    xmax = st.sidebar.slider(r"Choisir la distance maximale où évaluer les impacts", value=5000, min_value=1000, max_value=50000, step=10)
    Vs = st.sidebar.slider(r"Choisir la vitesse ($m.s^{-1}$) des gaz en sortie de cheminée ", value=13.9, min_value=8., max_value=23.4, step=0.1)

    d = 1.35
    print()
    v = meteo_slice.iloc[:, 3].mean()/3.6 # vitesse du vent en m/s
    Pa = meteo_slice.iloc[:, 4].mean()  # pression atmosphérique en Pa
    Ta = meteo_slice.iloc[:, 0].mean() # température de l'air en °C
    RSI = meteo_slice.iloc[:, 7].mean()  # insolation solaire moyenne sur 24H
    HR = meteo_slice.iloc[:, 2].mean() # Humidité moyenne sur 24H

    Ts = st.sidebar.slider(r"Choisir la température en sortie de cheminée", value=110, min_value=80, max_value=150, step=1)
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
    

st.set_page_config(page_title="Représentation cartographique", page_icon=":earth_africa:")
st.markdown("""
            # Représentation cartographique

            ## Introduction

            Nous disposons donc d'un <a href='ComprendreLeCalculEtSesOptions' target='_self'>outil de calcul</a> de la dispersion des fumées émises par les centrales à bitume.

            Nous commencerons par présenter des cartes de dilution des produits volatils émis par la centrale (quels qu'ils soient) en fonction des conditions météorologiques.

            Nous quantifierons ensuite l'exposition aux molécules spécifiques émises par la centrale. Il existe différents moyen de quantifier l'exposition aux molécules émises :
            <ol>
            <li> Nous pouvons calculer la concentration instantanée dans l'air d'un composé</li>
            <li> Nous pouvons calculer l'exposition moyenne à un composé sur une plage temporelle</li>
            <li> Nous pouvons calculer l'exposition cumulée à un composé sur une plage temporelle</li>
            </ol>

            A cette concentration, il conviendra d'ajouter la concentration de l'air ambiant. Ce que nous ferons dans une dernière partie.

            """, unsafe_allow_html=True)

st.markdown("""
            ## Cartographie de l'exposition, tout composés volatils émis par la centrale confondus

            """, unsafe_allow_html=True)

st.markdown("""
            ## Cartographie de l'exposition par composé volatil émis par la centrale

            """, unsafe_allow_html=True)

st.markdown("""
            ## Cartographie de l'exposition totale par composé volatil

            """, unsafe_allow_html=True)

st.sidebar.header("Paramètres")