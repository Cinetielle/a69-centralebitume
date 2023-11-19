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
from PIL import Image

image_DP = Image.open('./im/E9F7Q18WEAc7P8_.jpeg')
image_DP2 = Image.open('./im/Gaussian_Plume_fr.png')
image_DP3 = Image.open('./im/Turner1970.png')

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

meteo = pd.read_csv('./DATA/METEO/meteo_puylaurens.csv', sep=';', skiprows=3)
date = pd.to_datetime(meteo.iloc[:, 0], format="%d/%m/%y")

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
    global Vs, v, d, Ts, Ta, xmax, Qh
    date_meteo_increment =  st.sidebar.slider("Choisir rapidement une nouvelle journée après le premier août 2021", value=0, min_value=0, max_value=768, step=1)
    date_meteo = st.sidebar.date_input("Choisir la météo d'une journée particulière", pd.to_datetime('2021/08/01')+datetime.timedelta(days=date_meteo_increment))
    filtre = (date== pd.to_datetime(date_meteo)+datetime.timedelta(days=date_meteo_increment))
    meteo_slice = meteo[filtre]
    header = meteo.columns[1:]
    Vs = st.sidebar.slider(r"Choisir la vitesse ($m.s^{-1}$) des gaz en sortie de cheminée ", value=13.9, min_value=8., max_value=23.4, step=0.1)
    xmax = st.sidebar.slider(r"Choisir la distance maximale à évaluer", value=5000, min_value=1000, max_value=20000, step=10)
    d = 1.35
    v = meteo_slice[header[19]].iloc[0]/3.6 # vitesse du vent en m/s
    Pa = (meteo_slice['Pression_Min [hPa]'].iloc[0]+meteo_slice['Pression_Max [hPa]'].iloc[0])/2 # pression atmosphérique en Pa
    Ta = meteo_slice['Temp_Moy '].iloc[0] # température de l'air en °C
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
    x = np.arange(100, xmax, 10)
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
    global SA
    z = np.arange(0, 1000, 2)
    MCD = st.selectbox("Définir un modèle de coefficient de dispersion", ["Pasquill & Gifford, mode 2", "ASME 1979, mode 1", "Klug 1969, mode 1"], index=0)
    if MCD =="Pasquill & Gifford, mode 2":
        i=1
    elif MCD =="ASME 1979, mode 1":
        i = 2
    elif MCD =="Klug 1969, mode 1":
        i=3
    Xy = st.slider("Choisir la distance à la source de la coupe verticale perpendiculaire à la direction du vent", value=1000, min_value=100, max_value=10000, step=100)
    SA = st.selectbox("Définir la condition de stabilité atmosphérique", ['A', 'A-B', 'B', 'B-C', 'C', 'C-D', 'D', 'E', 'F'], index=6)
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
   

    y = np.arange(-2000, 2000, 2)
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




st.set_page_config(page_title="Le calcul et ses options", page_icon="📈")
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
    Il a été construit sur la base du support de cours (<a href="http://cerea.enpc.fr/fich/support_cours/SGE_M2_modelisation_2010-2011/SGE-Modelisation-Introduction.pdf">Introduction</a>, <a href="http://cerea.enpc.fr/fich/support_cours/SGE_M2_modelisation/SGE-Modelisation-Dynamique.pdf">Dynamique</a> et <a href="http://cerea.enpc.fr/fich/support_cours/SGE_M2_modelisation/SGE-Modelisation-Dispersion.pdf">Dispersion</a> ) du professeur <a href="http://www.christianseigneur.fr/Accueil/">Christian Seigneur</a> du <a href="https://www.cerea-lab.fr">CEREA</a>, auquel il est possible de se référer pour plus d'informations. La page <a href="https://fr.wikipedia.org/wiki/Mod%C3%A9lisation_de_la_dispersion_atmosph%C3%A9rique">wikipedia</a> pose également quelques éléments de réponses instructives. Le lecteur intéressé (et anglophone) trouvera également la lecture de <a href="https://drive.google.com/file/d/1_LbkRy5sfpjzgBUT1e8S5dCJ5hFHkWdL/view?usp=sharing">cet ouvrage</a> intéressant.
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
    Nous retiendrons qu'il existe des données moyennes journalières et des données avec un pas de 5 minutes.
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
            <li> -3 (rouge brique) signifique que dans 1 m<sup>3</sup> de cette zone, il y a un millième de la masse émise à la source en une seconde (ie par la cheminée). </li>
            <li> -6 (vert) signifique que dans 1 m<sup>3</sup> de cette zone, il y a un millionième de la masse émise à la source en une seconde (ie par la cheminée). </li>
            <li> -9 (bleu) signifique que dans 1 m<sup>3</sup> de cette zone, il y a un milliardième de la masse émise à la source en une seconde (ie par la cheminée). </li>
            </ul>
            </li>
            </ol>
            </p>
            """
    , unsafe_allow_html=True
)