# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK, pointsToVTK
import meshio
from scipy.stats import norm, beta
from matplotlib.animation import FuncAnimation, PillowWriter
import geopandas as gpd

###############################################################################
#
#              Infrastructure
#
###############################################################################
routes = gpd.read_file('route.shp')

batiments = gpd.read_file('batiments.shp')

villes = {'Revel': [619399.490,6262672.707],
         'Puylaurens':[620266.862,6275241.681],
         'St Germain des près': [624500.311,6274095.737],
         'Soual':[628528.524,6273191.962],
         'Lempaut':[624346.252,6270432.622],
         'Sémalens':[628217.241,6277447.042],
         'Vielmur Sur Agout':[626542.116,6280461.738],
         'St Paul Cap de Joux':[617341.329,6283743.400],
         'Villeneuve lès lavaur':[602022.138,6278485.487],
         'Lavaur':[604890.514,6289623.373],
         'Dicosa':[630705.899,6276426.029],
         'Saix':[634096.773,6276222.376],
         'Dourgne':[630321.279,6265464.647],
         'Escoussens':[636303.189,6267049.024],
         'Sorèze':[624423.786,6261910.745],
         'Lautrec':[630493.276,6289921.993],
         'Graulhet':[618445.042,6296272.167],
         'Blan':[619758.761,6270229.387]}

ecoles  = {'Jacques Durand Puylaurens': [619630.028,6274924.484],
         'La Source Puylaurens':[619897.520,6275309.631],
         "Jeanne D'Arc Puylaurens": [620347.561,6275357.115],
         'Ecole Publique St Germain des Près':[624420.092,6274105.652],
         "Ecole primaire de Lempaut":[624341.217,6270443.592],
         "Ecole primaire de Lescout":[627436.372,6271514.616],
         "Ecole de Soual":[628475.741,6273446.154],
         'Ecole Publique de Sémalens':[628475.741,6273446.154]}

###############################################################################
#
#              Sortie cheminée
#
###############################################################################

x0, y0, z0 = 623208.070, 6273468.332, 230-5+18.56

diam_chem = 1.35 #m
t_sortie_chem = 110 #°C

vitesse_min = 8 #m/s
debit_min = np.pi*(1.35/2)**2*8*3600 #m3/h

debit_max = 120633 #m3/h
vitesse_max = debit_max/(np.pi*(1.35/2)**2*3600) #m/s

vitesse = np.linspace(0, 1, 30)*(vitesse_max-vitesse_min) + vitesse_min
debit = np.linspace(0, 1, 30)*(debit_max-debit_min) + debit_min
output_chimney_probability = beta(2, 2).pdf(np.linspace(0, 1, 30))

#issu de SPIE batignole
vitesse_moyenne = np.asarray([13.9, 1.4]) #m/s mean, std
débit_moyen = np.asarray([71433, 8067]) #m3/h mean, std
#plt.plot(debit, vitesse_p)

#|||||||||||||||||||||| surélévation du panache   |||||||||||||||||||||||||||||

def Δh_Holland(Vs, v, d, Pa, Ts, Ta):
    """

    Parameters
    ----------
    Pa : TYPE
        pression atmospherique en Pa
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
    TYPE
        DESCRIPTION.

    """
    return Vs*d*(1.5+2.68*1E3*Pa*d*(Ts-Ta)/Ts)/v

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

###############################################################################
#
#              MNT 75m
#
###############################################################################

alt = pd.read_csv('BDALT.csv', header=None)
filtre = (alt.loc[:, 0] < 640000) & (alt.loc[:, 1] > 6.255*1E6)
alt = alt.loc[filtre, :]
vx, vy = np.unique(alt.loc[:, 0]), np.unique(alt.loc[:, 1])
nx, ny = len(vx), len(vy)
Z = np.zeros((ny, nx))
idY, idX = (alt.loc[:, 1]-vy.min())/75, (alt.loc[:, 0]-vx.min())/75
Z[idY.to_numpy(dtype=int), idX.to_numpy(dtype=int)] = alt.loc[:, 2]
X, Y = np.meshgrid(vx, vy)

X, Y, Z = X-x0, Y-y0, Z-z0
dist_XY = np.sqrt(X**2+Y**2)
dist_XYZ = np.sqrt(X**2+Y**2+Z**2)

#plt.imshow(Z, origin='lower')
#plt.colorbar()


###############################################################################
#
#              Météo
#
###############################################################################


meteo = pd.read_csv('meteo_puylaurens.csv', sep=';', skiprows=3)
meteo['DAY'] = pd.to_datetime(meteo['DAY'], format='%d/%m/%y')

vent_vitesse_ms = meteo.loc[:, 'AVG_WIND_SPEED'].to_numpy()/(1E-3*3600)
dir_vent = meteo['DOM_DIR'].to_numpy()
u_vent_unit = np.asarray([np.sin(dir_vent*np.pi/180), np.cos(dir_vent*np.pi/180)]).T*-1
u_vent = u_vent_unit*vent_vitesse_ms[:, np.newaxis]

T = meteo.loc[:, 'T_MEAN'].to_numpy()
rain = meteo.loc[:, 'RAIN'].to_numpy()
humidity = meteo.loc[:, 'Humidite_Moy [%]'].to_numpy()
Pmax = meteo.loc[:, 'Pression_Max [hPa]'].to_numpy()
Pmin = meteo.loc[:, 'Pression_Min [hPa]'].to_numpy()
dP = Pmax-Pmin
# plt.subplot(131)
# plt.scatter(T, Pmax, c=rain, s=0.5*humidity, alpha=0.5, cmap='jet', vmax=10)
# plt.colorbar(orientation='horizontal').set_label('Pluie (mm) \n taille symbole = humidité'  )
# plt.xlabel('Température (°C)')
# plt.ylabel('Pression maximum (hPa)')
# plt.subplot(132)
# plt.scatter(T, Pmax, c=dP, s=0.5*humidity, alpha=0.5, cmap='jet')
# plt.colorbar(orientation='horizontal').set_label('Amplitude maximum de variation de pression (hPa) \n taille symbole = humidité')
# plt.xlabel('Température (°C)')
# plt.ylabel('Pression maximum (hPa)')
# plt.subplot(133)
# plt.scatter(dP, humidity, c=vent_vitesse_ms, s=T, alpha=0.5, cmap='jet', vmax=3)
# plt.colorbar(orientation='horizontal').set_label('Vitesse du vent (m/s) \n taille symbole = Températureé' )
# plt.xlabel('Amplitude maximum de variation de pression (hPa)')
# plt.ylabel('Humidité (%)')
# plt.tight_layout()

def  stability_pasquill(v, mode='JOUR_RSI_fort'):
    """
    Parameters
    ----------
    v : TYPE
        wind speed at 10m , m/s
    mode : TYPE, optional
        RSI= rayonnement solaire incident
        'JOUR_RSI_fort', 'JOUR_RSI_modéré', 'JOUR_RSI_faible', 'NUIT_NEBULOSITE_4/8-7/8', 'NUIT_NEBULOSITE_<3/8'

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
    if v < 2:
        if mode=='JOUR_RSI_fort':
            return 'A'
        elif mode=='JOUR_RSI_modéré':
            return 'A-B'
        elif mode=='JOUR_RSI_faible':
            return 'B'
        elif mode=='NUIT_NEBULOSITE_4/8-7/8':
            return 'F'
        elif mode== 'NUIT_NEBULOSITE_<3/8':
            return 'F'
        else:
            print('mode error')
    elif (v >= 2) & (v < 3):
        if mode=='JOUR_RSI_fort':
            return 'A-B'
        elif mode=='JOUR_RSI_modéré':
            return 'B'
        elif mode=='JOUR_RSI_faible':
            return 'C'
        elif mode=='NUIT_NEBULOSITE_4/8-7/8':
            return 'E'
        elif mode== 'NUIT_NEBULOSITE_<3/8':
            return 'F'
        else:
            print('mode error')
    elif (v >= 3) & (v < 5):
        if mode=='JOUR_RSI_fort':
            return 'B'
        elif mode=='JOUR_RSI_modéré':
            return 'B-C'
        elif mode=='JOUR_RSI_faible':
            return 'C'
        elif mode=='NUIT_NEBULOSITE_4/8-7/8':
            return 'D'
        elif mode== 'NUIT_NEBULOSITE_<3/8':
            return 'E'
        else:
            print('mode error')
    elif (v >= 5) & (v < 6):
        if mode=='JOUR_RSI_fort':
            return 'C'
        elif mode=='JOUR_RSI_modéré':
            return 'C-D'
        elif mode=='JOUR_RSI_faible':
            return 'D'
        elif mode=='NUIT_NEBULOSITE_4/8-7/8':
            return 'D'
        elif mode== 'NUIT_NEBULOSITE_<3/8':
            return 'D'
        else:
            print('mode error')
    elif (v >= 6):
        if mode=='JOUR_RSI_fort':
            return 'C'
        elif mode=='JOUR_RSI_modéré':
            return 'D'
        elif mode=='JOUR_RSI_faible':
            return 'D'
        elif mode=='NUIT_NEBULOSITE_4/8-7/8':
            return 'D'
        elif mode== 'NUIT_NEBULOSITE_<3/8':
            return 'D'
        else:
            print('mode error')
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
                        [np.exp(-1.104+0.9878*np.log(x)-0.0076*np.log(x)**2) , np.exp(4.679-1.7172*np.log(x)+0.227*np.log(x)**2)],
                        [0.4*x**0.91, 0.4*x**0.91],
                        [0.469*x**0.903, 0.017*x**1.38]])
        return A
    elif stability == 'A-B':
        A = np.asarray([[0.443*x**0.894, empty],
                        [np.exp(-1.104+0.9878*np.log(x)-0.0076*np.log(x)**2) , np.exp(4.679-1.7172*np.log(x)+0.227*np.log(x)**2)],
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
        return 'D'

stability = [stability_pasquill(v, mode='JOUR_RSI_faible') for v in vent_vitesse_ms]

Δh = np.zeros((ny, nx, len(meteo)))
C = np.zeros((ny, nx, len(meteo)))
sigma_rec = []

for i in range(len(meteo)):
    v_i = vent_vitesse_ms[i]
    
    Δh[:, :, i] = Δh_Briggs(dist_XY, vitesse_moyenne[0], v_i, diam_chem, t_sortie_chem, T[i])
    
    # Need angle between point x, y and the wind direction, so use scalar product:
    dot_product=X*u_vent[i, 0]+Y*u_vent[i, 1]
    # product of magnitude of vectors:
    magnitudes=v_i*dist_XY

    # angle between wind and point (x,y)
    subtended=np.arccos(dot_product/(magnitudes+1e-15));
    # distance along the wind direction to perpendilcular line that intesects x,y
    downwind=np.cos(subtended)*dist_XY
    filtre = np.where(downwind > 0)
    crosswind=np.sin(subtended)*dist_XY
    
    sigma_rec.append(sigma(stability[i], downwind))
    
    sr = sigma_rec[i]

# =============================================================================
#     plt.subplot(421)
#     plt.imshow(sr[0, 0, :, :], origin='lower', vmin=10, vmax=2E3, cmap='nipy_spectral')
#     plt.colorbar()
#     plt.subplot(423)
#     plt.imshow(sr[1, 0, :, :], origin='lower', vmin=10, vmax=2E3,  cmap='nipy_spectral')
#     plt.colorbar()
#     plt.subplot(424)
#     plt.imshow(sr[1, 1, :, :], origin='lower', vmin=10, vmax=2E3, cmap='nipy_spectral')
#     plt.colorbar()
#     plt.subplot(425)
#     plt.imshow(sr[2, 0, :, :], origin='lower', vmin=10, vmax=2E3,  cmap='nipy_spectral')
#     plt.colorbar()
#     plt.subplot(426)
#     plt.imshow(sr[2, 1, :, :], origin='lower', vmin=10, vmax=2E3, cmap='nipy_spectral')
#     plt.colorbar()
#     plt.subplot(427)
#     plt.imshow(sr[3, 0, :, :], origin='lower', vmin=10, vmax=2E3,  cmap='nipy_spectral')
#     plt.colorbar()
#     plt.subplot(428)
#     plt.imshow(sr[3, 1, :, :], origin='lower', vmin=10, vmax=2E3, cmap='nipy_spectral')
#     plt.colorbar()
# =============================================================================

    σy =sr[1, 0, filtre[0],filtre[1]]
    σz =sr[1, 1, filtre[0],filtre[1]]
    C[filtre[0], filtre[1], i] = 1/(2.*np.pi*v_i*σy*σz)* \
    np.exp(-crosswind[filtre[0],filtre[1]]**2./(2.*σy**2.))* np.exp(-(Z[filtre[0],filtre[1]] + Δh[filtre[0],filtre[1], i])**2./(2.*σz**2.))

#%%
contour = np.arange(-15.,-4.)
contour = [10**i for i in contour]
extent = [X.min()+x0, X.max()+x0, Y.min()+y0, Y.max()+y0]

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
C[C==0] = np.nan
try:
    if np.all(np.isnan(C[:, :, 0])):
        print('all nan')
    else:
        print('not all nan')
        im = ax.contour(C[:, :, 0], contour, extent=extent, cmap='nipy_spectral', origin='lower', norm='log', zorder=1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_yticklabels(["{:.0e}".format(i) for i in contour])
        cbar.set_label('Facteur de dilution de la concentration ; le code couleur ne représente pas des seuils sanitaires')
except:
    print('bad contour')
plt.suptitle("2nd test d'évaluation de la dispersion \n du panache de la centrale à enrobée ; (jour 0)")
# for n, l in villes.items():
#     plt.scatter(l[0], l[1], c='w', zorder=2)
#     plt.text(l[0], l[1], n, zorder=3)

plt.xlim((extent[0], extent[1]))
plt.ylim((extent[2], extent[3]))

def animate(i):
    ax.clear()
    ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
    try:
        if np.all(np.isnan(C[:, :, i])):
            print('all nan')
        else:
            print('not all nan')
            ax.contour(C[:, :, i], contour, extent=extent, cmap='nipy_spectral', origin='lower', norm='log', zorder=1)
    except:
        print('bad contour')
    plt.suptitle(f"2nd test d'évaluation de la dispersion \n du panache de la centrale à enrobée ; (jour {i})")
    for n, l in villes.items():
        plt.scatter(l[0], l[1], c='w', zorder=2)
        plt.text(l[0], l[1], n, zorder=3)

ani = FuncAnimation(fig, animate, interval=500, frames=len(meteo))    
ani.save("REJ_ATM.gif", dpi=150, writer=PillowWriter(fps=5))


contour = np.arange(-7.,-2, 0.5)
contour = [10**i for i in contour]
extent = [X.min()+x0, X.max()+x0, Y.min()+y0, Y.max()+y0]
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(Z, extent=extent, cmap='terrain', origin='lower', zorder=0)
im = ax.contour(np.nansum(C[:, :, :], axis=-1), contour, extent=extent, cmap='jet', origin='lower', norm='log', zorder=1)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.set_yticklabels(["{:.0e}".format(i) for i in contour])
cbar.set_label("Facteur d'exposition au panache de la centrale à enrobée ; le code couleur ne représente pas des seuils sanitaires")
plt.suptitle("2nd test d'évaluation de l'exposition au panache de la centrale à enrobée")
for n, l in villes.items():
    plt.scatter(l[0], l[1], c='w', zorder=2)
    plt.text(l[0], l[1], n, zorder=3)
for n, l in ecoles.items():
    plt.scatter(l[0], l[1], marker='+', c='r', zorder=2)
routes.plot(ax=ax, color='k', alpha=0.2)
plt.xlim(extent[0], extent[1])
plt.ylim(extent[2], extent[3])
plt.tight_layout()

dx = vx[1]-vx[0]
dy = vy[1]-vy[0]
idxs, idxe = 250, 350
idys, idye = 225, 275
extent = [X.min()+x0+idxs*dx, X.max()+x0-(len(vx)-idxe)*dx, Y.min()+y0+idys*dy, Y.max()+y0-(len(vy)-idye)*dy]
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.imshow(Z[idys:idye, idxs:idxe], extent=extent, cmap='terrain', origin='lower', zorder=0)
im = ax.contour(np.nansum(C[idys:idye, idxs:idxe, :], axis=-1), contour, extent=extent, cmap='jet', origin='lower', norm='log', zorder=1)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.set_yticklabels(["{:.0e}".format(i) for i in contour])
cbar.set_label("Facteur d'exposition au panache de la centrale à enrobée ; le code couleur ne représente pas des seuils sanitaires")
plt.suptitle("2nd test d'évaluation de l'exposition au panache de la centrale à enrobée")
plt.xlim(extent[0], extent[1])
plt.ylim(extent[2], extent[3])
routes.plot(ax=ax, color='k', alpha=0.2, zorder=2)
batiments.plot(ax=ax, color='k', zorder=3)
plt.tight_layout()

plt.figure()
plt.imshow(C[:, :, 95], origin='lower')
