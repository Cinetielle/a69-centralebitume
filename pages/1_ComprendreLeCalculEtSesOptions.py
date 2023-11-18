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

image_DP = Image.open('./im/E9F7Q18WEAc7P8_.jpeg')
image_DP2 = Image.open('./im/Gaussian_Plume_fr.png')

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
    Il a été construit sur la base du <a href="http://cerea.enpc.fr/fich/support_cours/SGE_M2_modelisation_2010-2011/SGE-Modelisation-Introduction.pdf">support de cours</a> du professeur <a href="http://www.christianseigneur.fr/Accueil/">Christian Seigneur</a> du <a href="https://www.cerea-lab.fr">CEREA</a>, auquel il est possible de se référer pour plus d'informations. La page <a href="https://fr.wikipedia.org/wiki/Mod%C3%A9lisation_de_la_dispersion_atmosph%C3%A9rique">wikipedia</a> pose également quelques éléments de réponses instructives.
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
    Les données météorologique passées sont présentées dans une <a href='StationMeteoPuylaurens' target='_self'>page dédiée</a>.
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
    <li> Un débit moyen de 71 433 +/- 8 067 m<sup>3</sup>.h<sup>-1</sup>, et une vitesse moyenne en sortie de cheminée de 13.9 +/- 1.4 m.s<sup>-1</sup> sont indiquée dans <a href="https://drive.google.com/file/d/10J062gaUUuA9CHmDnayOKdIv6I0haijt/view?usp=sharing">les caractéristiques fournies par la SPIE Batignolles-Malet via ATOSCA</a>.</li>
    </ol>
    """
    , unsafe_allow_html=True
)

# show_code(plotting_demo)
