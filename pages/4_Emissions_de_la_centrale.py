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
from streamlit.hello.utils import show_code
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

data_CAREPS = pd.read_csv('./DATA/DATA_CAREPS.csv', sep=';')

st.set_page_config(layout="wide",page_title="Emissions", page_icon="")

def CAREPS():
    st.markdown("""## Les données du CAREPS
                
                Les données de l'étude du CAREPS ont été numérisées. Nous présentons ici des histogrammes de distribution des principales qualités des émissions.""", unsafe_allow_html=True)
    

    humidité = data_CAREPS['Humidité (%)'].to_numpy()
    fig, ax =  plt.subplots(1, 2, figsize=(5, 2))
    ax[0].hist(humidité, color="k", bins=30)
    ax[0].set_title("Humidité (%)")
    ax[1].boxplot(humidité[~np.isnan(humidité)])
    st.pyplot(fig)

    O2 = data_CAREPS['O2 (%)'].to_numpy()
    fig, ax =  plt.subplots(1, 2, figsize=(5, 2))
    ax[0].hist(O2, color="k", bins=30)
    ax[0].set_title("O2 (%)")
    ax[1].boxplot(O2[~np.isnan(O2)])
    st.pyplot(fig)

    T = data_CAREPS['T° moyenne (°C)'].to_numpy()
    fig, ax =  plt.subplots(1, 2, figsize=(5, 2))
    ax[0].hist(T, color="k", bins=30)
    ax[0].set_title("Température moyenne (°C)")
    ax[1].boxplot(T[~np.isnan(T)])
    st.pyplot(fig)

    v = data_CAREPS['v éjection gaz (m/s)'].to_numpy()
    fig, ax =  plt.subplots(1, 2, figsize=(5, 2))
    ax[0].hist(v, color="k", bins=30)
    ax[0].set_title("Vitesse d'émission (m/s)")
    ax[1].boxplot(v[~np.isnan(v)])
    st.pyplot(fig)

    fig, ax =  plt.subplots(figsize=(5, 2))
    f=ax.scatter(v, T, c=O2, s=humidité, cmap='jet')
    plt.colorbar(f, ax=ax).set_label('O2 (%)')
    ax.set_xlabel("Vitesse d'émission (m/s)")
    ax.set_ylabel('Température (°C)')
    st.pyplot(fig)

    st.write(data_CAREPS.to_html(escape=False, index=False), unsafe_allow_html=True)

def main():


    TableParticule = pd.DataFrame(
        [
            {"Désignation": "Dioxyde de carbone",
             "Symbole":'CO2',
             "Type": "gaz" ,
             "Quantité émise [kg/h]": "5000",
             "Dangerosité": "?",
             "Symptômes": "?",
             'Fiche complète': "<a target='_blank' href='https://www.inrs.fr/dms/ficheTox/FicheFicheTox/FICHETOX_238-1/FicheTox_238.pdf'>fiche INRS - Dioxyde de carbone</a>"},
            {"Désignation": "particule",
             "Symbole":'TBD',
             "Type": 'TBD',
             "Quantité émise [kg/h]": 'TBD',
             "Dangerosité": 'TBD',
             "Symptômes": 'TBD',
             'Fiche complète': "<a target='_blank' href=''></a>"},
        ]
    )

    st.write(TableParticule.to_html(escape=False, index=False), unsafe_allow_html=True)


#st.set_page_config(page_title="Les données", page_icon="")
st.markdown("# Particules émises par la centrale à bitume")
st.markdown(
    """
    Cette page présente les particules émises par les centrales à bitume.
    
    """
)

CAREPS()
