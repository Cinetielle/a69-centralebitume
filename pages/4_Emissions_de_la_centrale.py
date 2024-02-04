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
from utils import normalize, show_code

st.set_page_config(layout="wide",page_title="Emissions", page_icon="")

def CAREPS(data_CAREPS):
    st.markdown("""## Les données du CAREPS
                
                Les données de l'étude du CAREPS ont été numérisées. Nous présentons ici des histogrammes de distribution des principales qualités des émissions.""", unsafe_allow_html=True)
    run = st.checkbox("Voir les données")
    if run :
        humidité = data_CAREPS['Humidité (%)'].to_numpy()
        fig, ax =  plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(humidité, color="k", bins=30)
        ax[0].set_title("Humidité (%)")
        ax[1].boxplot(humidité[~np.isnan(humidité)])
        st.pyplot(fig)
    
        O2 = data_CAREPS['O2 (%)'].to_numpy()
        fig, ax =  plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(O2, color="k", bins=30)
        ax[0].set_title("O2 (%)")
        ax[1].boxplot(O2[~np.isnan(O2)])
        st.pyplot(fig)
    
        T = data_CAREPS['T° moyenne (°C)'].to_numpy()
        fig, ax =  plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(T, color="k", bins=20)
        ax[0].set_title("Température moyenne (°C)")
        ax[1].boxplot(T[~np.isnan(T)])
        st.pyplot(fig)
    
        v = data_CAREPS['v éjection gaz (m/s)'].to_numpy()
        fig, ax =  plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(v, color="k", bins=30)
        ax[0].set_title("Vitesse d'émission (m/s)")
        ax[1].boxplot(v[~np.isnan(v)])
        st.pyplot(fig)
    
        fig, ax =  plt.subplots(figsize=(10, 4))
        f=ax.scatter(v, T, c=O2, s=humidité, cmap='jet')
        plt.colorbar(f, ax=ax).set_label("O2 (%) \n (la taille reflète l'humidité)")
        ax.set_xlabel("Vitesse d'émission (m/s)")
        ax.set_ylabel('Température (°C)')
        st.pyplot(fig)
        
        RegBru = data_CAREPS['régime tube et brûleur (%)'].to_numpy()
        fig, ax =  plt.subplots(figsize=(10, 4))
        f=ax.scatter(O2, RegBru, c=T, s=humidité, cmap='jet')
        plt.colorbar(f, ax=ax).set_label("Température (°C)\n (la taille reflète l'humidité)")
        ax.set_xlabel('O2 (%) ')
        ax.set_ylabel('Régime des tubes et bruleurs (%)')
        st.pyplot(fig)
        
        DebEnr = data_CAREPS['débit des enrobés (t/h)'].to_numpy()/data_CAREPS['débit nominal (t/h)'].to_numpy()
        fig, ax =  plt.subplots(figsize=(10, 4))
        f=ax.scatter(DebEnr, RegBru, c=O2, s=humidité, cmap='jet')
        plt.colorbar(f, ax=ax).set_label("O2 (%) \n (la taille reflète l'humidité)")
        ax.set_xlabel('débit des enrobés (t/h) /  débit nominal de la centrale (t/h)')
        ax.set_ylabel('Régime des tubes et bruleurs (%)')
        st.pyplot(fig)
    
        DebGazNorm = data_CAREPS['débit gaz (Nm3/h)'].to_numpy()
        fig, ax =  plt.subplots(figsize=(10, 4))
        f=ax.scatter(DebGazNorm, v, c=O2, s=humidité, cmap='jet')
        plt.colorbar(f, ax=ax).set_label("O2 (%) \n (la taille reflète l'humidité)")
        ax.set_xlabel('débit gaz normalisé (Nm3/h)')
        ax.set_ylabel("Vitesse d'émission (m/s)")
        st.pyplot(fig)
        
        DebGazNorm = data_CAREPS['débit gaz (Nm3/h)'].to_numpy()
        fig, ax =  plt.subplots(figsize=(10, 4))
        f=ax.scatter(DebGazNorm, v, c=T, s=humidité, cmap='jet')
        plt.colorbar(f, ax=ax).set_label("Température (°C) \n (la taille reflète l'humidité)")
        ax.set_xlabel('débit gaz normalisé (Nm3/h)')
        ax.set_ylabel("Vitesse d'émission (m/s)")
        st.pyplot(fig)
        
        fig, ax =  plt.subplots(figsize=(10, 4))
        f=ax.scatter(DebGazNorm, T, c=O2, s=humidité, cmap='jet')
        plt.colorbar(f, ax=ax).set_label("O2 (%) \n (la taille reflète l'humidité)")
        ax.set_xlabel('débit gaz normalisé (Nm3/h)')
        ax.set_ylabel("Température (°C) ")
        st.pyplot(fig)
    
        st.write(data_CAREPS.to_html(escape=False, index=False), unsafe_allow_html=True)

def Normalisation():
    st.markdown("""## La normalisation
                
                Les concentrations réglementaires sont données pour des valeurs de débits normalisés. Nous allons détailler ici le calcul de normalisation.""", unsafe_allow_html=True)
    run = st.checkbox("Voir les calculs de normalisation")
    if run :        
        show_code(normalize)



#st.set_page_config(page_title="Les données", page_icon="")
st.markdown("# Particules émises par la centrale à bitume")
st.markdown(
    """
    Cette page présente les paramètres d'émission des centrales à bitume.
    
    """
)

@st.cache_data
def read_careps_data():
    data_CAREPS = pd.read_csv('./DATA/DATA_CAREPS.csv', sep=';')
    return data_CAREPS

data_CAREPS = read_careps_data()

CAREPS(data_CAREPS)

Normalisation()
