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
import datetime
import matplotlib.pyplot as plt

meteo = pd.read_csv('./DATA/METEO/meteo_puylaurens.csv', sep=';', skiprows=3)
date = pd.to_datetime(meteo.iloc[:, 0], format="%d/%m/%y")

def data_explore() -> None:

    # set time series
    start_date = st.sidebar.date_input('Début', date[0]+datetime.timedelta(days=5))
    end_date = st.sidebar.date_input('Fin', date[len(date)-1])

    filtre = (date>= pd.to_datetime(start_date)) & (date<= pd.to_datetime(end_date))
    meteo_slice = meteo[filtre]
    header = meteo.columns[1:]
    st.dataframe(meteo_slice)

    to_plot = st.sidebar.selectbox("Which data to display ?", header)

    fig, ax = plt.subplots()
    ax.plot(date[filtre], meteo_slice[to_plot], c='k') 
    ax.set_xlabel("Date")
    ax.set_ylabel(to_plot)
    st.pyplot(fig)

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


st.set_page_config(page_title="Les données", page_icon="📹")
st.markdown("# Les données")
st.sidebar.header("Les données")
st.markdown(
    """
    Cette page permet d'explorer et de configurer les données d'entrées.
    
    La météo : historique enregistré à cette [station](https://puylaurens.payrastre.fr).
    """
)

data_explore()
