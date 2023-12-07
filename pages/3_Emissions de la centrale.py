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
import streamlit as st
import pandas as pd
from streamlit.hello.utils import show_code

st.set_page_config(layout="wide",page_title="Emissions", page_icon="")

def data_explore() -> None:


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
    Cette page permet présentes les particules émises par la centrale à bitume de Puylaurens.
    
    """
)
data_explore()
