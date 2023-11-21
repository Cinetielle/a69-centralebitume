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

import streamlit as st
from streamlit.logger import get_logger
from PIL import Image

LOGGER = get_logger(__name__)

image_DP = Image.open('./im/logo-puylaurens-citoyens-deroutes-color-fond-blanc.png')

def run():
    st.set_page_config(
        page_title="Usines à bitume - A69",
        page_icon="👋",
    )
    st.image(image_DP)

    st.write("# Etudes d'impacts des centrales d'enrobage à bitume de l'A69")

    st.sidebar.success("Choississez une étude ci dessus.")

    st.markdown(
        """
        L'objet de ce site web est de quantifier objectivement et en totale transprence les impacts des émissions volatiles des cheminées des centrales d'enrobage à chaud prévu sur le site de Puylaurens. 

        Ce site est composé comme suit:

        <ol>
        <li> Une page décrivant les données météorologiques de Puylaurens, préalable indispensable aux modélisations, (en chantier)</li>
        <li> Une page décrivant le calcul effectué et les options disponibles, (en chantier)</li>
        <li> Une page décrivant les produits émis par les centrales à bitume à chaud et leurs potentiels impact sur la santé, (à construire) </li>
        <li> Une page décrivant l'exposition cartographique aux produits émis par la centrale en fonction des conditions météo passées, (à construire)</li>
        <li> Une page décrivant l'exposition de votre domicile aux produits émis par la centrale en fonction des conditions météo passées, (à construire)</li>
        <li> Une page décrivant l'exposition cartographique aux produits émis par la centrale en fonction des conditions météo futures, (à construire)</li>
        <li> Une page décrivant l'exposition de votre domicile aux produits émis par la centrale en fonction des conditions météo futures. (à construire)</li>
        </ol>

        Ce site est réalisé par Citoyens déroutés, un collectif du territoire Puylaurentais qui souhaite s'informer, informer et protéger.
    """, unsafe_allow_html=True
    )    
        
if __name__ == "__main__":
    run()
