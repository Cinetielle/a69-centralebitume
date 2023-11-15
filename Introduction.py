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

image_DP = Image.open('./im/Gaussian_Plume_fr.png')

def run():
    st.set_page_config(
        page_title="Usines à bitume - A69",
        page_icon="👋",
    )

    st.write("# Etudes d'impacts des centrales d'enrobage à bitume de l'A69")

    st.sidebar.success("Choississez une étude ci dessus.")

    st.markdown(
        """
        L'objet de ce site web est de quantifier objectivement les impacts des émissions volatiles des cheminées des centrales d'enrobage à chaud prévu sur les sites de Puylaurens et Villeneuve-lès-lavaur. 
    
        Des informations complémentaires peuvent-être trouvées  [en suivant ce lien](https://fr.wikipedia.org/wiki/Mod%C3%A9lisation_de_la_dispersion_atmosph%C3%A9rique).
    
        Le type de modèle utilisé ici est le suivant :
    """
    )

    st.image(image_DP, caption="Diagramme représentant une dispersion gaussienne d'un panache de fumée. Wikipedia common.")


if __name__ == "__main__":
    run()
