#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:20:43 2023

@author: m_a_m
"""
import pandas as pd
import numpy as np
#en µg/m3

pièce_E6_NOx = {'hiver 2015':{'fond rural':[20.3, 16.4, 10.4,  9.3],
                              'proximité routière':[31.6, 20.9, 9.1, 11.4, 17.9],
                              'fond urbain':[13.7]},
                'automne 2015':{'fond rural':[16.8, 17.7, 12.9, 10.1, 8.3, 5.2, 5.5, 6.2, 5.1, 4.7, 5.2, 6.6, 3.9, 6.5, 7.2, 8, 4.3, 6.5, 7.4, 7.7, 8.2, 5.8, 7.9],
                           'proximité routière':[32.2, 20.3, 10.8, 11.7, 17.6, 24.2, 21.2, 23.7, 16.1, 29.9, 13.4, 18.6, 13.3, 42.1],
                           'fond urbain':[13.3, 6, 8.1, 13, 6.8, 6.5, 9.6, 6.1, 6.3, 13.4, 13.6, 13.1]},
                'hiver 2016':{'fond rural':[12.3, 8.8, 7.9, 7.9, 5.3, 6.6, 6, 0.5, 4.5, 8, 8, 9.1, 1.1, 9.9, 10.3, 5.7, 9.9],
                              'proximité routière':[30.2, 23.1, 22.3, 17.3, 26.8, 13.6, 8.6, 21.6, 15.2, 41.5],
                              'fond urbain':[7.8, 10.3, 13.5, 9.1, 9.5, 13.2, 13.2, 7.7, 8.9, 14.4, 15.7, 17.5]}}

pièce_E6_Benzène = {'hiver 2015':{'fond rural':[1.1, 1.2],
                             'proximité routière':[1.1, 1.2, 1.5],
                             'fond urbain':[]},
                    'automne 2015':{'fond rural':[0.5, 0.5, 0.5, 0.4, 0.5, 0.6],
                                    'proximité routière':[0.5, 0.6, 0.7, 0.5, 0.5, 0.9],
                                    'fond urbain':[0.6, 0.4, 0.4, 0.6]},
                    'hiver 2016':{'fond rural':[0.8, 0.6, 0.6, 0.9],
                                  'proximité routière':[0.7, 0.7, 1.2, 1.3],
                                  'fond urbain':[0.8, 0.8, 1.1]}}

SOx_Background = pd.DataFrame([["Moyenne FR 2021 en fond urbain", 2, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-le-dioxyde-de-soufre-so2"],
                               ["Moyenne FR 2021 en fond industriel", 2.3, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-le-dioxyde-de-soufre-so2"]
                               ])

NOx_Background = pd.DataFrame([["Moyenne FR 2022 en fond urbain", 15, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-le-dioxyde-dazote-no2"],
                               ["Moyenne FR 2022 en proximité routière", 27, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-le-dioxyde-dazote-no2"],
                               ["Moyenne locale hiver 2015 fond rural", np.mean(pièce_E6_NOx['hiver 2015']['fond rural']), "µg/m3", "pièce E6, dossier ATOSCA"],
                               ["Moyenne locale hiver 2015 proximité routière", np.mean(pièce_E6_NOx['hiver 2015']['proximité routière']), "µg/m3", "pièce E6, dossier ATOSCA"],
                               ["Moyenne locale automne 2015 fond rural", np.mean(pièce_E6_NOx['automne 2015']['fond rural']), "µg/m3", "pièce E6, dossier ATOSCA"],
                               ["Moyenne locale automne 2015 proximité routière", np.mean(pièce_E6_NOx['automne 2015']['proximité routière']), "µg/m3", "pièce E6, dossier ATOSCA"],
                               ["Moyenne locale automne 2015 fond urbain", np.mean(pièce_E6_NOx['automne 2015']['fond urbain']), "µg/m3", "pièce E6, dossier ATOSCA"],
                               ["Moyenne locale hiver 2016 fond rural", np.mean(pièce_E6_NOx['hiver 2016']['fond rural']), "µg/m3", "pièce E6, dossier ATOSCA"],
                               ["Moyenne locale hiver 2016 proximité routière", np.mean(pièce_E6_NOx['hiver 2016']['proximité routière']), "µg/m3", "pièce E6, dossier ATOSCA"],
                               ["Moyenne locale hiver 2016 fond urbain", np.mean(pièce_E6_NOx['hiver 2016']['fond urbain']), "µg/m3", "pièce E6, dossier ATOSCA"],
                               ])

Benzène_Background = pd.DataFrame([["Moyenne FR 2021 en fond urbain", 0.73, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-le-benzene-c6h6"],
                                   ["Moyenne FR 2021 en proximité routière", 1.1, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-le-benzene-c6h6"],
                                   ["Moyenne FR 2021 en fond industriel", 1.4, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-le-benzene-c6h6"],
                                   ["Moyenne locale hiver 2015 fond rural", np.mean(pièce_E6_Benzène['hiver 2015']['fond rural']), "µg/m3", "pièce E6, dossier ATOSCA"],
                                   ["Moyenne locale hiver 2015 proximité routière", np.mean(pièce_E6_Benzène['hiver 2015']['proximité routière']), "µg/m3", "pièce E6, dossier ATOSCA"],
                                   ["Moyenne locale automne 2015 fond rural", np.mean(pièce_E6_Benzène['automne 2015']['fond rural']), "µg/m3", "pièce E6, dossier ATOSCA"],
                                   ["Moyenne locale automne 2015 proximité routière", np.mean(pièce_E6_Benzène['automne 2015']['proximité routière']), "µg/m3", "pièce E6, dossier ATOSCA"],
                                   ["Moyenne locale automne 2015 fond urbain", np.mean(pièce_E6_Benzène['automne 2015']['fond urbain']), "µg/m3", "pièce E6, dossier ATOSCA"],
                                   ["Moyenne locale hiver 2016 fond rural", np.mean(pièce_E6_Benzène['hiver 2016']['fond rural']), "µg/m3", "pièce E6, dossier ATOSCA"],
                                   ["Moyenne locale hiver 2016 proximité routière", np.mean(pièce_E6_Benzène['hiver 2016']['proximité routière']), "µg/m3", "pièce E6, dossier ATOSCA"],
                                   ["Moyenne locale hiver 2016 fond urbain", np.mean(pièce_E6_Benzène['hiver 2016']['fond urbain']), "µg/m3", "pièce E6, dossier ATOSCA"],
                                   ])

PM_Background = pd.DataFrame([["Moyenne PM2.5 FR 2021 en fond urbain", 9.6, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-les-particules-de-diametre-inferieur-ou-egal-25-micrometres-pm25"],
                              ["Moyenne PM10 FR 2021 en proximité routière", 20, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-les-particules-de-diametre-inferieur-ou-egal-10-micrometres-pm10"],
                              ["Moyenne PM10 FR 2021 en fond urbain", 15, "µg/m3", "https://www.statistiques.developpement-durable.gouv.fr/la-pollution-de-lair-par-les-particules-de-diametre-inferieur-ou-egal-10-micrometres-pm10"]
                              ])

