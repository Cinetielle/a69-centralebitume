#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:00:24 2023

@author: m_a_m
"""
import pandas as pd
from DATA.FOND_DE_L_AIR import NOx_Background, SOx_Background, Benzène_Background, PM_Background

SOx_VTR = pd.DataFrame([["Niveau critique pour la protection des écosystèmes", 20, "µg/m3", "UE", "en moyenne annuelle et en moyenne sur la période du 1er octobre au 31 mars"],
                       ["Objectif de qualité", 50, "µg/m3", "FR", "en moyenne annuelle"],
                       ["Valeurs limites pour la protection de la santé humaine", 125, "µg/m3", "UE", "en moyenne journalière à ne pas dépasser plus de 3 jours par an"],
                       ["Seuil d’information et de recommandation", 300, "µg/m3", "", "en moyenne horaire"],
                       ["Valeurs limites pour la protection de la santé humaine", 350, "µg/m3", "UE", "en moyenne horaire à ne pas dépasser plus de 24 heures par an"],
                       ["Seuil d’alerte", 500, "µg/m3", "", "en moyenne horaire pendant 3 heures consécutives"]])

NOx_VTR = pd.DataFrame([["Niveau critique pour la protection de la végétation (NOx)", 30, "µg/m3", "UE", "en moyenne annuelle d’oxydes d’azote"],
                       ["Objectif de qualité", 40, "µg/m3", "FR", "en moyenne annuelle"],
                       ["Valeurs limites pour la protection de la santé humaine", 200, "µg/m3", "UE", "en moyenne horaire à ne pas dépasser plus de 18 heures par an"],
                       ["Seuil d’information et de recommandation", 200, "µg/m3", "FR", "en moyenne horaire"],
                       ["Seuil d’alerte", 400, "µg/m3", "UE", "en moyenne horaire pendant 3 heures consécutives"]])

PM2_5_VTR = pd.DataFrame([["Valeurs limites", 25, "µg/m3", "FR", "en moyenne annuelle"],
                       ["Objectifs de qualité", 10, "µg/m3", "FR", "en moyenne annuelle"]])

CO_VTR = pd.DataFrame([["Valeurs limites", 10E3, "µg/m3", "FR", "Maximum journalier de la moyenne sur 8 heures"]])

As_VTR = pd.DataFrame([["critères nationaux de qualité de l'air", 6E-3, "µg/m3", "FR", "en moyenne annuelle"]])

Pb_VTR = pd.DataFrame([["critères nationaux de qualité de l'air", 0.25, "µg/m3", "FR", "en moyenne annuelle"],
                       ["Valeurs limites", 0.5, "µg/m3", "FR", "en moyenne annuelle"]])

Cd_VTR = pd.DataFrame([["Valeurs cibles", 5E-3, "µg/m3", "FR", "en moyenne annuelle"]])

Ni_VTR = pd.DataFrame([["Valeurs cibles", 20E-3, "µg/m3", "FR", "en moyenne annuelle"]])

BenzoAPyrène_VTR = pd.DataFrame([["Valeurs cibles", 1E-3, "µg/m3", "FR", "en moyenne annuelle"]])

Benzène_VTR = pd.DataFrame([["critères nationaux de qualité de l'air", 2, "µg/m3", "FR", "en moyenne annuelle"],
                            ["valeur limite", 5, "µg/m3", "FR", "en moyenne annuelle"]])

# par µg/m3
ERU_anses = {"As":[[1.5E-4],
                   ["cancérogène sans seuil"]],
             "Benzene (71-43-2)":[[2.6E-5],
                                  ["cancérogène sans seuil"]],
             "Benzo[a]pyrène (BaP)":[[1.1E-3],
                                     ["cancérogène sans seuil"]],
             "Chrome (Cr)":[['ERU'],
                            [4E-2],
                            ["cancérogène sans seuil, Cr VI"]],
             "Nickel (Ni)":[[1.7E-4],
                   ["cancérogène sans seuil"]],
             "napthalène":[[5.6E-6],
                           ["cancérogène sans seuil"]],
             "Particules dans  l'air ambiant PM2,5":[[1.28E-2],
                                                    ["long terme sans seuil"]],
             "Particules dans l'air ambiant PM10":[[1.28e-2],
                                                 ["Long terme sans seuil"]]}

#en µg/m3
VTR_seuils_anses = {"Acétaldéhyde":[[3e2],
                                    ["subchronique (8h)"]],

                    "Acroléine":[[6.9, 4.4E-1, 1.5E-1],
                                 ["aiguë", "subchronique", "chronique à seuil"]],

                    "Cd":[[4.5E-1, 3E-1],
                          ["chronique à seuil", "cancérogène à seuil"]],

                    "Chloroforme":[[6.3E1],
                                   ["chronique à seuil"]],

                    "Chrome (Cr)":[["Valeur toxicologique de référence", "Valeur toxicologique de référence"],
                                   [5, 3E-1],
                                   ["µg/m3","µg/m3"],
                                   ["subchronique, Cr II-III", "subchronique, Cr VI"]], # chrome VI 6 à 80 fois plus faible que Cr total

                    "Formaldéhyde":[["Valeur toxicologique de référence", "Valeur toxicologique de référence"],
                                    [1.23E2, 1.23E2],
                                    ["µg/m3","µg/m3"],
                                    ["aiguë", "chronique à seuil"]],

                    "Mn":[[3E-1],
                          ["chronique à seuil"]],

                    "CO":[["Valeur toxicologique de référence", "Valeur toxicologique de référence", "Valeur toxicologique de référence"],
                          [1E2*1E3, 3E1*1E3, 1E1*1E3],
                          ["µg/m3","µg/m3", "µg/m3"],
                          ["aigue (15 min)", "aigue (1 heure)", "chronique à seuil (8 heures)"]],

                    "napthalène":[[3.7E1],
                                  ["chronique à seuil"]],

                    "Nickel (Ni)":[[2.3E-1],
                                   ["chronique à seuil"]]}

#en µg/m3
VTR_seuils_CAREPS = {"Acétaldéhyde":[["Valeur toxicologique de référence", "Valeur toxicologique de référence", "Valeur toxicologique de référence"],
                                     [0.009*1e3, 0.39*1e3, 0.14*1e3],
                                     ["µg/m3","µg/m3", "µg/m3"],
                                     ["respiratoire (altération de l'épithélium nasal) ; cible: rat", "respiratoire (altération de l'épithélium nasal) ; cible: rat", "respiratoire (muqueuses nasal) ; cible: rat"]],

                    "Acroléine":[["Valeur toxicologique de référence", "Valeur toxicologique de référence", "Valeur toxicologique de référence"],
                                 [0.02, 0.4, 0.35],
                                 ["µg/m3","µg/m3", "µg/m3"],
                                 ["respiratoire (lésions nasales)  ; cible: rat ", "respiratoire (nez) ; cible: rat", "respiratoire (voies supérieures) ; cible: rat"]],

                    "Benzene (71-43-2)":[["Valeur toxicologique de référence", "Valeur toxicologique de référence", "Valeur toxicologique de référence"],
                                         [0.03*1E3, 9.78, 0.06*1E3],
                                         ["µg/m3","µg/m3", "µg/m3"],
                                         ["diminution du nb de lymphocytes ; cible: humain", "diminution du nb de lymphocytes ; cible: humain", "diminution du nb de globules blanc et rouges ; cible: humain"]],

                    "Chloroforme":[["Valeur toxicologique de référence", "Valeur toxicologique de référence", "Valeur toxicologique de référence", "Valeur toxicologique de référence"],
                                   [0.1*1e3, 0.1*1e3, 3.4*1e3, 0.3*1e3],
                                   ["µg/m3","µg/m3", "µg/m3","µg/m3"],
                                   ["foie ; cible: humain", "foie ; cible: rat", "foie ; cible: chien", "foie, rein ; cible: rat"]],

                    "Formaldéhyde":[["Valeur toxicologique de référence", "Valeur toxicologique de référence"],
                                    [0.01*1e3, 0.009*1e3],
                                    ["µg/m3","µg/m3"],
                                    ["respiratoire (altération de l'épithélium nasal) ; cible: humain", "respiratoire (irritation des yeux, des voies respiratoires et altération de l'épithélium nasal)"]],

                    "SOx":[["Valeur toxicologique de référence"],
                           [50000],
                           ["µg/m3"],
                           ["chronique à seuil (cette valeur semble erronée d'un facteur 1000)"]],

                    "NOx":[["Valeur toxicologique de référence"],
                            [40],
                            ["µg/m3"],
                            ["chronique à seuil, trouble respiratoire chez certains sujets sensibles (asthme)"]],
}

VTR_seuils_INERIS = {"SOx":[["Niveau de risque minimum (MRL)"],
                            [50],
                            ["µg/m3"],
                            ["aigue à seuil"]],

                    "CO":[["VTR retenu par l'ANSES","VTR retenu par l'ANSES"],
                          [10e3,30e3],
                          ["µg/m3","µg/m3"],
                          ["Pour une exposition de 8 heures","Pour une exposition de 1 heure"]],
                    
                    "Formaldéhyde":[["VTR retenu par l'INERIS","VTR retenu par l'INERIS", "VTR retenu par l'ANSES", "VTR retenu par l'ANSES"],
                                    [9,50,100,123],
                                    ["µg/m3","µg/m3","µg/m3","µg/m3"],
                                    ["","MRL (Minimum Risk Level)","	Pour une exposition à court terme et de manière répétée et continue pour toute la journée",""]],

                    "Acétaldéhyde":[["VTR retenu par l'INERIS et l'ANSES","VTR retenu par l'INERIS et l'ANSES"],
                                    [160,3e3],
                                    ["µg/m3","µg/m3"],
                                    ["Pour une exposition supérieure à 1 an","durée: 1heure"]],

                    "Acroléine":[["VTR retenu par l'INERIS et l'ANSES","VTR retenu par l'INERIS et l'ANSES"],
                                 [0.15,6.9],
                                 ["µg/m3","µg/m3"],
                                 ["","MRL"]],

                    "Benzene (71-43-2)":[["Objectif de qualité","Valeur limite","VTR retenu par l'ANSES","VTR retenu par l'ANSES"],
                                         [2,5,10,30],
                                         ["µg/m3","µg/m3","µg/m3","µg/m3"],
                                         ["moyenne annuelle","Protection de la santé Humaine : moyenne annuelle","Pour une exposition supérieure à 1 an","pour une exposition de 1 à 14 jours"]],

                    "Acide acrylique":[["VTR construite par les organismes reconnus","VTR construite par les organismes reconnus","Valeur publiée par l'INRS pour la population professionnelle"],
                                       [1,6e3,29e3],
                                       ["µg/m3","µg/m3","µg/m3"],
                                       ["","",""]],

                    "Nitrobenzène":[["VTR construite par les organismes reconnus","Valeur publiée par l'INRS pour la population professionnelle","VTR construite par les organismes reconnus"],
                                    [200,1e3,6],
                                    ["µg/m3","µg/m3","µg/m3"],
                                    ["MRL","","MRL"]],

                    "Benzo(a)pyrène + Naphtalène":[["Valeur cible du Benzo(a)pyrène","VTR retenu par l'INERIS du Benzo(a)pyrène","Valeur publiée pour la population professionnelle du Benzo(a)pyrène","VTR retenu par l'INERIS et l'ANSES du Naphtalène"],
                                                   [0.001,2,2e-3,37],
                                                   ["µg/m3","µg/m3","µg/m3","µg/m3"],
                                                   ["Moyenne calculée sur une année civile, du contenu total de la fraction PM10. Le volume d'échantillonnage se réfère aux conditions ambiantes. Les concentrations en arsenic, cadmium, nickel et benzo(a)pyrène correspondent à la teneur totale de ces éléments et composés dans la fraction PM10.","VTR retenu par l'INERIS","Valeur publiée par pour la population professionnelle",""]],

                    "Chrome (Cr)":[["Valeur publiée par l'INRS pour la population professionnelle","VTR retenu par l'INERIS","VTR retenu par l'INERIS"],
                                   [5,0.03,1],
                                   ["µg/m3","µg/m3","µg/m3"],
                                   ["Valeur réglementaire contraignante pour le chrome VI et ses composés","Chrome VI sous forme de particules","Valeur réglementaire contraignante pour le chrome VI et ses composés"]],
                    
                    "Phénol":[["Valeur publiée par l'INRS pour la population professionnelle","VTR construite par les organismes reconnus","Valeur publiée par l'INRS pour la population professionnelle","VTR construite par les organismes reconnus"],
                              [15.6e3,5.8e3,7.8e3,20],
                              ["µg/m3","µg/m3","µg/m3","µg/m3","µg/m3"],
                              ["Valeur limite réglementaire contraignante","","Valeur limite réglementaire contraignante",""]],

                    "Cobalt (Co)":[["VTR construite par les organismes reconnus"],
                                   [0.1],
                                   ["µg/m3"],
                                   ["cobalt et composés inorganiques"]],

                    "Arsenic (As)":[["Valeur cible","VTR construite par les organismes reconnus","VTR retenu par l'INERIS"],
                                    [0.006,0.015,0.015],
                                    ["µg/m3","µg/m3","µg/m3"],
                                    ["Les concentrations en arsenic, cadmium, nickel et benzo(a)pyrène correspondent à la teneur totale de ces éléments et composés dans la fraction PM10","8 hour inhalation. Arsenic & inorganic compounds including arsine","Arsenic et dérivés inorganiques"]],

                    "Béryllium (Be)":[["VTR retenu par l'INERIS","Valeur publiée par l'INRS pour la population professionnelle"],
                                      [0.007,0.0002],
                                      ["µg/m3","µg/m3"],
                                      ["Béryllium et dérivés","VLEP contraignante pour le béryllium et ses composés inorganiques (fraction inhalable)"]],

                    "Nickel (Ni)":[["VTR construite par les organismes reconnus","VTR retenu par l'INERIS"],
                                   [0.06,0.09],
                                   ["µg/m3","µg/m3"],
                                   ["","Nickel et composés (hors oxyde de nickel)"]],

                    "Cadmium (Cd)":[["Valeur cible","VTR construite par les organismes reconnus","VTR retenu par l'INERIS et l'ANSES"],
                                    [0.005,0.03,0.3],
                                    ["µg/m3","µg/m3"],
                                    ["Les concentrations en arsenic, cadmium, nickel et benzo(a)pyrène correspondent à la teneur totale de ces éléments et composés dans la fraction PM10","","Cadmium et ses composés. Effets cancérogènes"]],
}


###############################################################################
#
#              Synthèse
#
###############################################################################

Composés = {"SOx": {'titre':'Concentration en Oxyde de Soufre SOx',
                    "VTR":[[SOx_VTR, "https://www.airparif.asso.fr/la-reglementation-en-france", [2, 3, 4, 5], [0, 1]],
                           [pd.DataFrame(VTR_seuils_CAREPS['SOx']).transpose(), 'Etude CAREPS', [], [0]],
                           [pd.DataFrame(VTR_seuils_INERIS['SOx']).transpose(), 'Etude INERIS', [0], []]
                           ],
                    "contour_aigue":[0.0001, 0.1, 1, 10, 50, 125, 300, 350, 500, 50E3], 
                    "contour_aigue color":["indigo","navy", 'teal', "lightgreen", "orange", "deeppink", "fuchsia", "fuchsia", "fuchsia"],
                    "contour_chronique":[0.0001, 0.1, 1, 10, 20, 50, 50E3], 
                    "contour_chronique color":["indigo", "navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':SOx_Background},
            
            "NOx": {'titre':"Concentration en Oxyde d'Azote NOx",
                    "VTR":[[NOx_VTR, "https://www.airparif.asso.fr/la-reglementation-en-france", [2, 3, 4], [0, 1]],
                           [pd.DataFrame(VTR_seuils_CAREPS['NOx']).transpose(), 'Etude CAREPS', [], [0]],
                           ],
                    "contour_aigue":[0.0001, 0.1, 1, 10, 40, 200, 400], 
                    "contour_aigue color":["indigo", "navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 0.1, 1, 10, 30, 40, 100], 
                    "contour_chronique color":["indigo", "navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':NOx_Background},
            
            "Chrome (Cr)": {'titre':"Concentration en Chrome",
                            "VTR":[[pd.DataFrame(VTR_seuils_anses['Chrome (Cr)']).transpose(), 'ANSES', [], [0, 1]],
                                    [pd.DataFrame(VTR_seuils_INERIS['Chrome (Cr)']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/668', [0], [1, 2]]
                                   ],
                            "contour_aigue":[0.0001, 0.001, 0.03, 0.1, 1, 5, 10], 
                            "contour_aigue color":["indigo", "navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                            "contour_chronique":[0.000001, 0.00001, 0.0001, 0.001, 0.03, 1, 5], 
                            "contour_chronique color":["indigo", "navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                            "ERU":[[pd.DataFrame(ERU_anses['Chrome (Cr)']).transpose(), 'ANSES']],
                            'Background':None
                                   },

            "Formaldéhyde": {'titre':"Concentration en Formaldéhyde (COV)",
                    "VTR":[[pd.DataFrame(VTR_seuils_CAREPS['Formaldéhyde']).transpose(), 'Etude CAREPS', [0, 1], []],
                           [pd.DataFrame(VTR_seuils_anses['Formaldéhyde']).transpose(), 'ANSES', [0], [1]],
                           [pd.DataFrame(VTR_seuils_INERIS['Formaldéhyde']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/1008', [1,2], [0,3]],
                           ],
                    "contour_aigue":[0.0001, 1, 10, 50, 100, 200], 
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 1, 9, 30, 123, 200], 
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None},
                    
            "Acétaldéhyde": {'titre':"Concentration en Acétaldéhyde (COV)",
                    "VTR":[[pd.DataFrame(VTR_seuils_INERIS['Acétaldéhyde']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/309', [1], [0]],
                           ],
                    "contour_aigue":[0.0001, 1, 10, 50, 160, 200], 
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 1, 100, 1000, 3e3, 30e3], 
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None},
                    
            "Acroléine": {'titre':"Concentration en Acroléine (COV)",
                    "VTR":[[pd.DataFrame(VTR_seuils_INERIS['Acroléine']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/328', [1], [0]],
                           ],
                    "contour_aigue":[0.0001, 0.1, 1, 5, 6.9, 10, 20],
                    "contour_aigue color":["indigo", "navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.00001, 0.0001, 0.001, 0.1, 0.15, 1, 5], 
                    "contour_chronique color":["indigo", "navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None},
                    
            "Benzene (71-43-2)": {'titre':"Concentration en Benzène (COV)",
                    "VTR":[[Benzène_VTR, "https://www.airparif.asso.fr/la-reglementation-en-france", [], [0, 1]],
                           [pd.DataFrame(VTR_seuils_INERIS['Benzene (71-43-2)']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/439', [0,1,3], [2]],
                           ],
                    "contour_aigue":[0.0001, 1, 2, 5, 30, 50],
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 0.1, 1, 10, 20, 30],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':Benzène_Background},
                    
            "Acide acrylique": {'titre':"Concentration en Acide Acrylique",
                    "VTR":[[pd.DataFrame(VTR_seuils_INERIS['Acide acrylique']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/330', [1], [0,2]],
                           ],
                    "contour_aigue":[0.0001, 0.1, 10, 100, 6e3, 10e3],
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 0.01, 10, 100, 29e3, 50e3],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None},
                    
            "Nitrobenzène": {'titre':"Concentration en Nitrobenzène",
                    "VTR":[[pd.DataFrame(VTR_seuils_INERIS['Nitrobenzène']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/1320', [0,1], [2]],
                           ],
                    "contour_aigue":[0.0001, 0.1, 1, 200, 1000, 5000], 
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 0.001, 0.1, 1, 6, 15],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None},
                    
            "Benzo(a)pyrène + Naphtalène": {'titre':"Concentration en Benzo(a)pyrène et Naphtalène (COV)",
                    "VTR":[[pd.DataFrame(VTR_seuils_INERIS['Benzo(a)pyrène + Naphtalène']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/484 et https://substances.ineris.fr/fr/substance/1284', [0,1], [2]],
                           ],
                    "contour_aigue":[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 2e-3, 0.01, 1, 37, 0.15e3],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None},
            
            "Phénol": {'titre':"Concentration en Phénol",
                    "VTR":[[pd.DataFrame(VTR_seuils_INERIS['Phénol']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/1481', [0,1], [2,3]],
                           ],
                    "contour_aigue":[0.001, 0.01, 1, 10, 5.8e3, 15.6e3],
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.001, 0.01, 0.1, 1, 20, 8e3],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None},
                    
            "Poussières totale": {'titre':"Concentration en Poussières totale (PM 2.5, PM10)", #PM2.5=PM10 cf. VTR ANSES
                    "VTR":[[PM2_5_VTR, "https://www.airparif.asso.fr/la-reglementation-en-france", [], [0]],],
                    "contour_aigue":[0.001, 0.01, 0.0128, 0.05, 0.1, 0.5],
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.001, 0.01, 0.128, 0.05, 0.1, 0.5],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":[[pd.DataFrame(ERU_anses["Particules dans l'air ambiant PM10"]).transpose(), 'Etude ANSES', [0], [0]]],
                    'Background':PM_Background},
            
             "CO": {'titre':"Concentration en Monoxyde de Carbone",
                    "VTR":[[CO_VTR, "https://www.airparif.asso.fr/la-reglementation-en-france", [], [0]],
                           [pd.DataFrame(VTR_seuils_anses['CO']).transpose(), 'Etude ANSES', [1], [0]],
                           [pd.DataFrame(VTR_seuils_INERIS['CO']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/607#reference', [0,1], [0]]
                           ],
                    "contour_aigue":[0.1, 1, 10, 100, 1e4, 3e4], 
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.1, 1, 10, 100, 3e4, 10e4],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None},
             
             "Cobalt (Co)": {'titre':"Concentration en Cobalt (ETM)",
                    "VTR":[[pd.DataFrame(VTR_seuils_INERIS['Cobalt (Co)']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/681', [], [0]],
                           ],
                    "contour_aigue":[0.1, 1, 10, 100, 150, 180], 
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 0.001, 0.01, 0.1, 1, 10],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None}, 
             
             "Arsenic (As)": {'titre':"Concentration en Arsenic (ETM)",
                    "VTR":[[As_VTR, "https://www.airparif.asso.fr/la-reglementation-en-france", [], [0]],
                           [pd.DataFrame(VTR_seuils_INERIS['Arsenic (As)']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/417', [0,1], [2]],
                           ],
                    "contour_aigue":[0.0001, 0.001, 0.006, 0.015, 0.1, 1],
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 0.001, 0.006, 0.015, 0.1, 1],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None}, 

            "Béryllium (Be)": {'titre':"Concentration en Béryllium (ETM)",
                               "VTR":[[pd.DataFrame(VTR_seuils_INERIS['Béryllium (Be)']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/499', [], [0]],
                           ],
                            "contour_aigue":[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5], 
                            "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                            "contour_chronique":[0.0002, 0.001, 0.007, 0.01, 0.1, 1],
                            "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                            "ERU":None,
                            'Background':None}, 

            "Nickel (Ni)": {'titre':"Concentration en Nickel (ETM)",
                    "VTR":[[Ni_VTR, "https://www.airparif.asso.fr/la-reglementation-en-france", [], [0]],
                           [pd.DataFrame(VTR_seuils_INERIS['Nickel (Ni)']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/1301', [0], [1]],
                           ],
                    "contour_aigue":[0.0001, 0.001, 0.006, 0.009, 0.1, 0.5], 
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0002, 0.001, 0.006, 0.009, 0.1, 1],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None},
            
            "Cadmium (Cd)": {'titre':"Concentration en Cadmium (ETM)",
                    "VTR":[[Cd_VTR, "https://www.airparif.asso.fr/la-reglementation-en-france", [], [0]],
                           [pd.DataFrame(VTR_seuils_INERIS['Cadmium (Cd)']).transpose(), 'Etude INERIS https://substances.ineris.fr/fr/substance/586', [0,1], [2]],
                           ],
                    "contour_aigue":[0.0001, 0.001, 0.005, 0.03, 0.3, 0.5],
                    "contour_aigue color":["navy", 'teal', "lightgreen", 'orange', "fuchsia"],
                    "contour_chronique":[0.0001, 0.001, 0.005, 0.03, 0.3, 0.5],
                    "contour_chronique color":["navy", 'teal', "lightgreen", "orange", "fuchsia"],
                    "ERU":None,
                    'Background':None}, 
            }
