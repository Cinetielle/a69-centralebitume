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

import inspect
import textwrap

import streamlit as st

import numpy as np

def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

def Δh_Holland(Vs, v, d, Pa, Ts, Ta):
    """

    Parameters
    ----------
    Pa : TYPE
        pression atmospherique en mbar
    Vs : TYPE
        vitesse des gaz en sortie de cheminée en m/s
    v : TYPE
        vitesse du vent à la hauteur de la cheminée en m/s
    d : TYPE
        Diamètre de la cheminée en m
    Ts : TYPE
        Température en sortie de cheminée en °K
    Ta : TYPE
        Temperature ambiante en °K

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return Vs*d*(1.5+0.00268*Pa*d*((Ts-Ta)/(Ts+273.15)))/v

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

def  stability_pasquill(v, RSI, HR, mode='24H'):
    """
    Définis les conditions suivantes et renvoi la stabilité de Pasquill en conséquence
    'JOUR_RSI_fort', 'JOUR_RSI_modéré', 'JOUR_RSI_faible', 'NUIT_NEBULOSITE_4/8-7/8', 'NUIT_NEBULOSITE_<3/8'

    Parameters
    ----------
    v : float
        wind speed at 10m , m/s
    RSI : float
        RSI= rayonnement solaire incident moyen
    HR : float
        humidité relative
        on fait l'hypothèse que si l'air au sol est sec (< 60 %) la nébuolisté équivaut à 'NUIT_NEBULOSITE_<3/8'
        sinon la nébulosité équivaut à 'NUIT_NEBULOSITE_4/8-7/8'

    mode : str, optional
        '24H', '1H', '10M'
        

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
    
    if mode == "24H":
        if RSI > 260: # Q3 de l'insolation moyenne sur 24H
            RSI='JOUR_RSI_fort'
        elif (RSI < 90):  # Q1 de l'insolation moyenne sur 24H
            RSI='JOUR_RSI_faible'
        else:
            RSI='JOUR_RSI_modéré'
    else:
        if RSI > 470:  # Q3 de l'insolation moyenne journalière (si insolation > 0)
            RSI='JOUR_RSI_fort'
        elif (RSI < 200) & (RSI > 0): # Q1 de l'insolation moyenne journalière (si insolation > 0)
            RSI='JOUR_RSI_faible'
        elif (HR >= 60) & (RSI == 0):
            RSI='NUIT_NEBULOSITE_<3/8'
        elif (HR < 60) & (RSI == 0):
            RSI='NUIT_NEBULOSITE_4/8-7/8'            
        else:
            RSI='JOUR_RSI_modéré'
    
    if v < 2:
        if RSI=='JOUR_RSI_fort':
            return 'A'
        elif RSI=='JOUR_RSI_modéré':
            return 'A-B'
        elif RSI=='JOUR_RSI_faible':
            return 'B'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'F'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'F'
    elif (v >= 2) & (v < 3):
        if RSI=='JOUR_RSI_fort':
            return 'A-B'
        elif RSI=='JOUR_RSI_modéré':
            return 'B'
        elif RSI=='JOUR_RSI_faible':
            return 'C'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'E'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'F'
    elif (v >= 3) & (v < 5):
        if RSI=='JOUR_RSI_fort':
            return 'B'
        elif RSI=='JOUR_RSI_modéré':
            return 'B-C'
        elif RSI=='JOUR_RSI_faible':
            return 'C'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'D'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'E'
    elif (v >= 5) & (v < 6):
        if RSI=='JOUR_RSI_fort':
            return 'C'
        elif RSI=='JOUR_RSI_modéré':
            return 'C-D'
        elif RSI=='JOUR_RSI_faible':
            return 'D'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'D'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'D'
    elif (v >= 6):
        if RSI=='JOUR_RSI_fort':
            return 'C'
        elif RSI=='JOUR_RSI_modéré':
            return 'D'
        elif RSI=='JOUR_RSI_faible':
            return 'D'
        elif RSI=='NUIT_NEBULOSITE_4/8-7/8':
            return 'D'
        elif RSI== 'NUIT_NEBULOSITE_<3/8':
            return 'D'
    else:
        print('Pasquill Stability error')
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
                        [np.exp(-1.104+0.9878*np.log(x)-0.0076*np.log(x)**2) , np.exp(4.679-1.172*np.log(x)+0.227*np.log(x)**2)],
                        [0.4*x**0.91, 0.4*x**0.91],
                        [0.469*x**0.903, 0.017*x**1.38]])
        return A
    elif stability == 'A-B':
        A = np.asarray([[0.443*x**0.894, empty],
                        [np.exp(-1.104+0.9878*np.log(x)-0.0076*np.log(x)**2) , np.exp(4.679-1.172*np.log(x)+0.227*np.log(x)**2)],
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
        D = np.asarray([[0.141*x**0.894, empty],
                        [np.exp(-2.555+1.0423*np.log(x)-0.0087*np.log(x)**2) , np.exp(-3.186+1.1737*np.log(x)-0.0316*np.log(x)**2)],
                        [0.32*x**0.78, 0.22*x**0.78],
                        [0.219*x**0.764, 0.140*x**0.727]])
        return D
    
class normalize(object):
    """
    O2_ref : float, optional
        O2 fraction for reference
    P0 : float, optional
        °K, température de référence
    P0 : float, optional
        Pa, pression de référence
    """
    def __init__(self, O2_ref=17, T0=273, P0=101.3*1E3):
        self.O2ref = O2_ref
        self.T0 = T0
        self.P0 = P0
    
    def get_P_with_Qnorm(self, Q_norm, v_out, T, O2, H, S=np.pi*(1.35/2)**2):
        self.P = Q_norm/((v_out*S*3600)/(T/self.T0))*self.P0

    def get_Q_norm(self, v_out, T, O2sec, H, P, S=np.pi*(1.35/2)**2, output=True):
        """
        Parameters
        ----------
        v_out : float
            m/s, vitesse d'éjection des gaz de la cheminée'
        T : float
            °C, température du gaz en sortie de cheminée
        O2sec : float
            % O2 ; O2sec = O2hum/(100-H)
        H : float
            % humidité relative
        P : float
            Pa, pression de l'air en sortie de cheminée
        S : float, optional
            m2, surface (intérieure) du conduit de cheminée au niveau du rejet
        """
        T=T+self.T0
        hum = (100-H)/100
        if P is None:
            try:
                P =self.P
            except:
                print("La pression n'est pas définie")
                
        Q = v_out*S*3600
        
        Qhum =(Q*P*self.T0)/(self.P0*T)
        Qsec = Qhum*hum
        O2_corr = (21-O2sec)/(21-self.O2ref)
        Qhum_O2ref = Qhum*O2_corr
        self.Q0 = Qhum_O2ref
        if output:
            return {'Q, m3/h':Q, 'Qhum, (Nm3/h)h':Qhum, 'Qsec, (Nm3/h)s':Qsec, f'Qhum_{int(self.O2ref*100)}%O2, (Nm3/h)h_O2':Qhum_O2ref}
    
    def get_DébitMassique(self, concentration_0, mode='h_02'):
        """

        Parameters
        ----------
        concentration_0 : float
            (g/Nm3)h_17%O2

        Returns
        -------
        float
            g/s, le débit massique horaire

        """
        try:
            self.DM = concentration_0*self.Q0/3600
            return self.DM
        except:
            print("Il faut définir le débit normalisé (self.get_Q_norm())")
            
if __name__ == '__main__':
    centrale = normalize(O2_ref=17, T0=273, P0=101.3E3)
    Q=71100 #(m3/h)h
    Qh = 53700 #(Nm3/h)h
    Qsec=46300 #(Nm3/h)s
    O2 = 0.1494 # fraction d'O2
    T= 83.4 #°C
    v= 13.8 #m/s
    H= 13.7 #%humidité relative
    d=1.35
    S=np.pi*(d/2)**2 #m2
    P=99.8*1E3#-0.05*1E3 #Pa
    MVsec = 1.09 #(kg/Nm3)s
    MVhum = 0.93 #(kg/Nm3)h
    Cpoussière_h = 4.5 #(mg/Nm3)h
    Cpoussière_s = 5.2 #(mg/Nm3)s
    Cpoussière_h_02 = 2.2 #(mg/Nm3)h_O2
    FluxHoraire_poussière = 0.24 #kg/h 0.06666 g/s
    
    Q_test = centrale.get_Q_norm(v, T, O2, H, P, S=S)
    poussière = centrale.get_DébitMassique(Cpoussière_h_02*1E-3)
    53664.24864014637*Cpoussière_h*1E-3
    46312.24657644631*Cpoussière_s*1E-3
    82009.9799731468*Cpoussière_h_02*1E-3



    centrale = normalize(O2_ref=17, T0=273, P0=101.3E3)
    
    Qh = 53400 #(Nm3/h)h
    Qsec= 46500#(Nm3/h)s
    O2 = 13/(1-0.129) # fraction d'O2
    T= 131 #°C
    v= 21.8 #m/s
    H= 12.9 #%humidité relative
    d=1.15
    S=np.pi*(d/2)**2 #m2
    Q= v*S*3600 #(m3/h)h
    P=98.3*1E3#-24.2 #Pa


    Cpoussière_s = 11.8 #(mg/Nm3)s
    C_CO_h_02 = 98.1 #(mg/Nm3)h_O2
    FluxHoraire_poussière = 0.546 #kg/h
    FluxHoraire_CO = 7.95 #kg/h
    
    Q_test = centrale.get_Q_norm(v, T, O2, H, P, S=S)
    CO = centrale.get_DébitMassique(C_CO_h_02*1E-3)

    46557.329967947306*Cpoussière_s*1E-3
    81176.3511530517*C_CO_h_02*1E-3
    
    
    centrale = normalize(O2_ref=17, T0=273, P0=101.3E3)
    
    H= 5 #%humidité relative
    O2 = 10/(1-H/100) # fraction d'O2
    T= 60 #°C
    v= 23.4 #m/s
    d=1.35
    S=np.pi*(d/2)**2 #m2
    Q= v*S*3600 #(m3/h)h
    P=100.4*1E3#-24.2 #Pa
    Q_test = centrale.get_Q_norm(v, T, O2, H, P, S=S)