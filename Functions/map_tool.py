# map_tools.py>
# function
import numpy 
import streamlit as st

def ZoomLvl(max_delta):
    if max_delta<0.005: zoom_lvl=16
    elif max_delta<0.011:zoom_lvl=15
    elif max_delta<0.022:zoom_lvl=14
    elif max_delta<0.044:zoom_lvl=13
    elif max_delta<0.088:zoom_lvl=12
    elif max_delta<0.176:zoom_lvl=11
    elif max_delta<0.352:zoom_lvl=10
    elif max_delta<0.703:zoom_lvl=9
    elif max_delta<1.406:zoom_lvl=8
    elif max_delta<2.813:zoom_lvl=7
    elif max_delta<5.625:zoom_lvl=6
    elif max_delta<11.25:zoom_lvl=5
    elif max_delta<22.5:zoom_lvl=4
    elif max_delta<45:zoom_lvl=3
    elif max_delta<90:zoom_lvl=2
    elif max_delta<180:zoom_lvl=1
    else: zoom_lvl=0
    return zoom_lvl

def DistanceAB_Earth(LatA,LatB,LonA,LonB):
    Earth_rad = 6371
    
    Distance = numpy.arccos(numpy.sin(numpy.radians(float(LatA)))*numpy.sin(numpy.radians(float(LatB)))+
                        numpy.cos(numpy.radians(float(LatA)))*numpy.cos(numpy.radians(float(LatB)))*
                        numpy.cos(numpy.radians(float(LonA)-float(LonB))))*Earth_rad
    return Distance

def SwitchMapStyle():
    ToggleSat = st.toggle('Vue carte / Vue satellite')
    if ToggleSat:
        MapTiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'"
    else:
        MapTiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png'
    return MapTiles
