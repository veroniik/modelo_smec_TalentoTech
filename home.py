import pandas as pd
import streamlit as st
import utilidades as util

#configurar la pagina principal

st.set_page_config(
    page_title= "SMEC",
    page_icon = ":heartpulse:",
    initial_sidebar_state= "expanded",
    layout= "centered", 
)

#llamamos el menu

util.generarMenu()

st.title("SMEC - Síndrome Metabólico de Enfermedad Cardiovascular")

st.write(
    """
        Determinar si un paciente al cual se le realizan diferentes estudios clínicos para hallar enfermedades como: Hipertensión, Hiperglusemia, Colesterol HDL bajo, Hipertriglidicemia, Trastorno de cintura-altura y poliúrea. Además, se le preguntan datos como: Edad, Género, si fuma y si consume licor.
Todo esto con la finalidad de diagnosticar si la persona posee un síndrome metabólico asociado a enfermedad cardiovascular (SMEC), a la cual llamaremos enferdedad, una variable categórica que vamos a predecir a través del modelo de Bosques Aleatorios (Random Forest).

Los datos se encuentran en la carpeta:\n\n https://drive.google.com/drive/folders/1_E-q91yPR_wAi__1blntpTC3kXEpBHIb?usp=sharing
"""
)


#Poner imagenes a la página
from PIL import Image
imagen = Image.open("media/imagen_2.png")

#incrustamos la imagen
st.image(imagen, caption = "Enfermedad Cardiovascular",
         use_container_width= True, width= 200)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



st.header("Key Performance indicators (KPIs)")

st.write("""
- KPI: Identificar a través de los parámetros de las enfermedades de base de cada paciente, y sus datos médicos generales como el género, la edad, si consume o no, tabaco y alcohol, para determinar si el paciente puede padecer SMEC.
""")