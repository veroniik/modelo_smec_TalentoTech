import streamlit as st
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import utilidades as util
from pickle import dump
from pickle import load
import numpy as np


#configurar la pagina
st.set_page_config(
    page_title= "SMEC",
    page_icon = ":heartpulse:",
    initial_sidebar_state= "expanded",
    layout= "centered", 
)

#llamamos el menu

util.generarMenu()


st.title('SMEC- Sindrome metabolico de enfermedad cardiovascular')
st.header('Datos')

#leer los datos
df= pd.read_csv('data/Datos_Pacientes.csv', index_col= 0)

st.markdown('**Datos de los pacientes')
st.write(df)


st.markdown('**Resultado del modelo Random Forest para los datos')

util.modelo_rf(df)

#generamos los objetos para ingresar datos
with st.sidebar:
    st.header('Datos para el diagnostico')
    hipertension = st.checkbox('Hipertensión')
    hiperglucemia = st.checkbox('Hiperglucemia')
    hdl = st.checkbox('HDL bajo')
    hipertri = st.checkbox('Hipertridiglicemia')
    ica = st.checkbox('ICA')
    edad = st.number_input('Edad',min_value=18, max_value=90)
    genero = st.selectbox('Genero', ('Femenino', 'Masculino'))
    tabaco = st.checkbox('Fuma')
    alcohol = st.checkbox('Alcohol')
    poliu = st.checkbox('Poliurea')
    btn_ejecutar = st.button('Diagnosticar')

# lógica de la predicción
lista = [0,0,0,0,0,0,0,0,0,0]
if btn_ejecutar == True:
    if hipertension == True:
        lista[0] = 1
    else:
        lista[0] = 0
    
    if hiperglucemia == True:
        lista[1] = 1
    else:
        lista[1] = 0
    
    if hdl == True:
        lista[2] = 1
    else:
        lista[2] = 0
    
    if hipertri == True:
        lista[3] = 1
    else:
        lista[3] = 0
    
    if ica == True:
        lista[4] = 1
    else:
        lista[4] = 0
    
    lista[5]=edad
    if genero == 'Masculino':
        lista[6] = 1
    else:
        lista[6] = 0
    
    if tabaco == True:
        lista[7] = 1
    else:
        lista[7] = 0
   
    if alcohol == True:
        lista[8] = 1
    else:
        lista[8] = 0
    
    if poliu == True:
        lista[9] = 1
    else:
        lista[9] = 0
    arr = np.array([lista])
    #st.write(arr)
    util.prueba_modelo(arr)