import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import utilidades as util
from pickle import dump
from pickle import load
import numpy as np
import plotly.express as px

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
#leer los datos
df= pd.read_csv('data/Datos_Pacientes.csv', index_col= 0)

#Seleccionar graficos
st.header('Visualización')
tipo = st.selectbox('Seleccione el tipo de grafico',
                    ['Barras', 'Líneas', 'Dispersión'])

#selector de las variables a comparar
variable = st.selectbox('Seleccione la variable a comparar',
                        df.columns[1:].values)

#después de seleccionar

if tipo == 'Barras':
    fig = px.bar(df,x = 'Enfermedad', y= variable, barmode= 'group',
                 title= f'Pacientes con o sin SMEC {variable}')
elif tipo == 'Líneas':
    fig = px.line(df,x = 'Enfermedad', y= variable,
                 title= f'Pacientes con o sin SMEC {variable}')
elif tipo == 'Dispersión':
    fig = px.scatter(df,x = 'Enfermedad', y= variable,
                 title= f'Pacientes con o sin SMEC {variable}')
 

st.plotly_chart(fig)