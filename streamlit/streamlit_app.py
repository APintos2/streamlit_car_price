import streamlit as st
import streamlit_theme as stt
import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
import pickle
from PIL import Image

# Streamlit se ejecuta siempre desde scripts (.py) y desde el cmd en la carpeta donde está el .py y con el conda activate eda_env ejecutado antes
# pip install streamlit en la terminal
# Para lanzar la web ejecutar el comando: "streamlit run streamlit_app.py". Después se actualiza en la web en un botón cada vez que guardamos el .py

# Función para cargar modelos y datos
def cargar_modelos_datos():
    loaded_model = pickle.load(open("./streamlit/utils/modelo_final.pkl", 'rb'))
    make_model_set = pickle.load(open("./streamlit/utils/make_model_set.pkl", 'rb'))
    encoder_make = pickle.load(open("./streamlit/utils/label_encoder_make.pkl", 'rb'))
    encoder_model = pickle.load(open("./streamlit/utils/label_encoder_model.pkl", 'rb'))
    scaler = pickle.load(open("./streamlit/utils/scaler.pkl", 'rb'))

    return loaded_model, make_model_set, encoder_make, encoder_model, scaler

# Función principal para la calculadora de precios
def calculadora_precios():
    make = st.selectbox('Marca del coche:', sorted(list(make_model_set.keys())))
    model = st.selectbox('Modelo del coche:', sorted(make_model_set[make]))
    #fuel = st.selectbox('Combustible:', ('Diésel', 'Gasolina', 'Híbrido', 'Eléctrico', 'Híbrido enchufable', 'Gas licuado (GLP)', 'Gas natural (CNG)'))
    fuel = st.radio('Combustible:', options=['Diésel', 'Gasolina', 'Híbrido', 'Eléctrico', 'Híbrido enchufable', 'Gas licuado (GLP)', 'Gas natural (CNG)'], 
          horizontal=True)
    year = st.number_input('Año de matriculación:', 1980, date_time.year,  step=1)
    kms = st.number_input('Número de kilómetros:', 0, 500000, step=1000)
    power = st.number_input('Potencia (CV):', 0, 500, step=20)
    doors = st.number_input('Nº de puertas:', 3, 6, step=2)
    shift = st.selectbox('Tipo de transmisión:', ('Manual', 'Automático'))
    is_professional = st.selectbox('¿Vendedor profesional?', ("Sí", "No"))

    data_new = pd.DataFrame({
        'make': make,
        'model': model,
        'fuel': fuel,
        'year': year,
        'kms': kms,
        'power': power,
        'doors': doors,
        'shift': shift,
        'is_professional': is_professional
    }, index=[0])

    
    # Aplicar las transformaciones a data_new

    data_new["make"] = encoder_make.transform(data_new["make"])
    data_new["model"] = encoder_model.transform(data_new["model"])

    fuel_types = {"Diésel" : 0,
            "Gasolina" : 1,
            "Híbrido" : 2,
            "Eléctrico" : 3,
            "Híbrido enchufable" : 4, 
            "Gas licuado (GLP)" : 5,
            "Gas natural (CNG)" : 6}
                        
    data_new["fuel"] = data_new["fuel"].map(fuel_types)
    print(data_new["fuel"])

    num_puertas = {5: 5,
            4 : 5,
            3 : 3,
            2 : 3}

    data_new["doors"] = data_new["doors"].map(num_puertas)

    data_new["shift"] = data_new["shift"].apply(lambda x: 0 if x == "Manual" else 1)

    data_new["is_professional"] = data_new["is_professional"].apply(lambda x: 1 if x == "Sí" else 0)

    data_new["make"] = data_new["make"].astype("int")
    data_new["model"] = data_new["model"].astype("int")
    data_new["fuel"] = data_new["fuel"].astype("int")
    data_new["year"] = data_new["year"].astype("float")
    data_new["kms"] = data_new["kms"].astype("float")
    data_new["power"] = data_new["power"].astype("float")
    data_new["doors"] = data_new["doors"].astype("int")
    data_new["shift"] = data_new["shift"].astype("int")
    data_new["is_professional"] = data_new["is_professional"].astype("int")

    # Realizar la predicción

    try: 
        if st.button('Predecir precio'):
            data_new = scaler.transform(data_new)
            prediction = loaded_model.predict(data_new)
            if prediction>0:
                st.balloons()
                st.snow()
                prediction = np.exp(prediction)
                st.success(f'El precio estimado del coche es de {prediction[0].round(0)} euros.')
            else:
                st.warning("No puedes vender este coche.")
    except:
        st.warning("Ups!! Algo ha salido mal.\nPrueba de nuevo.")

    # if st.button('Recargar Página'):
    #     st.rerun()
    #     st.session_state.clear()


# Streamlit app
date_time = datetime.datetime.now()

st.set_page_config(
    page_title="Autos Pintos App",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

image_path = Image.open("./streamlit/utils/autos_pintos_logo_red.png")
st.image(image_path, width=300)

st.header("Bienvenid@ a Autos Pintos App")

st.subheader("Introduce las características del coche para predecir su precio:")

# st.sidebar.text("Menu")

# image_path = "../img/banner.png"
# st.image(image_path, width=300)

# home = st.sidebar.button("Home")
# data = st.sidebar.button("Data")
# calculadora = st.sidebar.button("Calculadora")

loaded_model, make_model_set, encoder_make, encoder_model, scaler = cargar_modelos_datos()
calculadora_precios()  # Llamada a la función


# if st.button('Recargar Página'):
#     st.rerun()
