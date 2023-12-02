# Predicción de precios de coches con ML.

<img src="img/autos_pintos_logo_red.png" alt="Autos Pintos logo" width="170"/>

## Introducción.

Proyecto de Machine Learning tradicional supervisado, sobre el uso de características de un determinado modelo de vehículo de 2ª mano, con el objetivo de predecir su precio. En él, se parte de un [dataset](https://www.kaggle.com/datasets/datamarket/venta-de-coches) que contiene cerca de 50000 datos de vehículos de segunda mano y sus características. 

A partir de estos datos, se realiza una limpieza y análisis exploratorio de las variables que formarán parte del modelo predictivo final. Finalmente, se escoge el mejor modelo predictivo de entre varios, y se realiza un análisis de los resultados obtenidos.

Además, se hace un despliegue de una Web App predictora de precios en Streamlit.

## Procedimiento.

1. Introducción.
2. Obtención de los datos.
3. Limpieza de datos.
4. Feature engineering (selección y trsnformación de variables).
5. Test y selección del modelo predictivo.
6. Extracción de conclusiones y propuestas de mejora.
7. Despliegue de Web App en Streamlit (adicional).

## Métricas.

Las métrica principal utilizada para la elección del modelo en este problema de regresión es:

* [__Mean Squared Error (MSE).__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)

Además, se han obtenido otras métricas, como:

* [__Mean Absolute Error (MAE).__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
* [__Mean Absolute Percentage Error (MAPE).__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html#sklearn.metrics.mean_absolute_percentage_error)
* [__r2 Score.__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)


## Tecnologías utilizadas.

```Python``` / ```Pandas``` / ```NumPy``` / ```Matplotlib``` / ```Seaborn``` / ```Scikit-Learn``` / ```Streamlit``` / ```Keras```

## Archivos y carpetas.

El proyecto contiene las siguientes carpetas:

1. data: contiene todos los archivos de datos utilizados en el análisis, tanto los utilizados inicialmente, como los elaborados a partir de los notebooks en la carpeta "notebooks".

2. img: contiene imágenes utilizadas en el notebook principal.

3. model: contiene los modelos guardados creados en el notebook principal.

4. src/notebooks/pruebas: notebooks de limpieza, procesado, EDA y pruebas de modelos de Machine Learning y Deep Learning.

5. src/notebooks/car_price_project.ipynb: notebook principal (memoria) del proyecto.

6. streamlit: contiene la Web App en Streamlit (streamlit_app.py), así como los archivos necesarios para que esta funcione.

7. utils: contiene los archivos necesarios para el procesado de datos y el desarrollo del proyecto.

## Link a la Web App en Streamlit.

[https://appcarprice-apintos2.streamlit.app/](https://appcarprice-apintos2.streamlit.app/)



