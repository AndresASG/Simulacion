import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import json
import sqlite3
# Funciones auxiliares
#
def graficar_predicciones(real, prediccion):
    plt.plot(real[0:len(prediccion)],color='red', label='Valor real de la acción')
    plt.plot(prediccion, color='blue', label='Predicción de la acción')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo en días')
    plt.ylabel('Valor de la acción')
    plt.legend()
    plt.show()

#
# Lectura de los datos
#
dataset = pd.read_csv('DatosAccionesApple.csv', index_col='Fecha', parse_dates=['Fecha'])
dataset.head()

#
# Sets de entrenamiento y validación 
# La LSTM se entrenará con datos de 2016 hacia atrás. La validación se hará con datos de 2017 en adelante.
# En ambos casos sólo se usará el valor más alto de la acción para cada día
#
dataset = dataset.sort_index()
set_entrenamiento = dataset[:'2022'].iloc[:,2:3]
set_validacion = dataset['2023':].iloc[:,2:3]
print("Set de entrenamiento Antes")
print(set_entrenamiento)
print()
print("Set de validación Antes")
print(set_validacion)

# Reemplazar comas por puntos en la columna 'Maximo'

set_entrenamiento['Máximo'] = set_entrenamiento['Máximo'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
# Reemplazar comas por puntos en la columna 'Máximo' para set_validacion
set_validacion['Máximo'] = set_validacion['Máximo'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)


set_entrenamiento['Máximo'].plot(legend=True)
set_validacion['Máximo'].plot(legend=True)

plt.legend(['Entrenamiento (2015-2022)', 'Validación (2023)'])
plt.show()

# Normalización del set de entrenamiento
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 60
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

print(X_train.shape)
# Reshape X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X_train shape")
print(X_train.shape)
#
# Red LSTM
#
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 50

modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train,Y_train,epochs=45,batch_size=32)


#
# Validación (predicción del valor de las acciones)
#
x_test = set_validacion.values
x_test = sc.transform(x_test)



X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

# Graficar resultados
graficar_predicciones(set_validacion.values,prediccion)

#Cambiamos el arreglo de numpy a un dataFrame de Pandas
df_predict = pd.DataFrame(prediccion, columns=['MaxHight'])
print("Set Validación")

print(set_validacion)
def baseDeDatos(df_predict,contexto):
   conn = sqlite3.connect("C:\\Users\\USUARIO\\Desktop\\Proyecto LVBP\\db.sqlite3")

   # Escribimos el DataFrame a la base de datos
   df_predict.to_sql(f'Tabla_{contexto}', conn, if_exists='replace', index=False)
   # Cerramos la conexión a la base de datos
   conn.close()
def guardadoEnDB(df_prediccion,set_validacion,nombreTabla,nombValTabla):

    baseDeDatos(df_prediccion,nombreTabla)
    baseDeDatos(set_validacion,nombValTabla)
    
guardadoEnDB(df_predict,set_validacion,"Appel","ValApple")








