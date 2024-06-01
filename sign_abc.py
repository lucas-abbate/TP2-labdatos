# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:39:20 2024

@author: docentes Labo Datos
"""
# script para plotear letras

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint
import funciones as fx
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics, tree
import seaborn as sns
from sklearn.model_selection import KFold

# %%######
### Explore Data
data = pd.read_csv("emnist_letters_tp.csv", header=None, index_col=0)

# %%
# print(data.max(axis='columns').sort_values(ascending=True))  # Todas las obs. tienen max = 255, asi que normalizarlo es dividir x 255
# data = data / 255 # Normalizo los datos

# %%######
data.head()
print(data.index.value_counts().sort_index())  # Balanceado
# Index etiqueta, el resto intensidades de pixeles
# 28x28 = 784 pixeles

# %%######
img = fx.show_image_from_data(26, data)  # Es una Y

# %%######

# Para el analisis exploratorio, renombro las columnas
# Quiero ver si en los datos originales van en filas o en columnas
# (o sea, si los primeros 28 son la primer fila o la primer columna)

plt.plot(img[4], "o-.", color="#D67236")
plt.xlabel("Posición")
plt.ylabel("Intensidad")
plt.title("Intensidad de la Fila 4 de la Fig. 2 (id: 26)")
plt.grid(axis="y", linewidth=0.5)
plt.gcf().set_size_inches(4, 6)

# Se ve que tiene dos picos: o sea, img[4] representa una fila

# %%######
data.iloc[26][28 * 4 : 28 * 5].reset_index(drop=True).plot(
    xlabel="Posición",
    ylabel="Intensidad",
    title="Intensidad de los datos 112 a 139 (id: 26)",
    color="#D67236",
    linestyle="-.",
    marker="o",
    ylim=(-4, 255),
    figsize=(4, 6),
)
plt.grid(axis="y", linewidth=0.5)

# Pero si ploteo los 28 numeros que van de 28*4 a 28*5, se ve que la intensidad es 0 (es decir, es la columna 4)


# %%######
# Renombro los nombres de las variables:
data = data.rename(
    columns={28 * i + j: f"col_{i+1}_row_{j}" for i in range(28) for j in range(1, 29)}
)

# %%######
# Veamos por ejemplo, la maxima intensidad de cada pixel (para cualquier letra)
maximos = data.max()
len(maximos[maximos == 0])
print(
    f"Hay {len(maximos[maximos == 0])} pixeles que no tienen intensidad en ninguna letra"
)

plot_maximos = fx.show_image(
    maximos.values, title="Maxima Intensidad de cada Pixel\n(entre todas las letras)"
)

# Todas las filas y columnas tienen al menos un pixel con intensidad
# Ninguna fila o columna tiene todos los pixeles en 0
# (aunque es muy probable que las de los bordes tengan un promedio de intensidad bajisimo)

# %%######
# Idem para la intensidad media
medias = data.mean()
len(medias[medias < 1])
print(f"Hay {len(medias[medias < 1])} pixeles con intensidad media menor a 1")

plot_medias = fx.show_image(
    medias.values,
    title="Intensidad Media de cada Pixel\n(entre todas las letras)",
    vmax=255,
)

# Viendo el grafico, practicamente se podrian descartar las 2 primeras y ultimas filas y columnas

# %%######
# Hago lo mismo para la mediana
medianas = data.median()
len(medianas[medianas < 1])
print(f"Hay {len(medianas[medianas < 1])} pixeles con mediana menor a 1")

plot_medianas = fx.show_image(
    medianas.values,
    title="Mediana de la Intensidad de cada Pixel\n(entre todas las letras)",
    vmax=255,
)

# Todas las letras tienen más pixeles vacíos que ocupados, tiene sentido
# que la mediana de menor a la media, ya que los valores altos van a pesar más.

# %%######
# Veamos cuales son los pixeles con mayor intensidad media
top_medianas = medianas.sort_values(ascending=False).head(10)
top_medianas = top_medianas.rename(
    index={
        k: k.replace("col_", "Col: ").replace("_row_", "\nFila: ")
        for k in top_medianas.index
    }
)

top_medianas.plot(
    kind="bar",
    xlabel="Pixel",
    ylabel="Intensidad Mediana",
    title="Top 15 de Intensidad Mediana de los Pixeles",
    rot=0,
)

# Se ve que los principales son los de la parte central de las letras
# (columna y fila entre 13 y 15). La "excepcion" pareciera ser col 8 y fila 20

# %%######
## Comparaciones entre letras
# Comparemos medianas entre L, E, M y I

medias_por_letra = []
medianas_por_letra = []
for letra in ["L", "E", "M", "I"]:
    letra_data = data.loc[letra]
    media_letra = letra_data.mean()
    mediana_letra = letra_data.median()
    media_letra = media_letra.rename(
        index={
            k: k.replace("col_", "Col: ").replace("_row_", "\nFila: ")
            for k in media_letra.index
        }
    )
    mediana_letra = mediana_letra.rename(
        index={
            k: k.replace("col_", "Col: ").replace("_row_", "\nFila: ")
            for k in mediana_letra.index
        }
    )
    medias_por_letra.append(media_letra)
    medianas_por_letra.append(mediana_letra)

# %%
for i in range(len(medias_por_letra)):
    fx.show_image(
        medias_por_letra[i].values,
        title="Media de la Intensidad de cada Pixel - " + ["L", "E", "M", "I"][i],
        vmax=255,
    )
    fx.show_image(
        medianas_por_letra[i].values,
        title="Mediana de la Intensidad de cada Pixel - " + ["L", "E", "M", "I"][i],
        vmax=255,
    )


# La de la L es muy parecida a la de la I, pero con las puntas mas dispersas

# Llama la atencion que el grafico de la e sea parecido al de la mediana general,
# pero mas comprimido

# %%###
# # Veamos histogramas para algunos pixeles
# Por ejemplo, el pixel 14,14
data["col_20_row_5"].hist(bins=20)
plt.axvline(data["col_20_row_5"].mean(), color="k", linestyle="dashed", linewidth=1)
plt.axvline(data["col_20_row_5"].median(), color="r", linestyle="dashed", linewidth=1)

# %%######
# Para analizar la dispersion de alguna de las clases, veamos el desvio estandar
C = data.loc["C"]
C_std = C.std()
c_plot = fx.show_image(
    C_std.values, title="Desvio de la Intensidad de cada Pixel - C", vmax=255
)

# El desvio en una letra "facil" como la C esta bastante concentrado en la forma de la letra

# %%######
T = data.loc["T"]
T_std = T.std()
t_plot = fx.show_image(
    T_std.values, title="Desvio de la Intensidad de cada Pixel - T", vmax=255
)

# El desvio aca se reparte mucho mas, ya que la T tiene una forma mas irregular

# %%######
A = data.loc["A"]
A_std = A.std()
a_plot = fx.show_image(
    A_std.values, title="Desvio de la Intensidad de cada Pixel - A", vmax=255
)

# idem con la a

#%%
# Eliminamos variables no necesarias
del A, a_plot, A_std, C, c_plot, C_std, i, img, letra, letra_data, maximos, media_letra, mediana_letra, medianas
del medianas_por_letra, medias, medias_por_letra, plot_maximos, plot_medianas, plot_medias, T, t_plot, T_std, top_medianas

# %%######################
### Clasificacion de L y A
# Conservo solo L y A
data_LA = data.loc[["L", "A"]]
# Ya se que esta balanceado por letra, asi que lo va a estar para 2 letras


# %%######
### Separo en conjunto de train y de test

X = data_LA.reset_index(drop=True)
y = data_LA.index.to_series()

y = (y == 'L') #Si es L vale 1, si es A vale 0

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3
)  # 70% para train y 30% para test

#%%
### Observamos la media de los datos
### Observo la distribución de las medias de los pixeles de cada letra
fx.show_image(data_LA.mean().values,
              title="Media de la Intensidad de cada pixel - Letras L y A",
              vmax=255)

#%%
### Observamos las medias de cada letra por separado

fx.show_image(data_LA.loc['L'].mean().values,
              title="Media de la Intensidad de cada pixel - Letra L",
              vmax=255)

fx.show_image(data_LA.loc['A'].mean().values,
              title="Media de la Intensidad de cada pixel - Letra A",
              vmax=255)

### Notamos que las medias son bastante distintas

#%%
### Observamos los pixeles de diferencia entre cada letra
difference = data_LA.loc['L'].mean().values - data_LA.loc['A'].mean().values
difference[difference<0] = 0

fx.show_image(difference,
              title="Pixeles con media en L muy superior a media en A",
              vmax=255)

difference = data_LA.loc['L'].mean().values - data_LA.loc['A'].mean().values
difference = -difference
difference[difference<0] = 0

fx.show_image(difference,
              title="Pixeles con media en A muy superior a media en L",
              vmax=255)

### Los pixeles de mayor intensidad son aquellos donde las diferencias son mayores

#%%
# Preparamos el dataFrame con los resultados
eval_ej2 = pd.DataFrame(columns=['Atributos','N_neighbors','Accuracy','Matriz confusión', 'Precisión', 'Recall', 'F1'])

#%%
# Elegimos 3 atributos para ajustar un modelo de KNN basándonos en distintos atributos
# Probamos con pixeles donde las medias no son distintivas para ninguna letra
atr = ['col_1_row_1','col_1_row_28','col_28_row_28']
caso = "Pixeles no distintivos"

eleccion = data_LA.mean()

for pixel in atr:
    eleccion[pixel] = 255

fx.show_image(eleccion.values,
              title="Eleccion (amarillo) sobre media general",
              vmax=255)


#%%
# Ajusto el modelo de KNN
vecinos = 5
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

print("Matriz confusión binaria:")

tp, tn, fp, fn = fx.matriz_confusion_binaria(Y_test, Y_pred)
matrx = np.array([[tp,fn],[fp,tn]])
print(matrx)

acc = fx.accuracy_score(tp, tn, fp, fn)
print("Exactitud del modelo:", acc)

# Como predice todo como lo mismo no calculamos precision, recall y f1
eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, 'N/A', 'N/A', 'N/A']

#%%
# Probamos aumentando el numero de vecinos
vecinos = 15
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

print("Matriz confusión binaria:")

tp, tn, fp, fn = fx.matriz_confusion_binaria(Y_test, Y_pred)
matrx = np.array([[tp,fn],[fp,tn]])
print(matrx)

acc = fx.accuracy_score(tp, tn, fp, fn)
print("Exactitud del modelo:", acc)
#matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)
eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, 'N/A', 'N/A', 'N/A']

# No hubo cambio en las predicciones

#%% #########
# Elegimos pixeles cuya media en A sea muy superior
atr = ['col_10_row_15','col_20_row_16','col_20_row_8']
caso = "Media en A muy superior"

eleccion = data_LA.loc['A'].mean() - data_LA.loc['L'].mean()

for pixel in atr:
    eleccion[pixel] = 255

eleccion[eleccion<0] = 0

fx.show_image(eleccion.values,
              title="Elección (amarillo) sobre mayor intensidad en A",
              vmax=255)

#%%
# Ajusto el modelo de KNN
vecinos = 5
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Cambiamos el numero de vecinos
vecinos = 15
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Probamos con pixeles cuya media en L sea muy superior
atr = ['col_13_row_23','col_12_row_22','col_13_row_21']
caso = "Media en L muy superior"

eleccion = data_LA.loc['L'].mean() - data_LA.loc['A'].mean()

for pixel in atr:
    eleccion[pixel] = 255

eleccion[eleccion<0] = 0

fx.show_image(eleccion.values,
              title="Elección (amarillo) sobre mayor intensidad en L",
              vmax=255)

#%%
# Ajusto el modelo de KNN
vecinos = 5
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Cambiamos el numero de vecinos
vecinos = 15
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Probamos con pixeles donde no hay una gran diferencia de intensidad
# Estos serán donde la media general sea mayor
atr = ['col_15_row_15','col_8_row_20','col_16_row_8']
caso = "No hay gran diferencia en medias"

eleccion = data_LA.mean()

for pixel in atr:
    eleccion[pixel] = 255

eleccion[eleccion<0] = 0

fx.show_image(eleccion.values,
              title="Elección (amarillo) sobre media general",
              vmax=255)

#%%
# Ajusto el modelo de KNN
vecinos = 5
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Cambiamos el numero de vecinos
vecinos = 15
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Aumentemos el numero de atributos a 6
# Probamos con los pixeles no significativos ya elegidos y los representativos de la letra A
atr = ['col_10_row_15','col_20_row_16','col_20_row_8',
       'col_1_row_1','col_1_row_28','col_28_row_28']
caso = "Media muy superior en A + No significativos"

eleccion = data_LA.loc['A'].mean() - data_LA.loc['L'].mean()

for pixel in atr:
    eleccion[pixel] = 255

eleccion[eleccion<0] = 0

fx.show_image(eleccion.values,
              title="Elección (amarillo) sobre mayor intensidad en A",
              vmax=255)

#%%
# Ajusto el modelo de KNN
vecinos = 5
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Cambiamos el numero de vecinos
vecinos = 15
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Probamos con los pixeles elegidos para representar una media de A superior y una de L superior
atr = ['col_10_row_15','col_20_row_16','col_20_row_8',
       'col_13_row_23','col_12_row_22','col_13_row_21']
caso = "Media en A muy superior + Media en L muy superior"

eleccion = data_LA.mean()

for pixel in atr:
    eleccion[pixel] = 255

eleccion[eleccion<0] = 0

fx.show_image(eleccion.values,
              title="Elección (amarillo) sobre media general",
              vmax=255)

#%%
# Ajusto el modelo de KNN
vecinos = 5
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Cambiamos el numero de vecinos
vecinos = 15
model = KNeighborsClassifier(n_neighbors = vecinos)
model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos

matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)

eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]

#%%
# Elimino variables no relevantes para los resultados
del matrx, acc, prec, rcll, f1, atr, vecinos, model, difference, eleccion, tp, tn, fp, fn, pixel, caso
del Y_test, Y_pred, Y_train, X_test, X_train, X, y

# %%######
# EJERCICIO 3
letras = ["A", "E", "I", "O", "U"]

data_ej3 = data.loc[letras]
# Defino el conjunto de test y train
X = data_ej3.reset_index(drop=True)
y = data_ej3.index.to_series().reset_index(drop=True)

# Divido en conjunto de entrenamiento y de test, con el tamaño de test del 30%
X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, random_state=1, test_size=0.3)

alturas = [i for i in range(3, 15)]  # Modificar con alturas de interés
nsplits = 5
kf = KFold(n_splits=nsplits)

# Creo la matriz para comparar accuracy donde cada fila es un fold y cada columna un modelo
res_acc_gini = np.zeros((nsplits, len(alturas)))
res_acc_entrpy = np.zeros((nsplits, len(alturas)))

# Creo una matriz para comparar precision y una para comparar recall donde cada fila
# es un fold, cada columna un modelo y cada profundidad es una vocal
res_prec_gini = np.zeros((nsplits, len(alturas), 5))
res_rcll_gini = np.zeros((nsplits, len(alturas), 5))
res_prec_entrpy = np.zeros((nsplits, len(alturas), 5))
res_rcll_entrpy = np.zeros((nsplits, len(alturas), 5))

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    for j, hmax in enumerate(alturas):

        arbol = tree.DecisionTreeClassifier(criterion='gini', max_depth=hmax)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)

        res_acc_gini[i, j] = metrics.accuracy_score(kf_y_test, pred)

        matrx_conf_train = fx.matriz_conf_bin_multiclass(letras, kf_y_test, pred)
        for k in range(5):
            res_prec_gini[i, j, k] = fx.precision_score_multiclass(
                letras[k], matrx_conf_train
            )
            res_rcll_gini[i, j, k] = fx.recall_score_multiclass(letras[k], matrx_conf_train)
            
        arbol = tree.DecisionTreeClassifier(criterion='entropy', max_depth=hmax)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)

        res_acc_entrpy[i, j] = metrics.accuracy_score(kf_y_test, pred)

        matrx_conf = fx.matriz_conf_bin_multiclass(letras, kf_y_test, pred)
        for k in range(5):
            res_prec_entrpy[i, j, k] = fx.precision_score_multiclass(
                letras[k], matrx_conf_train
            )
            res_rcll_entrpy[i, j, k] = fx.recall_score_multiclass(letras[k], matrx_conf)


acc_gini = res_acc_gini.mean(axis=0)
prec_gini = res_prec_gini.mean(axis=0)  # cada letra es una columna, cada fila un modelo
rcll_gini = res_rcll_gini.mean(axis=0)  # cada letra es una columna, cada fila un modelo

acc_entrpy = res_acc_entrpy.mean(axis=0)
prec_entrpy = res_prec_entrpy.mean(axis=0)  # cada letra es una columna, cada fila un modelo
rcll_entrpy = res_rcll_entrpy.mean(axis=0)  # cada letra es una columna, cada fila un modelo

#%% ######
# Comparo arboles bajo criterio Gini

# Cada fila es el mejor arbol de cada criterio, cada columna los factores de comparacion
# Comparo con la accuracy, el promedio de precision de todas las letras y el promedio de recall
best_trees = np.zeros((2,4))

print("Analizamos el criterio Gini")

metricas_gini = pd.DataFrame()

metricas_gini["altura"] = alturas
metricas_gini["accuracy"] = acc_gini
metricas_gini["criterio"] = "Gini"

for j, hmax in enumerate(alturas):

    for i in range(len(letras)):
        prec_column = "prec_" + letras[i]
        rcll_column = "rcll_" + letras[i]
        metricas_gini.loc[j, prec_column] = prec_gini[j, i]
        metricas_gini.loc[j, rcll_column] = rcll_gini[j, i]

prom_acc_g = acc_gini # Lo llamo promedio para mantener estructura en el codigo
prom_prec_g = prec_gini.mean(axis=1)
prom_rcll_g = rcll_gini.mean(axis=1)

metricas_gini = metricas_gini.reindex(sorted(metricas_gini.columns), axis=1)
metricas_gini.insert(2,'accuracy', metricas_gini.pop('accuracy'))

puntaje = np.zeros((3, len(alturas)))  # 1 fila por métrica, 1 columna por altura a analizar

print("Basandonos en su accuracy, los mejores arboles son:")
order = np.argsort(-prom_acc_g)
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
    puntaje[0, i] += order.argsort()[i]
res = res[:-2]
print(res + "\n")


print("Basandonos en su precision promedio, los mejores arboles son:")
order = np.argsort(-prom_prec_g)
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
    puntaje[1, i] += order.argsort()[i]
res = res[:-2]
print(res + "\n")


print("Basandonos en su recall promedio, los mejores arboles son:")
order = np.argsort(-prom_rcll_g)
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
    puntaje[2, i] += order.argsort()[i]
res = res[:-2]
print(res + "\n")


# Decido el mejor como el que en promedio de la posicion en el top tiene mejor calidad
puntaje = puntaje.mean(axis=0)
mejor_pos = np.argsort(puntaje)[0]

# Propiedades del mejor Gini
best_trees[:,0] = alturas[mejor_pos]
best_trees[0,1:] = [prom_acc_g[mejor_pos],
                   prom_prec_g[mejor_pos],
                   prom_rcll_g[mejor_pos]]

print(f"La mejor altura con criterio Gini basándonos en las 3 métricas es la altura {alturas[mejor_pos]}")

#%% ####
# Comparo los arboles de criterio Entropy

print("\nAnalizamos el criterio Entropy")

metricas_entrpy = pd.DataFrame()

metricas_entrpy["altura"] = alturas
metricas_entrpy["accuracy"] = acc_entrpy
metricas_entrpy["criterio"] = "Entropy"

for j, hmax in enumerate(alturas):

    for i in range(len(letras)):
        prec_column = "prec_" + letras[i]
        rcll_column = "rcll_" + letras[i]
        metricas_entrpy.loc[j, prec_column] = prec_entrpy[j, i]
        metricas_entrpy.loc[j, rcll_column] = rcll_entrpy[j, i]

prom_acc_e = acc_entrpy # Lo llamo promedio para mantener estructura en el codigo
prom_prec_e = prec_entrpy.mean(axis=1)
prom_rcll_e = rcll_entrpy.mean(axis=1)

metricas_entrpy = metricas_entrpy.reindex(sorted(metricas_entrpy.columns), axis=1)
metricas_entrpy.insert(2,'accuracy', metricas_entrpy.pop('accuracy'))

puntaje = np.zeros((3, len(alturas)))  # 1 fila por métrica, 1 columna por altura a analizar

print("Basandonos en su accuracy, los mejores arboles son:")
order = np.argsort(-prom_acc_e)
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
    puntaje[0, i] += order.argsort()[i]
res = res[:-2]
print(res + "\n")


print("Basandonos en su precision promedio, los mejores arboles son:")
order = np.argsort(-prom_prec_e)
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
    puntaje[1, i] += order.argsort()[i]
res = res[:-2]
print(res + "\n")


print("Basandonos en su recall promedio, los mejores arboles son:")
order = np.argsort(-prom_rcll_e)
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
    puntaje[2, i] += order.argsort()[i]
res = res[:-2]
print(res + "\n")


# Decido el mejor como el que en promedio de la posicion en el top predice mejor
puntaje = puntaje.mean(axis=0)
mejor_pos = np.argsort(puntaje)[0]

# Propiedades del mejor Entropy
best_trees[:,0] = alturas[mejor_pos]
best_trees[1,1:] = [prom_acc_e[mejor_pos],
                   prom_prec_e[mejor_pos],
                   prom_rcll_e[mejor_pos]]

print(f"La mejor altura con criterio Entropy basándonos en las 3 métricas es la altura {alturas[mejor_pos]}")

metricas_train = pd.concat([metricas_gini, metricas_entrpy])

#%% #####
# Comparo los mejores arboles de cada criterio

puntaje = np.zeros(2)

for i in range(1,4):
    puntaje[(np.argsort(-best_trees[:,i])[0])] += 1
mejor_pos = np.argsort(-puntaje)[0]
    
criterios = ["gini", "entropy"]

best_hyperparam = {}
best_hyperparam['criterio'] = criterios[mejor_pos]
best_hyperparam['altura'] = int(best_trees[mejor_pos,0])

print('Los mejores hiperparametros fueron:')
print('Criterio:', best_hyperparam['criterio'])
print('Altura:', best_hyperparam['altura'])

#%% ####
# Ajusto el arbol bajo los mejores hiperparámetros

arbol = tree.DecisionTreeClassifier(criterion = best_hyperparam['criterio'],
                                    max_depth= best_hyperparam['altura'])

arbol.fit(X_dev, y_dev)
pred = arbol.predict(X_eval)

#%% ####
# Analizo su performance bajo todas las medidas
print('Analisis de performance del arbol elegido:\n')

performance = pd.Series(dtype=object)

performance['Accuracy'] = metrics.accuracy_score(y_eval, pred)
performance['Promedio de precision'] = 0
performance['Promedio de recall'] = 0

print('Accuracy: ' + str(performance['Accuracy']) + '\n')

matrx_conf_test = fx.matriz_conf_bin_multiclass(letras, y_eval, pred)
for k in range(5):
    prec_column = "Precision letra " + letras[k]
    performance[prec_column] = fx.precision_score_multiclass(letras[k], matrx_conf_test)
    performance['Promedio de precision'] += performance[prec_column]
    
    rcll_column = "Recall letra " + letras[k]
    performance[rcll_column] = fx.recall_score_multiclass(letras[k], matrx_conf_test)
    performance['Promedio de recall'] += performance[rcll_column]
    
    print('Letra ' + letras[k] + ":")
    print('Precision =', performance[prec_column])
    print('Recall    =', performance[rcll_column], '\n')

performance['Promedio de precision'] /= 5
performance['Promedio de recall'] /= 5

print('Promedio de precision:', performance['Promedio de precision'])
print('Promedio de recall:', performance['Promedio de recall'], '\n')

