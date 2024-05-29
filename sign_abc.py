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


# %%######
### Explore Data
data = pd.read_csv("emnist_letters_tp.csv", header=None, index_col=0)

# data = data / sum(data. max(axis='index')) # Normalizo los datos
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

plt.plot(img[4])
plt.xlabel("Columna")
plt.ylabel("Intensidad")
plt.title("Intensidad de la Fila 4 de una Y (id: 26)")

# Se ve que tiene dos picos: o sea, img[4] representa una fila

# %%######
data.iloc[26][28 * 4 : 28 * 5].plot(
    xlabel="Fila",
    ylabel="Intensidad",
    title="Intensidad de la Fila 4 de una Y (id: 26)",
)
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

# L
L = data.loc["L"]
L_media = L.mean()
L_media = L_media.rename(
    index={
        k: k.replace("col_", "Col: ").replace("_row_", "\nFila: ")
        for k in L_media.index
    }
)

# E
E = data.loc["E"]
E_media = E.mean()
E_media = E_media.rename(
    index={
        k: k.replace("col_", "Col: ").replace("_row_", "\nFila: ")
        for k in E_media.index
    }
)

# M
M = data.loc["M"]
M_media = M.mean()
M_media = M_media.rename(
    index={
        k: k.replace("col_", "Col: ").replace("_row_", "\nFila: ")
        for k in M_media.index
    }
)

# N
I = data.loc["I"]
I_media = I.mean()
I_media = I_media.rename(
    index={
        k: k.replace("col_", "Col: ").replace("_row_", "\nFila: ")
        for k in I_media.index
    }
)


# %%
medias_por_letra = [L_media, I_media, M_media, E_media]
for i, media in enumerate(medias_por_letra):
    fx.show_image(
        medias_por_letra[i].values,
        title="Mediana de la Intensidad de cada Pixel - " + ["L", "N", "M", "E"][i],
        vmax=255,
    )


# La de la L es muy parecida a la de la I, pero con las puntas mas dispersas

# Llama la atencion que el grafico de la e sea parecido al de la mediana general,
# pero mas comprimido

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

# %%######################
### Clasificacion de L y A
# Conservo solo L y A
data_LA = data.loc[["L", "A"]]
# Ya se que esta balanceado por letra, asi que lo va a estar para 2 letras


# %%######
X = data_LA.reset_index(drop=True)
y = data_LA.index

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3
)  # 70% para train y 30% para test

# %%######
# EJERCICIO 3
letras = ["A","E","I","O","U"]

data_ej3 = data.loc[letras]
# Defino el conjunto de test y train
X = data_ej3.reset_index(drop=True)
y = data_ej3.index.to_series().reset_index(drop=True)

# Divido en conjunto de entrenamiento y de test, con el tamaño de test del 20%
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,random_state=1,test_size=0.2)

alturas = [i for i in range(3,15)] #Modificar con alturas de interés
nsplits = 5
kf = KFold(n_splits=nsplits)

# Creo la matriz para comparar accuracy donde cada fila es un fold y cada columna un modelo
res_acc = np.zeros((nsplits, len(alturas)))

# Creo una matriz para comparar precision y una para comparar recall donde cada fila 
# es un fold, cada columna un modelo y cada profundidad es una vocal
res_prec = np.zeros((nsplits, len(alturas), 5))
res_rcll = np.zeros((nsplits, len(alturas), 5))

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, hmax in enumerate(alturas):
        
        arbol = tree.DecisionTreeClassifier(max_depth = hmax)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        
        res_acc[i,j] = metrics.accuracy_score(kf_y_test,pred)
        
        matrx_conf_train = fx.matriz_conf_bin_multiclass(letras, kf_y_test, pred)
        for k in range(5):
            res_prec[i,j,k] = fx.precision_score_multiclass(letras[k], matrx_conf_train)
            res_rcll[i,j,k] = fx.recall_score_multiclass(letras[k], matrx_conf_train)


calidad_acc = res_acc.mean(axis=0)
calidad_prec = res_prec.mean(axis=0) #cada letra es una columna, cada fila un modelo
calidad_rcll = res_rcll.mean(axis=0) #cada letra es una columna, cada fila un modelo


#%%
# Comparo arboles en caso de train
resultados_train = pd.DataFrame()

resultados_train['altura'] = alturas
resultados_train['accuracy'] = calidad_acc

for j, hmax in enumerate(alturas):
    
    for i in range(len(letras)):
        prec_column = "prec_" + letras[i]
        rcll_column = "rcll_" + letras[i]
        resultados_train.loc[j,prec_column] = calidad_prec[j,i]
        resultados_train.loc[j,rcll_column] = calidad_rcll[j,i]

prom_acc_train = calidad_acc
prom_prec_train = calidad_prec.mean(axis=1)
prom_rcll_train = calidad_rcll.mean(axis=1)

resultados_train = resultados_train.reindex(sorted(resultados_train.columns), axis=1)

# Me quedo con que sean el top 3 bajo cualquier criterio
mejores_arboles = []

print('Sobre el caso de train:\n')

print('Basandonos en su accuracy, los mejores arboles son:')
order = np.argsort(-prom_acc_train)

for i in range(3):
    if alturas[order[i]] not in mejores_arboles:
        mejores_arboles.append(alturas[order[i]])

res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
res = res[:-2]
print(res + '\n')


print('Basandonos en su precision promedio, los mejores arboles son:')
order = np.argsort(-prom_prec_train)

for i in range(3):
    if alturas[order[i]] not in mejores_arboles:
        mejores_arboles.append(alturas[order[i]])
        
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
res = res[:-2]
print(res + '\n')


print('Basandonos en su recall promedio, los mejores arboles son:')
order = np.argsort(-prom_rcll_train)

for i in range(3):
    if alturas[order[i]] not in mejores_arboles:
        mejores_arboles.append(alturas[order[i]])

res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
res = res[:-2]
print(res + '\n')
    
#%% Comparo arboles en caso de test
#Elijo las alturas a analizar segun ejercicio anterior
mejores_arboles.sort()

alturas = mejores_arboles

calidad_acc = np.zeros(len(alturas))
calidad_prec = np.zeros((len(alturas),len(letras))) #Fila por modelo, columna por letra
calidad_rcll = np.zeros((len(alturas),len(letras))) #Fila por modelo, columna por letra
    
for i, hmax in enumerate(alturas):
      
    arbol = tree.DecisionTreeClassifier(max_depth = hmax)
    arbol.fit(X_dev, y_dev)
    pred = arbol.predict(X_eval)
      
    calidad_acc[i] = metrics.accuracy_score(y_eval,pred)
      
    matrx_conf_test = fx.matriz_conf_bin_multiclass(letras, y_eval, pred)
    for k in range(5):
        calidad_prec[i,k] = fx.precision_score_multiclass(letras[k], matrx_conf_test)
        calidad_rcll[i,k] = fx.recall_score_multiclass(letras[k], matrx_conf_test)  


resultados_test = pd.DataFrame()

resultados_test['altura'] = alturas
resultados_test['accuracy'] = calidad_acc

for j, hmax in enumerate(alturas):
    for i in range(len(letras)):
        prec_column = "prec_" + letras[i]
        rcll_column = "rcll_" + letras[i]
        resultados_test.loc[j,prec_column] = calidad_prec[j,i]
        resultados_test.loc[j,rcll_column] = calidad_rcll[j,i]

prom_acc_test = calidad_acc
prom_prec_test = calidad_prec.mean(axis=1)
prom_rcll_test = calidad_rcll.mean(axis=1)

resultados_test = resultados_test.reindex(sorted(resultados_test.columns), axis=1)

#Notamos que la I es la letra con mayor precision, la U suele ser la de menor
#Notamos que la I y O son las de mayor recall, la de menor recall varía


#%% Analisis sobre caso de evaluacion

print('Sobre el caso de test:\n')

puntaje = np.zeros((3,len(alturas))) #1 fila por métrica, 1 columna por altura a analizar

print('Basandonos en su accuracy, los mejores arboles son:')
order = np.argsort(-prom_acc_test)
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
    puntaje[0,i] += order.argsort()[i]
res = res[:-2]
print(res + '\n')


print('Basandonos en su precision promedio, los mejores arboles son:')
order = np.argsort(-prom_prec_test)
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
    puntaje[1,i] += order.argsort()[i]
res = res[:-2]
print(res + '\n')


print('Basandonos en su recall promedio, los mejores arboles son:')
order = np.argsort(-prom_rcll_test)
res = ""
for i, hmax in enumerate(alturas):
    res += str(alturas[order[i]]) + ", "
    puntaje[2,i] += order.argsort()[i]
res = res[:-2]
print(res + '\n')


# Decido el mejor como el que en promedio de la posicion en el top predice mejor
puntaje = puntaje.mean(axis=0)
best_tree = alturas[(np.argsort(puntaje)[0])]

print(f'La mejor altura basándonos en las 3 métricas es la altura {best_tree}')
