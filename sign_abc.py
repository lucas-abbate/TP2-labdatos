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
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns


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
    vmax=1,
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
    vmax=1,
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
        title="Media de la Intensidad de cada Pixel - " + ["L", "N", "M", "E"][i],
        vmax=255,
    )
    fx.show_image(
        medianas_por_letra[i].values,
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

# %% ...
