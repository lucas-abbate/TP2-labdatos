# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 2024

@author: Equipo 2 (ABC)

"""
# script para plotear letras

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import funciones as fx
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics, tree
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

# %%##########################################################################
#######       Seccion 1:   análisis exploratorio de datos

# Carga
data = pd.read_csv("emnist_letters_tp.csv", header=None, index_col=0)

# %%
# print(data.max(axis='columns').sort_values(ascending=True))  # Todas las obs. tienen max = 255, asi que normalizarlo es dividir x 255
# data = data / 255 # Normalizo los datos

# %%######
data.head()
print(data.index.value_counts().sort_index())
# Observamos que esta balanceado, 2400 ocurrencias de cada una de las 26 letras (alfabeto ingles)
# Index = etiqueta, el resto intensidades de pixeles
# 28x28 = 784 pixeles

# También está normalizado: todos tienen max 255 y min 0
print(data.min(axis=1).describe())
print(data.max(axis=1).describe())

# %%######
img = fx.show_image_from_data(26, data, return_array=True)  # Es una Y

# %%######

# Para el analisis exploratorio, renombro las columnas
# Quiero ver si en los datos originales van en filas o en columnas
# (o sea, si los primeros 28 son la primer fila o la primer columna)

fig, ax = plt.subplots(figsize=(4, 6))
ax.plot(img[4], "o-.", color="#D67236")
ax.set_xlabel("Posición")
ax.set_ylabel("Intensidad")
ax.set_title("Intensidad de la Fila 4 de la Fig. 2 (id: 26)")
ax.grid(axis="y", linewidth=0.5)
plt.show()

# Se ve que tiene dos picos: o sea, img[4] representa una fila

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
# Renombro las variables por filas y columnas:
data = data.rename(
    columns={28 * i + j + 1: f"col_{i+1}_row_{j+1}" for i in range(28) for j in range(28)}
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
# %%###
# # Veamos histogramas para algunos pixeles
# Por ejemplo, el pixel 13,14 (uno de los que tienen mayor intensidad mediana)
for pixel in ["col_20_row_5", "col_13_row_14", "col_6_row_8", "col_15_row_10"]:
    col = pixel.split('_')[1]
    row = pixel.split('_')[3]
    data[pixel].plot(kind='hist', density=False, bins=20, color="#D67236", edgecolor='k', 
                     title=f'Intensidad del Pixel en Col. {col} y Fila {row} (todas las letras)',
                     label="Frecuencia",
                     weights = np.ones_like(data.index) / len(data.index))
    plt.axvline(data[pixel].mean(), color="k", linestyle="dashed", linewidth=1, label="Media")
    plt.axvline(data[pixel].median(), color="blue", linestyle="dashed", linewidth=1, label='Mediana')
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia Relativa")
    plt.ylim(0, 0.75)
    plt.legend()
    plt.show()

# %%######
# Veamos cuales son los pixeles con mayor intensidad media
top_medianas = medianas.sort_values(ascending=False).head(10)
top_medianas = top_medianas.rename(
    index={
        k: k.replace("col_", "C ").replace("_row_", "\nF ")
        for k in top_medianas.index
    }
)

top_medianas.plot(
    kind="bar",
    xlabel="Pixel",
    ylabel="Intensidad Mediana",
    title="Top 10 de Intensidad Mediana de los Pixeles",
    rot=0,
    color="#D67236",
    edgecolor='black'
)

# Se ve que los principales son los de la parte central de las letras
# (columna y fila entre 13 y 15). La "excepcion" pareciera ser col 8 y fila 20

# %%######
## Comparaciones entre letras
# Comparemos medianas entre L, E, M y I

letras_comp = ["L", "E", "M", "I"]
medias_por_letra = []
medianas_por_letra = []
for letra in letras_comp:
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
        title="Media de la Intensidad de cada Pixel - " + letras_comp[i],
        vmax=255,
    )
    fx.show_image(
        medianas_por_letra[i].values,
        title="Mediana de la Intensidad de cada Pixel - " + letras_comp[i],
        vmax=255,
    )


# La de la L es muy parecida a la de la I, pero con las puntas mas dispersas

# Llama la atencion que el grafico de la e sea parecido al de la mediana general,
# pero mas comprimido

# %%###
# Para estimar la dificultad de diferenciar 2 letras calculamos la diferencia de sus "firmas" en media y mediana

dif_media_EL = medias_por_letra[1] - medias_por_letra[0]
fx.show_image(dif_media_EL.values, title="Diferencia de la Media entre 'E' y 'L'", vmax=255, vmin=-255)

dif_mediana_EL = medianas_por_letra[1] - medianas_por_letra[0]
fx.show_image(dif_mediana_EL.values, title="Diferencia de la Mediana entre 'E' y 'L'", vmax=255, vmin=-255)

dif_media_EM = medias_por_letra[1] - medias_por_letra[2]
fx.show_image(dif_media_EM.values, title="Diferencia de la Media entre 'E' y 'M'", vmax=255, vmin=-255)

dif_mediana_EM = medianas_por_letra[1] - medianas_por_letra[2]
fx.show_image(dif_mediana_EM.values, title="Diferencia de la Mediana entre 'E' y 'M'", vmax=255, vmin=-255)

dif_mediana_LI = medianas_por_letra[0] - medianas_por_letra[3]
fx.show_image(dif_mediana_LI.values, title="Diferencia de la Mediana entre 'L' y 'I'", vmax=255, vmin=-255)# %%###
dif_media_LI = medias_por_letra[0] - medias_por_letra[3]
fx.show_image(dif_media_LI.values, title="Diferencia de la Media entre 'L' y 'I'", vmax=255, vmin=-255)


dif_mediana_EI = medianas_por_letra[1] - medianas_por_letra[3]
fx.show_image(dif_mediana_EI.values, title="Diferencia de la Mediana entre 'E' y 'I'", vmax=255, vmin=-255)

dif_media_EI = medias_por_letra[1] - medias_por_letra[3]
fx.show_image(dif_media_EI.values, title="Diferencia de la Media entre 'E' y 'I'", vmax=255, vmin=-255)

### ¿queremos mostrar las diferencias de las medias?
### creo que conviene solo mostrar que las diferencias de medianas son mas dispersas

print(f"Desvío de la diferencia de medias entre 'E' y 'L' = {dif_media_EL.std()}")
print(f"Desvío de la diferencia de medianas entre 'E' y 'L' = {dif_mediana_EL.std()}")
print(f"Desvío de la diferencia de medias entre 'E' y 'M' = {dif_media_EM.std()}")
print(f"Desvío de la diferencia de medianas entre 'E' y 'M' = {dif_mediana_EM.std()}")
print(f"Desvío de la diferencia de medias entre 'I' y 'L' = {dif_media_LI.std()}")
print(f"Desvío de la diferencia de medianas entre 'I' y 'L' = {dif_mediana_LI.std()}")
print(f"Desvío de la diferencia de medias entre 'E' y 'I' = {dif_media_EI.std()}")
print(f"Desvío de la diferencia de medianas entre 'E' y 'I' = {dif_mediana_EI.std()}")# Con ambos criterios (media y mediana) pacece a priori m
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

#%%
I = data.loc["I"]
I_std = I.std()
I_plot = fx.show_image(
    I_std.values, title="Desvio de la Intensidad de cada Pixel - I", vmax=255
)

# %%##########################################################################
#######       Seccion 2: Clasificacion binaria de L y A con modelos KNN

# Conservo solo L y A
data_LA = data.loc[["L", "A"]]
# Ya se que esta balanceado por letra, asi que lo va a estar para 2 letras

# %%######
### Separo en conjunto de train y de test

X = data_LA.reset_index(drop=True)
y = data_LA.index.to_series()

y = (y == 'L') #Si es L vale 1, si es A vale 0

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3,
    random_state=1
)  # 70% para train y 30% para test, fijamos la semilla para reproducibilidad

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
# Probamos con pixeles donde hay una gran diferencia de intensidad
# Donde la diferencia absoluta de medias o medianas (probaremos ambos criterios) sea mayor

mean_dif_LA = data_LA.loc['L'].mean()-data_LA.loc['A'].mean()
median_dif_LA = data_LA.loc['L'].median()-data_LA.loc['A'].median()

fx.show_image(mean_dif_LA.values, title="Diferencia de medias de 'L' y 'A'", vmax=255, vmin=-255)
fx.show_image(median_dif_LA.values, title="Diferencia de medianas de 'L' y 'A'", vmax=255, vmin=-255)

#%%
top_mean_dif = mean_dif_LA.abs().sort_values(ascending=False).head(10)
top_mean_dif_labels = top_mean_dif.rename(
    index={
        k: k.replace("col_", "C ").replace("_row_", "\nF ")
        for k in top_mean_dif.index
    }
)

top_mean_dif_labels.plot(
    kind="bar",
    xlabel="Pixel",
    ylabel="Intensidad Media",
    title="Top 10 de Diferencia de Media de los Pixeles de 'L' y 'A'",
    rot=0,
    ylim=(0, 255)
)

#%%
top_median_dif = median_dif_LA.abs().sort_values(ascending=False).head(10)
top_median_dif_labels = top_median_dif.rename(
    index={
        k: k.replace("col_", "C ").replace("_row_", "\nF ")
        for k in top_median_dif.index
    }
)

top_median_dif_labels.plot(
    kind="bar",
    xlabel="Pixel",
    ylabel="Intensidad Mediana",
    title="Top 10 de Diferencia de Mediana de los Pixeles de 'L' y 'A'",
    rot=0,
    ylim=(0, 255)
)

#%%
# Preparamos el dataFrame con los resultados
eval_ej2 = pd.DataFrame(columns=['Atributos','N_neighbors','Accuracy','Matriz confusión', 'Precisión', 'Recall', 'F1'])

#%%
# Elegimos atributos para ajustar un modelo de KNN basándonos en distintos atributos
# Primero 3 atributos con distintos criterios y distinta cantidad de vecinos
# Finalmente con mas atributos con criterios similares

atr_x_caso = [
    ['col_1_row_1','col_1_row_28','col_28_row_28', ],
    ['col_10_row_15','col_20_row_16','col_20_row_8'],
    ['col_13_row_23','col_12_row_22','col_13_row_21'],
    ['col_15_row_15','col_8_row_20','col_16_row_8'],
    list(top_mean_dif.head(3).index),
    list(top_median_dif.head(3).index),
    ['col_10_row_15','col_20_row_16','col_20_row_8',
     'col_13_row_23','col_12_row_22','col_13_row_21'],
    list(top_mean_dif.head(6).index),
    list(top_mean_dif.head(10).index),
    list(top_median_dif.head(6).index),
    list(top_median_dif.head(10).index),
    list(top_mean_dif.head(5).index)+list(top_median_dif.head(8).index), # son 10 distintos
    ['col_20_row_15', 'col_20_row_16', 'col_19_row_17', 'col_19_row_16', 'col_14_row_19',
     'col_16_row_11', 'col_13_row_19', 'col_10_row_17', 'col_10_row_16', 'col_11_row_15']
    ]
casos = [
    "Pixeles no distintivos - Hand-picked", # pixeles donde las medias no son distintivas para ninguna letra
    "Media en A muy superior - Hand-picked", # Pixeles cuya media en A sea muy superior
    "Media en L muy superior - Hand-picked", # Pixeles cuya media en L sea muy superior
    "No hay gran diferencia en medias - Hand-picked", # Donde la media general sea mayor
    "Top 3 dif. de medias", # Donde la diferencia absoluta de medias sea mayor
    "Top 3 dif. de medianas", # Donde la diferencia absoluta de medianas sea mayor
    "Media en A muy superior + Media en L muy superior - Hand-picked", # Probamos con los pixeles 
                    # elegidos para representar una media de A superior y una de L superior
    "Top 6 dif. de medias",
    "Top 10 dif. de medias",
    "Top 6 dif. de medianas",
    "Top 10 dif. de medianas",
    "Top 5 dif. de medias + Top 5 dif. de medianas", # 5 top medias 
                                        #+ 5 top medianas distintas de las anteriores
    "10 atributos Hand-picked de distintas regiones"
    ]

#%% Ajustamos y evaluamos KNN con cada set de atributos y para 5 y 15 vecinos

for atr, caso in zip(atr_x_caso, casos):
    # Grafico los pixeles elegidos
    eleccion = median_dif_LA.copy()
    for pixel in atr:
        eleccion[pixel] = 255
    fx.show_image(eleccion.values,
                  title=f'Eleccion (amarillo) sobre dif. de medianas \n{caso}',
                  vmax=255, vmin=-255)
    
    for vecinos in [5, 15]:
        # Ajusto el modelo de KNN
        model = KNeighborsClassifier(n_neighbors = vecinos)
        model.fit(X_train[atr], Y_train) # entreno el modelo con los datos X e Y
        Y_pred = model.predict(X_test[atr]) # me fijo qué clases les asigna el modelo a mis datos
        
        matrx, acc, prec, rcll, f1 = fx.calidad_modelo(Y_test, Y_pred)
        
        eval_ej2.loc[len(eval_ej2)] = [caso, vecinos, acc, matrx, prec, rcll, f1]


#%% Imprimimos exactitud para cada experimento 
last_attr = ""
for _ , row in eval_ej2.iterrows():
    if last_attr != row['Atributos']:
        print()
        last_attr = row['Atributos']
        print(last_attr)
    print("Vecinos (N):", row['N_neighbors'], "\tAccuracy:", round(row['Accuracy'], 4))

# En nuestra ejecucion la mejor exactitud (0.9736) la logramos usando
# los 10 pixeles con mayor diferencia entre medias y 5 vecinos

#%%
# Guardamos los resultados en el anexo
eval_ej2 = eval_ej2.round(5) # Redondeo para que sea más legible
eval_ej2.to_csv('./Anexo/clasif_binaria_res.csv')

# %%##########################################################################
#######       Seccion 3: Clasificación multiclase de vocales con árboles de de decisión

letras = ["A", "E", "I", "O", "U"]

data_ej3 = data.loc[letras]
# Defino el conjunto de test y train
X = data_ej3.reset_index(drop=True)
y = data_ej3.index.to_series().reset_index(drop=True)

# Divido en conjunto de entrenamiento y de test, con el tamaño de test del 30%
X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, random_state=1, test_size=0.3) # Fijamos semilla

alturas = [i for i in range(3, 15)]  # Modificar con alturas de interés
nsplits = 5
kf = KFold(n_splits=nsplits)

# Creo la matriz para comparar accuracy donde cada fila es un fold y cada columna un modelo
res_acc_gini = np.zeros((nsplits, len(alturas)))
res_acc_entropy = np.zeros((nsplits, len(alturas)))

# Creo una matriz para comparar precision y una para comparar recall donde cada fila
# es un fold, cada columna un modelo y cada profundidad es una vocal
res_prec_gini = np.zeros((nsplits, len(alturas), 5))
res_rcll_gini = np.zeros((nsplits, len(alturas), 5))
res_prec_entropy = np.zeros((nsplits, len(alturas), 5))
res_rcll_entropy = np.zeros((nsplits, len(alturas), 5))

for i, (train_index, test_index) in tqdm(enumerate(kf.split(X_dev)), desc='KFold', position=0, total=nsplits):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    for j, hmax in tqdm(enumerate(alturas), desc='Alturas', position=1, total=len(alturas), leave=False):

        arbol = tree.DecisionTreeClassifier(criterion='gini', max_depth=hmax, random_state=1) # Fijamos semilla
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)

        res_acc_gini[i, j] = metrics.accuracy_score(kf_y_test, pred)

        matrx_conf = fx.matriz_conf_bin_multiclass(letras, kf_y_test, pred)
        for k in range(5):
            res_prec_gini[i, j, k] = fx.precision_score_multiclass(
                letras[k], matrx_conf
            )
            res_rcll_gini[i, j, k] = fx.recall_score_multiclass(letras[k], matrx_conf)
            
        arbol = tree.DecisionTreeClassifier(criterion='entropy', max_depth=hmax, random_state=1) # Fijamos semilla
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)

        res_acc_entropy[i, j] = metrics.accuracy_score(kf_y_test, pred)

        matrx_conf = fx.matriz_conf_bin_multiclass(letras, kf_y_test, pred)
        for k in range(5):
            res_prec_entropy[i, j, k] = fx.precision_score_multiclass(
                letras[k], matrx_conf
            )
            res_rcll_entropy[i, j, k] = fx.recall_score_multiclass(letras[k], matrx_conf)


acc_gini = res_acc_gini.mean(axis=0)
prec_gini = res_prec_gini.mean(axis=0)  # cada letra es una columna, cada fila un modelo
rcll_gini = res_rcll_gini.mean(axis=0)  # cada letra es una columna, cada fila un modelo

acc_entropy = res_acc_entropy.mean(axis=0)
prec_entropy = res_prec_entropy.mean(axis=0)  # cada letra es una columna, cada fila un modelo
rcll_entropy = res_rcll_entropy.mean(axis=0)  # cada letra es una columna, cada fila un modelo

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
best_trees[0,0] = alturas[mejor_pos]
best_trees[0,1:] = [prom_acc_g[mejor_pos],
                   prom_prec_g[mejor_pos],
                   prom_rcll_g[mejor_pos]]

print(f"La mejor altura con criterio Gini basándonos en las 3 métricas es la altura {alturas[mejor_pos]}")

#%% ####
# Comparo los arboles de criterio Entropy

print("\nAnalizamos el criterio Entropy")

metricas_entropy = pd.DataFrame()

metricas_entropy["altura"] = alturas
metricas_entropy["accuracy"] = acc_entropy
metricas_entropy["criterio"] = "Entropy"

for j, hmax in enumerate(alturas):

    for i in range(len(letras)):
        prec_column = "prec_" + letras[i]
        rcll_column = "rcll_" + letras[i]
        metricas_entropy.loc[j, prec_column] = prec_entropy[j, i]
        metricas_entropy.loc[j, rcll_column] = rcll_entropy[j, i]

prom_acc_e = acc_entropy # Lo llamo promedio para mantener estructura en el codigo
prom_prec_e = prec_entropy.mean(axis=1)
prom_rcll_e = rcll_entropy.mean(axis=1)

metricas_entropy = metricas_entropy.reindex(sorted(metricas_entropy.columns), axis=1)
metricas_entropy.insert(2,'accuracy', metricas_entropy.pop('accuracy'))

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
best_trees[1,0] = alturas[mejor_pos]
best_trees[1,1:] = [prom_acc_e[mejor_pos],
                   prom_prec_e[mejor_pos],
                   prom_rcll_e[mejor_pos]]

print(f"La mejor altura con criterio Entropy basándonos en las 3 métricas es la altura {alturas[mejor_pos]}")

metricas_train = pd.concat([metricas_gini, metricas_entropy])

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

#%%
# Guardamos resultados en el anexo
metricas_train = metricas_train.round(5) # Redondeo para que sea más legible
metricas_train.to_csv('./Anexo/clasif_multi_train_res.csv')

performance = performance.round(5) # Redondeo para que sea más legible
performance = pd.concat([pd.Series(data=best_hyperparam),performance])
performance.to_csv('./Anexo/clasif_multi_best_tree.csv')


# %%
