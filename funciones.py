import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint


def show_image(
    px_array: np.array, title: str = "", vmin=None, vmax=None, return_array=False
):
    """Funcion que recibe un array de numpy de 784 elementos
    y lo plotea como una imagen de 28x28 (donde los primeros 28 representan
    la primer columna).

    Args:
        px_array (np.array): np.array de 784 floats.
        title (str, optional): String del titulo del grafico. Defaults to "".

    Returns:
        np.array: px_array rotado y espejado (los primeros 28 elementos representan la primer fila)
    """
    # Ploteo el grafico
    px_array = flip_rotate(px_array)
    plt.imshow(px_array, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, 28, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 28, 1), minor=True)
    ax.set_xticks(np.arange(-0.5, 28, 5), minor=False, labels=np.arange(0, 28, 5))
    ax.set_yticks(np.arange(-0.5, 28, 5), minor=False, labels=np.arange(0, 28, 5))
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    # plt.axis('off')
    # ax.set_xticks()
    plt.grid(color="white", which="both", linewidth=0.3)
    plt.show()
    if return_array:
        return px_array


def show_image_from_data(
    n_row: int, data: pd.DataFrame, vmin=None, vmax=None, return_array=False
):
    """
    Función que recibe un número de fila y un dataframe de pandas
    con los datos de las letras. Plotea la imagen correspondiente
    a la fila n_row.
    """
    row = data.iloc[n_row].drop(0, errors="ignore")
    letra = data.iloc[n_row].name

    image_array = np.array(row).astype(np.float32)

    array = show_image(
        image_array,
        title="letra: " + letra + ", row: " + str(n_row),
        vmin=vmin,
        vmax=vmax,
        return_array=return_array,
    )
    if return_array:
        return array


def flip_rotate(image: np.array) -> np.array:
    """
    Función que recibe un array de numpy representando una
    imagen de 28x28. Espeja el array y lo rota en 90°.
    """
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


def matriz_confusion_binaria(y_test, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_test)):
        if y_test[i]:
            if y_pred[i]:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i]:
                fp += 1
            else:
                tn += 1

    return tp, tn, fp, fn


def accuracy_score(tp, tn, fp, fn):
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn) > 0 else 0
    return acc


def precision_score(tp, tn, fp, fn):
    prec = tp / (tp + fp) if tp > 0 else 0
    return prec


def recall_score(tp, tn, fp, fn):
    rec = tp / (tp + fn) if tp > 0 else 0
    return rec


def f1_score(tp, tn, fp, fn):
    prec = precision_score(tp, tn, fp, fn)
    rec = recall_score(tp, tn, fp, fn)
    f1 = 2 * prec * rec / (prec + rec) if prec * rec > 0 else 0
    return f1


def calidad_modelo(y_test, y_pred):
    tp, tn, fp, fn = matriz_confusion_binaria(y_test, y_pred)

    acc = accuracy_score(tp, tn, fp, fn)
    print("Accuracy:", acc)

    prec = precision_score(tp, tn, fp, fn)
    print("Precision:", prec)

    rcll = recall_score(tp, tn, fp, fn)
    print("Recall", rcll)

    f1 = f1_score(tp, tn, fp, fn)
    print("F1:", f1)

    matrx_conf = np.array([[tp, fn], [fp, tn]])
    print("Matriz de confusión binaria:")
    print(matrx_conf)

    return matrx_conf, acc, prec, rcll, f1


def matriz_conf_bin_multiclass(classes, y_test, y_pred):
    """
    input:
        classes: lista de categorias que podría tomar la predicción
        y_test: series con las categorias reales de cada fila
        y_pred: array con las categorías predecidas de cada fila

    output:
        res: dataFrame de dimension nxn, n el numero de clases donde cada
             res[i,j] representa cuántos "i" fueron predecidos como "j", con
             i, j las categorias en las posiciones i, j de clases, respectivamente
    """
    res = pd.DataFrame(index=classes, columns=classes)
    res = res.fillna(0)

    for i in range(len(y_test)):
        res.loc[y_test.iloc[i], y_pred[i]] += 1

    return res


def recall_score_multiclass(cat, confusion_matrix):
    """
    input:
        cat: string, nombre de la categoria que puntuar
        confusion_matrix: dataFrame, matriz de confusion binaria
    output:
        res: float, puntaje recall de la clase
    """
    casos = confusion_matrix.loc[cat]  # Tomo fila
    casos = casos.sum()

    aciertos = confusion_matrix.loc[cat, cat]

    res = 1

    if casos > 0:
        res = aciertos / casos

    return res


def precision_score_multiclass(cat, confusion_matrix):
    """
    input:
        cat: string, nombre de la categoria que puntuar
        confusion_matrix: dataFrame, matriz de confusion binaria
    output:
        res: float, puntaje precision de la clase
    """
    casos = confusion_matrix[cat]  # Tomo columna
    casos = casos.sum()

    aciertos = confusion_matrix.loc[cat, cat]

    res = 1

    if casos > 0:
        res = aciertos / casos

    return res
