import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint


def show_image(px_array: np.array, title: str = "", vmin=None, vmax=None):
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
    return px_array


def show_image_from_data(n_row: int, data: pd.DataFrame, vmin=None, vmax=None):
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
    )

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
