#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Oleksandr Turytsia (xturyt00)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    """
    The integrate function takes a function f, and two floats a and b.
    It returns the integral of f from a to b using the trapezoidal rule.
    The number of steps used is 1000 by default, but can be changed with an optional argument.

    :param f: Callable[[NDArray]: Specify the function to be integrated
    :param a: float: Set the lower limit of integration
    :param b: float: Specify the upper bound of the integral
    :param steps: Determine the number of points to use in the integration
    :return: The integral of the function f between a and b
    """
    a = np.float64(a)
    b = np.float64(b)
    steps = np.int32(steps)
    return np.sum(f(np.linspace(a, b, steps))) * (b - a) / steps


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    The generate_graph function generates a graph of the function f(x) for different values of a.
    According to a task a can only be [1.0, 1.5, 2.0], for this reason plot is limited by x (-3, 3) and by y (0, 40)

    :param a: List[float]: Pass a list of float values to the function
    :param show_figure: bool: Show the figure or not
    :param save_path: str | None: Save the graph to a file
    :return: A graph
    """
    a = np.array(a)

    def f(_x: NDArray) -> NDArray:
        """
        The f function is a simple function of x

        :param _x: NDArray: Pass the array of parameters to the function
        :return: A matrix
        """
        return a[:, np.newaxis] ** 2 * _x ** 3 * np.sin(_x)

    x = np.linspace(-3, 3, 1000)
    y = f(x)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    labels = [f"$Y_{{{float(a_)}}}(x)$" for a_ in a]

    ax.plot(x, y.T, label=labels)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f_{a}(x)$')

    for _a, _y in zip(a, y):
        annotation_text = f'$\\int f_{{{float(_a)}}}(x)dx={np.trapz(x=x, y=_y):.2f}$'
        ax.annotate(annotation_text, (3, _y[-1]), textcoords='offset points', xytext=(1, -5))
        ax.fill_between(x, 0, _y, alpha=0.1)

    ax.legend(ncol=3, bbox_to_anchor=(0.75, 1.15))
    ax.set_xticks(np.arange(-3, 4, 1))

    ax.set_xlim(-3, 5)
    ax.set_ylim(0, 40)

    if show_figure:
        plt.show()

    if save_path:
        fig.savefig(save_path)

    plt.close(fig)


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    The generate_sinus function generates a figure with three subplots.

    :param show_figure: bool: Show the figure or not
    :param save_path: str | None: Save the figure to a file
    :return: A figure with the three functions
    """

    def f1(t: NDArray) -> NDArray:
        """
        The f1 function

        :param t: NDArray: Time
        :return: A numpy array with the values of f(t)
        """
        return 0.5 * np.cos(1 / 50 * np.pi * t)

    def f2(t: NDArray) -> NDArray:
        """
        The f2 function

        :param t: NDArray: Time
        :return: A numpy array with the values of f(t)
        """
        return 0.25 * (np.sin(np.pi * t) + np.sin((3 / 2) * np.pi * t))

    def f3(t: NDArray) -> NDArray:
        """
        The f3 function is the sum of f1 and f2.

        :param t: NDArray: Time
        :return: The sum of the two functions f1 and f2
        """
        return f1(t) + f2(t)

    x = np.linspace(0, 100, 10000)
    funcs = [f1(x), f2(x), f3(x)]

    figs, axs = plt.subplots(3, 1, figsize=(6, 8))

    for i, ax in enumerate(axs):
        y = funcs[i]

        ax.plot(x, y)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xlim(0, 100)
        ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
        ax.set_xlabel("$t$")
        if i == 2:
            ax.plot(x, y, color="green")
            y[y > funcs[0]] = np.nan
            ax.plot(x, y, color="red")
            ax.set_ylabel("$f_{1}(t) + f_{2}(t)$")
        else:
            ax.set_ylabel(f"$f_{{{i + 1}}}(t)$")

    figs.tight_layout()

    if show_figure:
        plt.show()

    if save_path:
        figs.savefig(save_path)

    plt.close(figs)


def download_data() -> List[Dict[str, Any]]:
    """
    The download_data function downloads the data from the website and returns a list of dictionaries.
    Each dictionary contains information about one position: name, latitude, longitude and height above sea level.

    :return: A list of dictionaries
    """

    def parse_raw_number(raw: str) -> float:
        """
        The parse_raw_number function takes a string as input and returns a float.
        It is used to parse the raw data from the website into numbers that can be used for calculations.
        The function replaces commas with dots, removes degree symbols and strips whitespace.

        :param raw: str: Define the input of the function
        :return: A float number
        """
        return float(raw.replace(",", ".").replace("°", "").strip())

    response = requests.get("https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html")

    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return []

    bs = BeautifulSoup(response.content, features="html.parser")

    table = bs.find_all("table")[-1]

    rows = table.find_all("tr")

    return [
        {
            "position": td[0].text,
            "lat": parse_raw_number(td[2].text),
            "long": parse_raw_number(td[4].text),
            "height": parse_raw_number(td[6].text)
        }
        for tr in rows[1:]
        for td in [tr.find_all("td")]
    ]
