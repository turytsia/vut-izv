#!/usr/bin/env python3.11
# coding=utf-8

# @author Oleksandr Turytsia

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
#import io

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru


def load_data(filename: str) -> pd.DataFrame:
    """
    Load and concatenate accident data from a compressed file, organizing it into a DataFrame.

    Parameters:
    - filename (str): The path to the compressed file containing accident data.

    Returns:
    pd.DataFrame: A DataFrame containing concatenated accident data with additional 'region' column.
    """
    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]

    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    reverse_regions = {v: k for k, v in regions.items()}

    reverse_regions_keys = list(reverse_regions.keys())

    dfs = []
    
    # TODO https://stackoverflow.com/questions/29806936/why-is-pandas-concatenation-pandas-concat-so-memory-inefficient
    with zipfile.ZipFile(filename, 'r') as zip:

        for zip_name in zip.namelist():

            with zipfile.ZipFile(zip.open(zip_name), 'r') as inner_zip:

                for csv_file_name in inner_zip.namelist():

                    name = csv_file_name.split('.')[0]

                    if name not in reverse_regions_keys:
                        continue

                    df = pd.read_csv(
                        inner_zip.open(csv_file_name), sep=";", names=headers, encoding="cp1250", low_memory=False)

                    df["region"] = reverse_regions.get(name, None)

                    dfs.append(df)
                    
    return pd.concat(dfs, ignore_index=True)


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Parse and preprocess the input DataFrame to optimize memory usage and handle date columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing raw data.
    - verbose (bool, optional): If True, print original and new memory usage. Default is False.

    Returns:
    pd.DataFrame: The parsed and preprocessed DataFrame.
    """

    def get_memory_usage(df: pd.DataFrame) -> str:

        return str(round(df.memory_usage(deep=True).sum() / 10**6, 1))

    df_copy = df.copy()

    df_copy = df_copy.drop(columns=["p2a"])

    df_copy["date"] = pd.to_datetime(
        df["p2a"], format="%Y-%m-%d", errors='coerce')

    df_copy["d"] = pd.to_numeric(df_copy["d"], errors="coerce")

    df_copy["e"] = pd.to_numeric(df_copy["e"], errors="coerce")

    df_copy = df_copy.drop_duplicates(subset='p1')

    category_colums = df_copy.select_dtypes(
        include=["object"], exclude=["datetime"]).columns.difference(['region'])

    df_copy[category_colums] = df_copy[category_colums].astype('category')

    if verbose:

        print(f"orig_size={get_memory_usage(df)} MB")

        print(f"new_size={get_memory_usage(df_copy)} MB")

    return df_copy


def plot_state(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """
    Generate bar plots illustrating the count of accidents based on the state of the driver across regions.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing accident data.
    - fig_location (str, optional): The file path where the plot should be saved. Default is None.
    - show_figure (bool, optional): Whether to display the plot. Default is False.

    Returns:
    None
    """
    df_copy = df.copy()

    df_copy = df_copy.loc[np.logical_and(
        df_copy["p57"] > 1, df_copy["p57"] < 8)]

    map = {2: "unaven, usnul, náhlá fyzická indispozice", 3: "pod vlivem léků, narkotik",
           4: "pod vlivem alkoholu do 0,99 ‰", 5: "pod vlivem alkoholu 1 ‰ a vice", 6: "nemoc úraz apod",
           7: "invalida"}

    df_copy["p57"] = df_copy["p57"].map(map)

    data = df_copy.groupby(["p57", "region"])["p57"].agg(count="count")

    sns.set(style="darkgrid")
    plot: sns.FacetGrid = sns.catplot(data=data, x="region", y="count", col="p57", kind="bar", hue="region",
                                      sharey=False, col_wrap=2, height=4, aspect=1.5)

    plot.set_titles("Stav řidiče: {col_name}")
    plot.set_axis_labels("Kraj", "Počet nehod")

    plt.subplots_adjust(top=0.9)
    plot.figure.suptitle('Počet nehod dle stavu řidiče při nedobrém stavu')
    plot.legend.remove()

    if fig_location is not None:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()

# Ukol4: alkohol v jednotlivých hodinách


def plot_alcohol(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Generate bar plots illustrating the count of accidents involving alcohol across regions and hours.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing accident data.
    - fig_location (str, optional): The file path where the plot should be saved. Default is None.
    - show_figure (bool, optional): Whether to display the plot. Default is False.

    Returns:
    None
    """

    df_copy = df.copy()

    df_copy = df_copy[df_copy["region"].isin(["JHM", "MSK", "OLK", "ZLK"])]

    df_copy["alk"] = "Ne"
    df_copy.loc[df_copy["p11"] >= 3, "alk"] = "Ano"

    df_copy = df_copy[np.logical_and(
        df_copy["p2b"] >= 0, df_copy["p2b"] < 2400
    )]

    df_copy["p2b"] = df_copy["p2b"] // 100

    data = df_copy.groupby(["alk", "region", "p2b"])["p2b"].agg(count="count")

    sns.set(style="darkgrid")

    plot: sns.FacetGrid = sns.catplot(data=data, x="p2b", y="count", col="region", kind="bar", hue="alk",
                                      sharey=False, col_wrap=2, height=4, aspect=1.5)

    plot.set_titles("Kraj: {col_name}")

    plot.set_axis_labels("Hodina", "Počet nehod")

    plot.legend.set_title("Alkohol")

    if fig_location is not None:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()

# Ukol 5: Zavinění nehody v čase


def plot_fault(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """
    Generate a line plot visualizing accident counts for different fault categories across regions and dates.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing accident data.
    - fig_location (str, optional): The file path where the plot should be saved. Default is None.
    - show_figure (bool, optional): Whether to display the plot. Default is False.

    Returns:
    None
    """
    regions = ["JHM", "MSK", "OLK", "ZLK"]
    labels = ["01/16", "01/17", "01/18", "01/19",
              "01/20", "01/21", "01/22", "01/23"]
    
    label_names = ["Chodcem", "Zvířetem", "Řidičem motorového vozidla", "Řidičem nemotorového vozidla"]
    
    def filter_and_transform_data(df: pd.DataFrame):
        """
        Filter and transform the input DataFrame to prepare it for visualization.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing accident data.

        Returns:
        pd.DataFrame: The filtered and transformed DataFrame.
        """
        df_copy = df.copy()

        df_copy = df_copy[df_copy["region"].isin(regions)]
        df_copy = df_copy[df_copy["p10"].between(1, 4)]

        conditions = [df_copy["p10"] == i for i in range(2, 5)]
        choices = ["Řidičem nemotorového vozidla", "Chodcem", "Zvířetem"]
        df_copy["fault"] = np.select(conditions, choices, default="Řidičem motorového vozidla")

        df_copy["month"] = df_copy["date"].dt.month
        df_copy["year"] = df_copy["date"].dt.year

        return df_copy

    def create_and_plot_visualization(filtered_df: pd.DataFrame):
        """
        Create and display a line plot visualizing accident counts for different fault categories.

        Parameters:
        - filtered_df (pd.DataFrame): The filtered and transformed DataFrame.

        Returns:
        None
        """
        pv = filtered_df.pivot_table(
            index=["region", "year", "month"], columns="fault", values="p1", aggfunc="count")
        pv = pv.stack().reset_index()

        pv["date"] = pd.to_datetime(pv[["year", "month"]].assign(DAY=1))
        pv = pv.drop(columns=["year", "month"]).pivot_table(
            index=["region", "date"], columns="fault").stack()

        pv.columns = ["count"]

        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=False)
        axs = axes.flatten()

        for i, ax in enumerate(axs):
            region_data = pv.iloc[pv.index.get_level_values(
                'region') == regions[i]]
            sns.lineplot(data=region_data, x="date", y="count",
                         ax=ax, style="fault", hue="fault", errorbar=None, legend=False)
            ax.margins(x=0)

            locs, _ = ax.get_xticks(), ax.get_xticklabels()
            locs = np.append(locs, (locs[-1] + locs[-1] - locs[-2]))

            ax.set_title("Kraj:" + regions[i])
            ax.set_xlabel("")
            ax.set_ylabel("Počet nehod")
            ax.set_xticks(ticks=locs[:], labels=labels)

        fig.legend(title="Zavinění", labels=label_names, loc="center left",
                bbox_to_anchor=(1, 0.5), edgecolor='black', frameon=True)

        fig.tight_layout()
        
    filtered_data = filter_and_transform_data(df)
    create_and_plot_visualization(filtered_data)

    if fig_location is not None:
        plt.savefig(fig_location, bbox_inches='tight')

    if show_figure:
        plt.show()


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.
    df = load_data("data/data.zip")
    df2 = parse_data(df, True)

    # plot_state(df2, "01_state.png")
    # plot_alcohol(df2, "02_alcohol.png", True)
    plot_fault(df2, "03_fault.png")


# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
