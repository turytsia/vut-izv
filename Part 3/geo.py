#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
from sklearn.cluster import KMeans
import seaborn as sns
from shapely import MultiPoint

def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""
    df['p2a'] = pd.to_datetime(df['p2a'])

    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df["d"], df["e"]),
        crs="EPSG:5514"
    )

    return gdf


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s nehodami  """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.set_title("2021")
    ax2.set_title("2022")

    ax1.set_axis_off()
    ax2.set_axis_off()

    gdf_copy = gdf.copy()
    
    gdf_common = gdf_copy[
        (gdf_copy["p10"] == 4) &
        (gdf_copy["region"] == "JHM")
    ].to_crs("EPSG:3857")
    
    gdf_2021: geopandas.GeoDataFrame = gdf_common[gdf_common["p2a"].dt.year == 2021]
    
    gdf_2022: geopandas.GeoDataFrame = gdf_common[gdf_common["p2a"].dt.year == 2022]

    gdf_2021.plot(
        ax=ax1, 
        markersize=0.5,
        color="red"
    )
    
    gdf_2022.plot(
        ax=ax2,
        markersize=0.5,
        color="red"
    )
    
    jhm = ctx.Place("Jihomoravský kraj",
                    source=ctx.providers.OpenStreetMap.Mapnik)
    
    jhm.plot(ax=ax1)
    
    jhm.plot(ax=ax2)

    if fig_location:
        plt.savefig(fig_location, dpi=300, bbox_inches='tight')

    if show_figure:
        plt.show()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """
    
    place = ctx.Place("Jihomoravský kraj",
                      source=ctx.providers.OpenStreetMap.Mapnik)
    
    gdf_copy = gdf.copy()

    gdf_copy: geopandas.GeoDataFrame = gdf_copy[
        (gdf_copy["p11"] >= 4) &
        (gdf_copy["region"] == "JHM")
    ].to_crs("EPSG:3857").clip(mask=place.bbox_map)

    cluster = KMeans(n_clusters=12, n_init="auto")

    points = np.column_stack(
        [gdf_copy['geometry'].x.values, gdf_copy['geometry'].y.values])

    gdf_copy["cluster"] = cluster.fit_predict(np.vstack(points))
    
    norm = plt.Normalize(vmin=0, vmax=gdf_copy["cluster"].value_counts().max())
    
    cmap = sns.color_palette("viridis", as_cmap=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(cluster.n_clusters):

        cluster_points = points[gdf_copy["cluster"] == i]

        mp = MultiPoint(cluster_points)

        geopandas.GeoSeries(mp.convex_hull).plot(
            ax=ax,
            color="black",
            alpha=0.25,
            linewidth=1
        )

        gdf_points: geopandas.GeoDataFrame = gdf_copy[gdf_copy["cluster"] == i]

        gdf_points.plot(
            ax=ax,
            legend=False,
            alpha=1,
            markersize=0.5,
            color=cmap(norm(gdf_points["cluster"].value_counts()))
        )
        
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    cbar = plt.colorbar(
        mappable,
        ax=ax,
        pad=0.01,
        orientation="horizontal",
    )

    cbar.set_label("Počet nehod v úseku")
    
    place.plot(ax=ax)

    plt.title("Nehody v JHM kraji s významnou měrou alkoholu")
    
    plt.axis("off")

    if fig_location:
        plt.savefig(fig_location, dpi=300, bbox_inches='tight')

    if show_figure:
        plt.show()


if __name__ == "__main__":
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
