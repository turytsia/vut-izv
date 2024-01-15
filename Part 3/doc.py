#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read(path: str) -> pd.DataFrame:
    """
    Reads a pickled DataFrame from the specified file path.

    Parameters:
    - path (str): The file path to the pickled DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame loaded from the pickled file.
    """
    return pd.read_pickle(path)


def parse(df: pd.DataFrame) -> None:
    """
    Parse a DataFrame to obtain counts of different injury levels based on influences.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing relevant data.

    Returns:
    - tuple: A tuple containing lists of death counts, heavily wounded counts, and wounded counts.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()
    
    df_copy["p13a"] = pd.to_numeric(df_copy["p13a"], errors='coerce')
    df_copy["p13b"] = pd.to_numeric(df_copy["p13b"], errors='coerce')
    df_copy["p13c"] = pd.to_numeric(df_copy["p13c"], errors='coerce')

    # Separate data based on influences
    df_alcohol = df_copy[df_copy["p11"] == 1]
    df_drugs = df_copy[df_copy["p11"] == 4]
    df_both = df_copy[df_copy["p11"] == 5]
    df_none = df_copy[df_copy["p11"] == 2]

    # Define categories and count outcomes for each influence
    death_counts = [
        df_alcohol[df_alcohol["p13a"] > 0]["p13a"].sum(),
        df_drugs[df_drugs["p13a"] > 0]["p13a"].sum(),
        df_both[df_both["p13a"] > 0]["p13a"].sum(),
        df_none[df_none["p13a"] > 0]["p13a"].sum()
    ]
    heavily_wounded_counts = [
        df_alcohol[df_alcohol["p13b"] > 0]["p13b"].sum(),
        df_drugs[df_drugs["p13b"] > 0]["p13b"].sum(),
        df_both[df_both["p13b"] > 0]["p13b"].sum(),
        df_none[df_none["p13b"] > 0]["p13b"].sum()
    ]
    wounded_counts = [
        df_alcohol[df_alcohol["p13c"] > 0]["p13c"].sum(),
        df_drugs[df_drugs["p13c"] > 0]["p13c"].sum(),
        df_both[df_both["p13c"] > 0]["p13c"].sum(),
        df_none[df_none["p13c"] > 0]["p13c"].sum()
    ]
    
    well_counts = [
        len(df_alcohol[(df_alcohol["p13a"] == 0) & (
            df_alcohol["p13b"] == 0) & (df_alcohol["p13c"] == 0)]),
        len(df_drugs[(df_drugs["p13a"] == 0) & (
            df_drugs["p13b"] == 0) & (df_drugs["p13c"] == 0)]),
        len(df_both[(df_both["p13a"] == 0) & (
            df_both["p13b"] == 0) & (df_both["p13c"] == 0)]),
        len(df_none[(df_none["p13a"] == 0) & (
            df_none["p13b"] == 0) & (df_none["p13c"] == 0)])
    ]

    return death_counts, heavily_wounded_counts, wounded_counts, well_counts

def graph(df: pd.DataFrame) -> None:
    """
    Generates a bar chart depicting the distribution of outcomes (Death, Heavily Wounded, Wounded) 
    based on different influences (Alcohol, Drugs, Both, None) in car crashes.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing relevant data.

    Returns:
    - None
    """
    # Define categories and count outcomes for each influence
    categories = ['Alcohol', 'Drugs', 'Both', 'None']
    
    death_counts, heavily_wounded_counts, wounded_counts, well_counts = parse(
        df)

    # Calculate total counts for normalization
    total_counts = np.array(death_counts) + \
        np.array(heavily_wounded_counts) + \
        np.array(wounded_counts) + np.array(well_counts)

    # Normalize the data to percentages
    death_percentages = np.array(death_counts) / total_counts * 100
    heavily_wounded_percentages = np.array(heavily_wounded_counts) / total_counts * 100
    wounded_percentages = np.array(wounded_counts) / total_counts * 100
    well_percentages = np.array(well_counts) / total_counts * 100

    # Bar chart configuration
    bar_width = 0.2
    index = np.arange(len(categories))

    fig, ax = plt.subplots()

    bar1 = ax.bar(index, death_percentages, bar_width, label='Death')
    bar2 = ax.bar(index + bar_width, heavily_wounded_percentages,
                bar_width, label='Heavily Wounded')
    bar3 = ax.bar(index + 2 * bar_width, wounded_percentages,
                bar_width, label='Wounded')
    
    bar4 = ax.bar(index + 3 * bar_width, well_percentages,
                  bar_width, label='No injuries')

    ax.set_xlabel('Influence')
    ax.set_ylabel('Percentage')
    ax.set_title('Car Crashes with Different Influences and Results')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Display the percentage values on top of the bars
    for bars in [bar1, bar2, bar3, bar4]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval +
                    1, round(yval, 1), ha='center', va='bottom')

    # Save
    plt.savefig("fig.png", bbox_inches='tight')
    
def table(df: pd.DataFrame) -> None:
    """
    Generates a cross-tabulation table displaying the counts of different injury levels
    based on influences (Alcohol, None, Drugs, Both) in car crashes.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing relevant data.

    Returns:
    - None
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()
    
    # Filter DataFrame to include only specified influences, according to excel (1, 2, 4, 5)
    df_copy = df_copy[df_copy["p11"].isin([1,2,4,5])]

    # Define conditions and corresponding values for injury levels
    conditions = [
        (df_copy['p13a'] == 1),
        (df_copy['p13b'] == 1),
        (df_copy['p13c'] == 1)
    ]

    values = [1, 2, 3]

    # Create a new column 'p13' based on the specified conditions
    df_copy['p13'] = np.select(conditions, values, default=0) 
    
    # Generate a cross-tabulation table
    table = pd.crosstab(df_copy['p13'], df_copy['p11'])
    
    row_labels = ["Not injured", "Deceased", "Heavily wounded", "Wounded"]
    table.index = row_labels

    # Rename index and column labels for better clarity
    col_labels = ["Alcohol", "None", "Drugs", "Both"]
    table.columns = col_labels
    
    print(table)

def values(df: pd.DataFrame) -> None:
    """
    Calculate and print various statistics based on injury outcomes from a given DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing relevant data.

    Returns:
    - None
    """
    death_counts, heavily_wounded_counts, wounded_counts, well_counts = parse(df)

    # Calculate total counts for normalization
    total_counts = np.array(death_counts) + \
        np.array(heavily_wounded_counts) + \
        np.array(wounded_counts) + np.array(well_counts)

    # Normalize the data to percentages
    death_percentages = np.array(death_counts) / total_counts * 100
    heavily_wounded_percentages = np.array(
        heavily_wounded_counts) / total_counts * 100
    
    print("Calculated values:")
    
    # Calculating the average percentage of deaths across influences
    average_death_percentage = np.mean(death_percentages)
    print(f"Average Percentage of Deaths: {round(average_death_percentage, 2)}%")

   # Finding the maximum percentage of heavily wounded cases
    max_heavily_wounded_percentage = np.max(heavily_wounded_percentages)
    print(
        f"Maximum Percentage of Heavily Wounded Cases: {round(max_heavily_wounded_percentage, 2)}%")

    # Calculating the total number of wounded cases across influences
    total_wounded_cases = np.sum(wounded_counts)
    print(f"Total Number of Wounded Cases: {total_wounded_cases}")

if __name__ == "__main__":
    
    df = read("accidents.pkl.gz")
    
    graph(df)
    
    table(df)
    
    values(df)
    
