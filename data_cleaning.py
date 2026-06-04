"""
Sophia Patrin & Melodie Nekoorad
CSE 163
Final Project
Description: This program includes data loading, cleaning,
and merging the EPA and CDC datasets to prepare for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EPA:
    """
    A class to load and clean the EPA PM2.5 dataset.
    """

    FILEPATH: str = "EPA_PM2.5_WA_2022.csv"

    def __init__(self, filepath: str = FILEPATH) -> None:
        """
        Initialize the EPAAnalysis object.
        """
        self._filepath: str = filepath
        self._epa: pd.DataFrame | None = None

    def load_data(self) -> None:
        """
        Load the EPA dataset and store it in the class.
        """
        self._epa = pd.read_csv(self._filepath)

    def clean_data(self) -> None:
        """
        Clean the dataset by:
        - Removing whitespace from column names
        - Converting PM2.5 column to numeric
        - Converting Date column to datetime
        - Removing invalid negative PM2.5 values
        """
        self._epa.columns = self._epa.columns.str.strip()

        self._epa["Daily Mean PM2.5 Concentration"] = pd.to_numeric(
            self._epa["Daily Mean PM2.5 Concentration"],
            errors="coerce")

        self._epa["Date"] = pd.to_datetime(self._epa["Date"])
        self._epa = self._epa[self._epa["Daily Mean PM2.5 Concentration"] >= 0]

    def compute_county_avg(self) -> pd.DataFrame:
        """
        Compute average PM2.5 concentration per county.
        Returns a pd.DataFrame of the County-level averages
        """
        county_avg = self._epa.groupby("County")[
            "Daily Mean PM2.5 Concentration"
        ].mean().reset_index()
        return county_avg

    def get_epa_data(self) -> pd.DataFrame:
        """
        Returns the cleaned EPA dataframe for merging.
        """
        return self._epa


class CDC:
    """
    A class to load and clean CDC PLACES asthma dataset.
    """

    FILEPATH: str = "CDC_PLACES_2024.csv"

    def __init__(self, filepath: str = FILEPATH) -> None:
        """
        Initialize the CDC object.
        """
        self._filepath: str = filepath
        self._cdc: pd.DataFrame | None = None
        self._asthma: pd.DataFrame | None = None

    def load_data(self) -> None:
        """
        Load the CDC dataset and filter to Washington State.
        """
        self._cdc = pd.read_csv(self._filepath, low_memory=False)
        self._cdc = self._cdc[self._cdc['StateDesc'] == 'Washington']

    def clean_data(self) -> None:
        """
        Clean the dataset by: Filtering to 2022,
        Filtering to current asthma among adults measure,
        using Crude prevalence, and removing rows with missing Data_Value
        """
        self._asthma = self._cdc[
            (self._cdc['Year'] == 2022) &
            (self._cdc['Measure'] == 'Current asthma among adults') &
            (self._cdc['Data_Value_Type'] == 'Crude prevalence')
        ].copy()

        self._asthma = self._asthma.dropna(subset=['Data_Value'])
        self._asthma = self._asthma[['LocationName', 'Data_Value']].copy()
        self._asthma.columns = ['county', 'asthma_prevalence']

    def get_asthma_data(self) -> pd.DataFrame:
        """
        Returns the cleaned asthma dataframe for merging.
        """
        return self._asthma


def merge_datasets(epa_df: pd.DataFrame,
                   asthma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge EPA PM2.5 county averages with CDC asthma prevalence
    data by computing country averages from EPA data and
    cleaning names, then merging.
    """
    county_avg = epa_df.groupby("County")[
        "Daily Mean PM2.5 Concentration"
    ].mean().reset_index()
    county_avg.columns = ['county', 'annual_mean_pm25']

    county_avg["county"] = county_avg["county"].str.replace(
        " County", ""
    ).str.strip()

    asthma_df['county'] = asthma_df['county'].str.strip()

    merged = pd.merge(county_avg, asthma_df, on='county', how='inner')

    print("\n")
    print("MERGED DATASET SUMMARY")
    print(f"Matched counties: {len(merged)}")
    print(f"Columns: {list(merged.columns)}")
    print("\nFirst 5 rows:")
    print(merged.head())

    return merged


def plot_merged_scatter(merged: pd.DataFrame) -> None:
    """
    Create scatter plot of PM2.5 vs Asthma prevalence.
    """
    corr = merged['annual_mean_pm25'].corr(merged['asthma_prevalence'])
    print(f"\nCorrelation (PM2.5 vs Asthma): {corr:.4f}")

    plt.figure(figsize=(10, 6))
    sns.regplot(data=merged, x='annual_mean_pm25', y='asthma_prevalence',
                scatter_kws={'alpha': 0.7, 'color': 'steelblue'},
                line_kws={'color': 'red', 'linewidth': 2})

    plt.text(0.05, 0.95, f'Correlation: r = {corr:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('Annual Mean PM2.5 Concentration (μg/m³)')
    plt.ylabel('Adult Asthma Prevalence (%)')
    plt.title(
        'PM2.5 Exposure vs. Asthma Prevalence by Washington County (2022)')
    plt.savefig("fig7_pm25_vs_asthma.png", dpi=150)
    plt.close()
    print("\nFigure 7 Caption:")
    print(
        "Scatter plot showing the relationship between annual mean"
        "PM2.5 concentration and asthma prevalence across 31 Washington"
        "counties. The correlation coefficient is r = -0.076, indicating"
        "no significant linear relationship between PM2.5 exposure and"
        "asthma prevalence in Washington State in 2022.")


def load_and_merge_data() -> pd.DataFrame:
    """
    returns clean, merged dataframe.
    """
    epa = EPA()
    epa.load_data()
    epa.clean_data()

    cdc = CDC()
    cdc.load_data()
    cdc.clean_data()

    df = merge_datasets(epa.get_epa_data(), cdc.get_asthma_data())
    return df
