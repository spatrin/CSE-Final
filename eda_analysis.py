"""
Sophia Patrin & Melodie Nekoorad
CSE 163
Final Project
Exploratory Data Analysis

Description:
This program performs exploratory data analysis on the EPA
PM2.5 daily air quality dataset for Washington State in 2022.
It includes data loading, cleaning, summary statistics, missing
value analysis, and visualizations to understand pollution trends
across time and counties. All plots are saved as image files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EPAAnalysis:
    """
    A class to perform exploratory data analysis on EPA PM2.5 dataset.
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
        print("Data loaded successfully.\n")
        print(self._epa.head())

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

        print("\nData cleaning complete.")

    def report_summary(self) -> None:
        """
        Prints summary information.
        """
        print("\nDATASET SIZE:")
        print(f"Rows: {self._epa.shape[0]}")
        print(f"Columns: {self._epa.shape[1]}")

        print("\nCOLUMN NAMES:")
        print(list(self._epa.columns))

        print("\nMISSING VALUES:")
        print(self._epa.isnull().sum())

        print("\nNUMBER OF COUNTIES:")
        print(self._epa["County"].nunique())

        print("\nSUMMARY STATISTICS (PM2.5):")
        print(self._epa["Daily Mean PM2.5 Concentration"].describe())

        print("\nSUMMARY STATISTICS (AQI):")
        print(self._epa["Daily AQI Value"].describe())

    def plot_dist(self) -> None:
        """
        Create and save histogram of PM2.5 distribution.
        """
        plt.figure()
        sns.histplot(self._epa["Daily Mean PM2.5 Concentration"], bins=50)
        plt.title("Distribution of PM2.5 Levels in Washington (2022)")
        plt.xlabel("PM2.5 Concentration (µg/m³)")
        plt.ylabel("Frequency")
        plt.xlim(0, 50)
        plt.savefig("pm25_distribution.png")
        plt.close()
        print("\nFigure 1 Caption:")
        print(
            "Distribution of daily PM2.5 concentrations showing most values "
            "are low, with a right-skewed tail representing extreme pollution "
            "events.")

    def plot_time_series(self) -> None:
        """
        Create and save time series plot of average daily PM2.5.
        """
        daily_avg = self._epa.groupby("Date")[
            "Daily Mean PM2.5 Concentration"
        ].mean()
        plt.figure(figsize=(10, 5))
        daily_avg.plot()
        plt.title("Average Daily PM2.5 levels in Washington (2022)")
        plt.xlabel("Date")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.savefig("pm25_time_series.png")
        plt.close()
        print("\nFigure 2 Caption:")
        print("Time series of PM2.5 levels showing seasonal spikes,"
              "particularly during late summer wildfire season.")

    def compute_county_avg(self) -> pd.DataFrame:
        """
        Compute average PM2.5 concentration per county.
        Returns a pd.DataFrame of the County-level averages
        """
        county_avg = self._epa.groupby("County")[
            "Daily Mean PM2.5 Concentration"
        ].mean().reset_index()
        return county_avg

    def plot_county_bar(self, county_avg: pd.DataFrame) -> None:
        """
        Create and save bar chart of top 10 counties by PM2.5.
        Takes in county_avg (pd.DataFrame).
        """
        county_avg = county_avg.sort_values(
            by="Daily Mean PM2.5 Concentration",
            ascending=False)
        plt.figure()
        sns.barplot(
            data=county_avg.head(10),
            x="Daily Mean PM2.5 Concentration",
            y="County")
        plt.title("Top 10 Counties by Average PM2.5 Levels (2022)")
        plt.xlabel("Average PM2.5 Concentration (µg/m³)")
        plt.ylabel("County")
        plt.savefig("pm25_top_counties.png")
        plt.close()
        print("\nFigure 3 Caption:")
        print(
            "Bar chart of counties with highest average PM2.5 levels, "
            "highlighting geographic variation in pollution exposure.")

    def plot_pm25_vs_aqi(self) -> None:
        """
        Create and save a scatterplot showing the relationship between
        PM2.5 concentration and AQI with a regression line.
        """
        corr = self._epa[
            "Daily Mean PM2.5 Concentration"
        ].corr(self._epa["Daily AQI Value"])
        print(f"Correlation coefficient (PM2.5 vs AQI): {corr:.4f}")

        plt.figure()
        sns.scatterplot(
            data=self._epa,
            x="Daily Mean PM2.5 Concentration",
            y="Daily AQI Value",
            alpha=0.2, s=10)

        sns.regplot(
            data=self._epa,
            x="Daily Mean PM2.5 Concentration",
            y="Daily AQI Value",
            scatter=False,
            ci=None,
            color="red")

        plt.title("Relationship Between PM2.5 and AQI")
        plt.xlabel("PM2.5 Concentration (µg/m³)")
        plt.ylabel("Daily AQI Value")

        plt.xlim(0, 80)
        plt.ylim(0, 200)

        plt.savefig("pm25_vs_aqi.png")
        plt.close()
        print("\nFigure 4 Caption:")
        print(
            "Scatterplot showing a strong positive linear relationship "
            "between PM2.5 concentration and AQI where higher particle "
            "concentrations correspond to worse air quality.")

    def get_epa_data(self) -> pd.DataFrame:
        """
        Returns the cleaned EPA dataframe for merging.
        """
        return self._epa


# MELS PART
class CDCAnalysis:
    """
    A class to perform exploratory data analysis on CDC PLACES asthma dataset.
    """

    FILEPATH: str = "CDC_PLACES_2024.csv"

    def __init__(self, filepath: str = FILEPATH) -> None:
        """
        Initialize the CDCAnalysis object.
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
        print("\nCDC Data loaded successfully.\n")
        print(self._cdc.head())

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

        print("\nCDC Data cleaning complete.")

    def report_summary(self) -> None:
        """
        Prints the summary information.
        """
        print("\n" + "="*50)
        print("CDC DATASET SUMMARY")
        print("="*50)
        print(f"Original CDC rows (Washington): {self._cdc.shape[0]}")
        print(f"Filtered Asthma rows: {self._asthma.shape[0]}")

        print("\nMISSING VALUES IN ASTHMA DATA:")
        print(self._asthma.isnull().sum())

        print("\nNUMBER OF COUNTIES:")
        print(self._asthma["county"].nunique())

        print("\nSUMMARY STATISTICS (Asthma Prevalence):")
        print(self._asthma["asthma_prevalence"].describe())

    def plot_histogram(self) -> None:
        """
        Create and save histogram of asthma prevalence distribution.
        """
        plt.figure()

        sns.histplot(
            self._asthma["asthma_prevalence"],
            bins=10,
            color="steelblue",
            edgecolor="black")

        mean_val = self._asthma["asthma_prevalence"].mean()

        plt.axvline(
            mean_val,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_val:.1f}%")

        plt.title(
            "Distribution of Asthma Prevalence Across "
            "Washington Counties (2022)")
        plt.xlabel("Asthma Prevalence (%)")
        plt.ylabel("Number of Counties")

        plt.legend()
        plt.savefig("fig5_asthma_histogram.png")
        plt.close()

        print("\nFigure 5 Caption:")
        print(
            "Histogram showing the distribution of adult asthma "
            "prevalence across all 39 Washington counties. The "
            "distribution is roughly symmetric with a mean of 11.8%, "
            "suggesting asthma rates are relatively consistent across "
            "the state.")

    def plot_boxplot(self) -> None:
        """
        Create and save box plot of asthma prevalence.
        """
        plt.figure(figsize=(8, 6))
        plt.boxplot(self._asthma["asthma_prevalence"],
                    vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    medianprops=dict(color='red', linewidth=2))

        mean_val = self._asthma["asthma_prevalence"].mean()

        plt.axhline(
            mean_val,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_val:.1f}%")

        plt.ylabel("Asthma Prevalence (%)")
        plt.title(
            "Summary of Asthma Prevalence Across Washington Counties (2022)")
        plt.xticks([1], ['Asthma Prevalence'])
        plt.legend()
        plt.savefig("fig6_asthma_boxplot.png")
        plt.close()
        print("\nFigure 6 Caption:")
        print(
            "Box plot summarizing asthma prevalence across"
            "Washington counties. The median is 11.9%,"
            "with the middle 50% of counties falling"
            "between 11.6% and 12.2%. One mild outlier is present"
            "(King County at 9.3%), but it falls within normal range"
            "and does not violate assumptions for parametric testing.")

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


def main() -> None:
    '''
    Main driver function for the EPA and CDC exploratory data analysis program.
    '''
    print("CSE 163 FINAL PROJECT - EXPLORATORY DATA ANALYSIS")
    print("Washington State PM2.5 and Asthma Prevalence (2022)")

    print("PART 1: EPA PM2.5 DATA ANALYSIS")
    epa_analysis = EPAAnalysis()
    epa_analysis.load_data()
    epa_analysis.clean_data()
    epa_analysis.report_summary()
    epa_analysis.plot_dist()
    epa_analysis.plot_time_series()
    county_avg = epa_analysis.compute_county_avg()
    epa_analysis.plot_county_bar(county_avg)
    epa_analysis.plot_pm25_vs_aqi()

    print("PART 2: CDC ASTHMA DATA ANALYSIS")
    cdc_analysis = CDCAnalysis()
    cdc_analysis.load_data()
    cdc_analysis.clean_data()
    cdc_analysis.report_summary()
    cdc_analysis.plot_histogram()
    cdc_analysis.plot_boxplot()

    print("PART 3: MERGED ANALYSIS")
    merged = merge_datasets(epa_analysis.get_epa_data(),
                            cdc_analysis.get_asthma_data())
    plot_merged_scatter(merged)

    print("  fig1_pm25_distribution.png")
    print("  fig2_pm25_time_series.png")
    print("  fig3_top10_pm25.png")
    print("  fig4_pm25_vs_aqi.png")
    print("  fig5_asthma_histogram.png")
    print("  fig6_asthma_boxplot.png")
    print("  fig7_pm25_vs_asthma.png")


if __name__ == "__main__":
    main()