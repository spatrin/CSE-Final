"""
Sophia Patrin
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

    FILEPATH: str = "EPA_PM2.5_daily.csv"

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
        print("Distribution of daily PM2.5 concentrations showing most values are low, with a right-skewed tail representing extreme pollution events.")



    def plot_time_series(self) -> None:
        """
        Create and save time series plot of average daily PM2.5.
        """
        daily_avg = self._epa.groupby("Date")["Daily Mean PM2.5 Concentration"].mean()
        plt.figure(figsize=(10, 5))
        daily_avg.plot()
        plt.title("Average Daily PM2.5 levels in Washington (2022)")
        plt.xlabel("Date")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.savefig("pm25_time_series.png")
        plt.close()
        print("\nFigure 2 Caption:")
        print("Time series of PM2.5 levels showing seasonal spikes, particularly during late summer wildfire season.")


    def compute_county_avg(self) -> pd.DataFrame:
        """
        Compute average PM2.5 concentration per county.
        Returns a pd.DataFrame of the County-level averages
        """
        county_avg = self._epa.groupby("County")["Daily Mean PM2.5 Concentration"].mean().reset_index()
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
        print("Bar chart of counties with highest average PM2.5 levels, highlighting geographic variation in pollution exposure.")


    def plot_pm25_vs_aqi(self) -> None:
        """
        Create and save a scatterplot showing the relationship between
        PM2.5 concentration and AQI with a regression line.
        """
        corr = self._epa["Daily Mean PM2.5 Concentration"].corr(self._epa["Daily AQI Value"])
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
        print("Scatterplot showing a strong positive linear relationship between PM2.5 concentration and AQI where higher particle concentrations correspond to worse air quality.")




def main() -> None:
    '''
    Main driver function for the EPA exploratory data analysis program.
    '''
    analysis = EPAAnalysis()

    analysis.load_data()
    analysis.clean_data()
    analysis.report_summary()

    analysis.plot_dist()
    analysis.plot_time_series()

    county_avg = analysis.compute_county_avg()

    print("\nCounty averages preview:")
    print(county_avg.head())

    analysis.plot_county_bar(county_avg)
    analysis.plot_pm25_vs_aqi()


if __name__ == "__main__":
    main()