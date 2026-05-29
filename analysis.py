"""
Sophia Patrin & Melodie Nekoorad
CSE 163
Final Project
Description:
This program performs the complete statistical analysis
for the final project.

Research Question 1:
Is there a statistically significant correlation between
county-level annual mean PM2.5 concentration and hospitalization
rates for respiratory disease in Washington State in 2022?

Research Question 2:
Do counties with the highest PM2.5 concentrations have
significantly different asthma prevalence rates than other
Washington counties?

Research Question 3:
Can county-level PM2.5 concentrations predict respiratory
hospitalization rates using confidence intervals?
"""

import data_cleaning
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import pandas as pd


class ResearchQuestions:
    """
    Contains methods for each research question.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with merged dataset
        """
        self._df = df

    def rq_1(self) -> tuple:
        """
        Computes Pearson correlation between PM2.5 and
        asthma prevalence. Takes in the merged dataset
        as a dataframe with PM2.5 and asthma columns.
        Returns a tuple of the correlation coefficient
        and the p-value respectively.
        """
        pm25 = self._df["annual_mean_pm25"]
        asthma = self._df["asthma_prevalence"]
        r, p_value = pearsonr(pm25, asthma)

        print("\n--- Research Question 1 ---")
        print("Correlation between PM2.5 and Asthma Prevalence")
        print(f"Number of counties: {len(self._df)}")
        print(f"Correlation coefficient (r): {r:.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("Conclusion: Statistically significant relationship.")
        else:
            print("Conclusion: No statistically significant relationship.")

        return (r, p_value)

    def rq_2(self) -> None:
        """
        Research Question 2:
        Do counties with the highest PM2.5 concentrations have significantly
        different asthma prevalence than other counties?

        This function compares top 10 PM2.5 counties vs
        remaining counties using an independent t-test.
        """
        sorted_df = self._df.sort_values(
            by="annual_mean_pm25", ascending=False)

        top_10 = sorted_df.head(10)
        remaining = sorted_df.tail(len(self._df) - 10)

        top_asthma = top_10["asthma_prevalence"]
        remaining_asthma = remaining["asthma_prevalence"]

        t_stat, p_value = stats.ttest_ind(
            top_asthma,
            remaining_asthma,
            equal_var=False)

        print("Research Question 2")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        print("\nTop 10 Highest PM2.5 Counties:")
        print(top_10[["county", "annual_mean_pm25"]])

        if p_value < 0.05:
            print("Difference is statistically significant.")
        else:
            print("Difference is NOT statistically significant.")

        # Visualization for RQ2
        plt.figure(figsize=(8, 6))
        means = [top_asthma.mean(), remaining_asthma.mean()]
        errors = [top_asthma.std(), remaining_asthma.std()]

        plt.bar(
            ['Top 10 PM2.5 Counties', 'Other Counties'],
            means, yerr=errors,
            capsize=5, color=['darkred', 'steelblue'], alpha=0.7)
        plt.ylabel('Mean Asthma Prevalence (%)')
        plt.title(
            f'RQ2: Asthma Prevalence Comparison\n'
            f'(t = {t_stat:.3f}, p = {p_value:.4f})')
        plt.savefig("rq2_ttest_comparison.png")
        plt.close()

    def rq_3(self) -> None:
        """
        Research Question 3:
        Can county-level PM2.5 concentrations predict asthma prevalence?
        Uses statsmodels for linear regression with confidence intervals.
        """
        x = self._df["annual_mean_pm25"]
        y = self._df["asthma_prevalence"]
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()

        print("\nRESEARCH QUESTION 3")
        print("=" * 50)
        r_squared = model.rsquared
        coef = model.params["annual_mean_pm25"]
        p_value = model.pvalues["annual_mean_pm25"]
        conf_int = model.conf_int().loc["annual_mean_pm25"]
        lower = conf_int[0]
        upper = conf_int[1]

        print(f"R-squared: {r_squared:.4f}")
        print(f"PM2.5 coefficient: {coef:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")

        if p_value < 0.05:
            print(
                "PM2.5 is a statistically significant predictor."
                )
        else:
            print(
                "PM2.5 is not a statistically significant predictor."
                )

        if lower <= 0 <= upper:
            print("Confidence interval includes 0")
        else:
            print("Confidence interval does not include 0")

        plt.figure(figsize=(10, 6))
        sns.regplot(data=self._df, x="annual_mean_pm25", y="asthma_prevalence",
                    scatter_kws={"alpha": 0.7}, line_kws={"color": "red"})

        plt.title("Linear Regression: PM2.5 vs Asthma Prevalence")
        plt.xlabel("Annual Mean PM2.5 Concentration")
        plt.ylabel("Asthma Prevalence (%)")
        plt.savefig("rq3_regression_model.png")
        plt.close()
        return model


def main():
    """
    Main driver function for final analysis.
    """
    print("Running Analysis for Final Project")
    df = data_cleaning.load_and_merge_data()

    rq = ResearchQuestions(df)

    rq.rq_1()
    rq.rq_2()
    rq.rq_3()


if __name__ == "__main__":
    main()
