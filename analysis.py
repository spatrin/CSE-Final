"""
Sophia Patrin & Melodie Nekoorad
CSE 163
Final Project - Final Analysis

Description:
This program performs the complete statistical analysis
for the final project.

Research Question 1:
Correlation between PM2.5 and asthma prevalence

Research Question 2:
T-test comparing high vs low PM2.5 counties

Research Question 3:
Linear regression using statsmodels
"""

import data_cleaning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

class ResearchQuestions:
    """
     A class containing all three research question analyses.
    """
    def rq_1(merged: pd.DataFrame) -> None:
        """
        Research Question 1:
        Is there a statistically significant correlation between county-level
        annual mean PM2.5 concentration and asthma prevalence?

        This function uses SciPy's pearsonr to compute correlation coefficient and p-value
        """
        

    def rq_2(merged: pd.DataFrame) -> None:
        """
        Research Question 2:
        Do counties with the highest PM2.5 concentrations have significantly
        different asthma prevalence than other counties?
    
        This function compares top 10 PM2.5 counties vs remaining counties using an independent t-test.
        """
        sorted_df = merged.sort_values(by="annual_mean_pm25", ascending=False)

        top_10 = sorted_df.head(10)
        remaining = sorted_df.tail(len(merged) - 10)
        
        top_asthma = top_10["asthma_prevalence"]
        remaining_asthma = remaining["asthma_prevalence"]

        t_stat, p_value = stats.ttest_ind(top_asthma, remaining_asthma, equal_var=False)

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
        
        plt.bar(['Top 10 PM2.5 Counties', 'Other Counties'], means, yerr=errors,
                capsize=5, color=['darkred', 'steelblue'], alpha=0.7)
        plt.ylabel('Mean Asthma Prevalence (%)')
        plt.title(
            f'RQ2: Asthma Prevalence Comparison\n'
            f'(t = {t_stat:.3f}, p = {p_value:.4f})')
        plt.savefig("rq2_ttest_comparison.png", dpi=150)
        plt.close()
        print("\nSaved: rq2_ttest_comparison.png")

    def rq_3(merged: pd.DataFrame) -> None:
        """
        Research Question 3:
        Can county-level PM2.5 concentrations predict asthma prevalence?
    
        Uses statsmodels (new library) for linear regression with confidence intervals.
        """
        x = merged["annual_mean_pm25"]
        y = merged["asthma_prevalence"]
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()

        print("\nRESEARCH QUESTION 3")
        print("=" * 50)
        print(model.summary())

        # Regression plot for R3
        plt.figure(figsize=(10, 6))
        sns.regplot(data=merged, x="annual_mean_pm25", y="asthma_prevalence",
                    scatter_kws={"alpha": 0.7}, line_kws={"color": "red"})

        plt.title("Linear Regression: PM2.5 vs Asthma Prevalence")
        plt.xlabel("Annual Mean PM2.5 Concentration")
        plt.ylabel( "Asthma Prevalence (%)")
        plt.savefig("fig8_regression_model.png")
        plt.close()
        print("\nRegression plot saved.")

def main() -> None:
    """
    Main driver function for final analysis.
    """
    print()
    merged = data_cleaning.load_and_merge_data()

    ResearchQuestions.rq_2(merged)
    ResearchQuestions.rq_3(merged)

if __name__ == "__main__":
    main()
    


        
