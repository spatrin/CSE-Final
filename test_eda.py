"""
Sophia Patrin & Melodie Nekoorad
CSE 163
Final Project

Description:
Testing file for PM2.5 and asthma prevalence analysis.
"""

import pandas as pd
import data_cleaning
from analysis import ResearchQuestions

df = data_cleaning.load_and_merge_data()

def test_load_and_merge_data() -> None:
    """
    Tests that merged dataset loads correctly.
    """
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_required_columns() -> None:
    """
    Tests that merged dataframe contains required columns.
    """
    assert "county" in df.columns
    assert "annual_mean_pm25" in df.columns
    assert "asthma_prevalence" in df.columns


def test_no_missing_values() -> None:
    """
    Tests that key columns contain no missing values.
    """
    assert df["annual_mean_pm25"].isnull().sum() == 0
    assert df["asthma_prevalence"].isnull().sum() == 0


def test_pm25_nonnegative() -> None:
    """
    PM2.5 concentrations should not be negative.
    """
    assert (df["annual_mean_pm25"] >= 0).all()


def test_asthma_nonnegative() -> None:
    """
    Asthma prevalence values should not be negative.
    """
    assert (df["asthma_prevalence"] >= 0).all()


def test_county_count() -> None:
    """
    Washington should contain counties after merge.
    """
    assert len(df) > 0
    assert len(df["county"].unique()) > 0


def test_rq1_runs() -> None:
    """
    Tests that Research Question 1 runs without errors.
    """
    ResearchQuestions.rq_1(df)


def test_rq2_runs() -> None:
    """
    Tests that Research Question 2 runs without errors.
    """
    ResearchQuestions.rq_2(df)


def test_rq3_runs() -> None:
    """
    Tests that Research Question 3 runs without errors.
    """
    ResearchQuestions.rq_3(df)


def main() -> None:
    """
    Runs all tests.
    """
    test_load_and_merge_data()
    test_required_columns()
    test_no_missing_values()
    test_pm25_nonnegative()
    test_asthma_nonnegative()
    test_county_count()
    test_rq1_runs()
    test_rq2_runs()
    test_rq3_runs()

    print("All tests passed!")


if __name__ == "__main__":
    main()
