"""
Sophia Patrin & Melodie Nekoorad
CSE 163
Final Project
Description: Testing file for data_cleaning and analysis programs.
"""

import data_cleaning
from analysis import ResearchQuestions
import pandas as pd

df = data_cleaning.load_and_merge_data()
rq = ResearchQuestions(df)


def test_load_and_merge() -> None:
    """
    Test merged dataset loads correctly
    """

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "annual_mean_pm25" in df.columns
    assert "asthma_prevalence" in df.columns


def test_no_missing_values() -> None:
    """
    Check no missing values in final dataset
    """

    assert df.isnull().sum().sum() == 0


def test_no_negative_pm25() -> None:
    """
    Ensure PM2.5 values are all non-negative after cleaning.
    """
    assert (df["annual_mean_pm25"] >= 0).all()


def test_correct_number_counties() -> None:
    """
    Ensure merged dataset has correct number of counties.
    """
    assert len(df) == 31


def test_no_missing_asthma() -> None:
    """
    Ensure asthma data has no missing values.
    """
    assert df["asthma_prevalence"].isnull().sum() == 0


def test_rq1_output() -> None:
    """
    Test RQ1 returns valid correlation values
    """
    r, p = rq.rq_1()

    assert -1 <= r <= 1
    assert 0 <= p <= 1


def test_rq2_runs() -> None:
    """
    Ensure RQ2 runs without crashing
    """
    t, p = rq.rq_2()

    assert isinstance(t, float)
    assert 0 <= p <= 1


def test_rq3_output() -> None:
    """
    Test RQ3 regression behavior
    """
    model = rq.rq_3()

    assert model.rsquared >= 0
    assert "annual_mean_pm25" in model.params


def main():
    test_load_and_merge()
    test_no_missing_values()
    test_no_negative_pm25()
    test_correct_number_counties()
    test_no_missing_asthma()
    test_rq1_output()
    test_rq2_runs()
    test_rq3_output()

    print("All tests passed")


if __name__ == "__main__":
    main()
