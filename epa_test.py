"""
Sophia Patrin & Melodie Nekoorad
CSE 163
Final Project
Exploratory Data Analysis
Description: Testing file for EPAAnalysis class.
"""

import pandas as pd
from EPA_analysis import EPAAnalysis


def test_load() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()

    assert analysis._epa is not None
    assert isinstance(analysis._epa, pd.DataFrame)


def test_columns() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()

    assert "Date" in analysis._epa.columns
    assert "Daily Mean PM2.5 Concentration" in analysis._epa.columns
    assert "Daily AQI Value" in analysis._epa.columns
    assert "County" in analysis._epa.columns


def test_types() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()

    assert pd.api.types.is_datetime64_any_dtype(analysis._epa["Date"])
    assert pd.api.types.is_numeric_dtype(
        analysis._epa["Daily Mean PM2.5 Concentration"])


def test_no_negative_pm25() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()

    assert (analysis._epa["Daily Mean PM2.5 Concentration"] >= 0).all()


def test_county_avg() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()

    df = analysis.compute_county_avg()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_correlation() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()

    corr = analysis._epa[
        "Daily Mean PM2.5 Concentration"
    ].corr(
        analysis._epa["Daily AQI Value"])
    assert corr > 0


def main() -> None:
    test_load()
    test_columns()
    test_types()
    test_no_negative_pm25()
    test_county_avg()
    test_correlation()

    print("All tests passed")


if __name__ == "__main__":
    main()
