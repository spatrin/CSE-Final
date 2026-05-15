"""
Sophia Patrin & Melodie Nekoorad
CSE 163
Final Project
Exploratory Data Analysis

Description: Testing file for EPA and CDC analysis.
"""

import pandas as pd
from eda_analysis import EPAAnalysis, CDCAnalysis, merge_datasets


def test_epa_load() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    
    assert analysis._epa is not None
    assert isinstance(analysis._epa, pd.DataFrame)


def test_epa_columns() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()
    
    assert "Date" in analysis._epa.columns
    assert "Daily Mean PM2.5 Concentration" in analysis._epa.columns
    assert "Daily AQI Value" in analysis._epa.columns
    assert "County" in analysis._epa.columns


def test_epa_types() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()
    
    assert pd.api.types.is_datetime64_any_dtype(analysis._epa["Date"])
    assert pd.api.types.is_numeric_dtype(analysis._epa["Daily Mean PM2.5 Concentration"])


def test_epa_no_negative_pm25() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()
    
    assert (analysis._epa["Daily Mean PM2.5 Concentration"] >= 0).all()


def test_epa_county_avg() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()
    
    df = analysis.compute_county_avg()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_epa_correlation() -> None:
    analysis = EPAAnalysis()
    analysis.load_data()
    analysis.clean_data()
    
    corr = analysis._epa["Daily Mean PM2.5 Concentration"].corr(analysis._epa["Daily AQI Value"])
    assert corr > 0


def test_cdc_load() -> None:
    analysis = CDCAnalysis()
    analysis.load_data()
    
    assert analysis._cdc is not None
    assert isinstance(analysis._cdc, pd.DataFrame)


def test_cdc_clean() -> None:
    analysis = CDCAnalysis()
    analysis.load_data()
    analysis.clean_data()
    
    assert analysis._asthma is not None
    assert len(analysis._asthma) == 39


def test_cdc_columns() -> None:
    analysis = CDCAnalysis()
    analysis.load_data()
    analysis.clean_data()
    
    assert "county" in analysis._asthma.columns
    assert "asthma_prevalence" in analysis._asthma.columns


def test_cdc_no_missing() -> None:
    analysis = CDCAnalysis()
    analysis.load_data()
    analysis.clean_data()
    
    assert analysis._asthma["asthma_prevalence"].isnull().sum() == 0


def test_merge() -> None:
    epa_analysis = EPAAnalysis()
    epa_analysis.load_data()
    epa_analysis.clean_data()
    
    cdc_analysis = CDCAnalysis()
    cdc_analysis.load_data()
    cdc_analysis.clean_data()
    
    merged = merge_datasets(epa_analysis.get_epa_data(), cdc_analysis.get_asthma_data())
    
    assert isinstance(merged, pd.DataFrame)
    assert len(merged) > 0
    assert "annual_mean_pm25" in merged.columns
    assert "asthma_prevalence" in merged.columns


def main() -> None:
    test_epa_load()
    test_epa_columns()
    test_epa_types()
    test_epa_no_negative_pm25()
    test_epa_county_avg()
    test_epa_correlation()
    test_cdc_load()
    test_cdc_clean()
    test_cdc_columns()
    test_cdc_no_missing()
    test_merge()
    
    print("All tests passed")


if __name__ == "__main__":
    main()