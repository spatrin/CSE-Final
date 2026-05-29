# Final Project: PM2.5 Exposure and Asthma Outcomes in Washington State

## Overview
This project analyzes the relationship between county-level PM2.5 air pollution and asthma prevalence in Washington State for 2022. 

The project addresses three research questions:
1. Is there a statistically significant correlation between county-level annual mean PM2.5 concentration
   and hospitalization rates for respiratory disease in Washington State in 2022?
2. Do counties with the highest PM2.5 concentrations have significantly different asthma prevalence rates
   than other Washington counties?
3. Can county-level PM2.5 concentrations predict respiratory hospitalization rates using confidence intervals?

---

## Files

### analysis.py
Runs the full statistical analysis:
- Research Question 1: Pearson correlation
- Research Question 2: Independent t-test
- Research Question 3: Linear regression with confidence intervals

### data_cleaning.py
Handles data processing:
- Loads EPA PM2.5 dataset
- Loads CDC asthma dataset
- Cleans and filters data
- Merges datasets at the county level

### final_test.py
Testing file:
- Validates data cleaning and merging
- Tests dataset integrity (no missing values, correct size)
- Tests each research question function

---

## Requirements

This project requires the following Python libraries:
- pandas
- matplotlib
- seaborn
- scipy
- statsmodels

## Data

You must download the datasets yourself:

### EPA PM2.5 Data (2022)
Download from:
https://www.epa.gov/outdoor-air-quality-data/download-daily-data
pollutant = PM2.5
year = 2022
geographic area = Washington
rename to: EPA_PM2.5_WA_2022.csv

### CDC PLACES Data (2024 Release)
Download from:
https://data.cdc.gov/](https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/fu4u-a9bh/about_data)
rename to: CDC_PLACES_2024.csv

### Important
- Place both CSV files in the same directory as your project files
- Update file paths in `data_cleaning.py` if necessary

## How to Run

### 1. Run the analysis

`python analysis.py`

This will:
- Load and clean the data
- Run all three research questions
- Print statistical results
- Save visualizations

---

### 2. Run the tests

`python final_test.py`

You should see: "All tests passed"

---

## Project Structure
 project/
│
├── analysis.py
├── data_cleaning.py
├── final_test.py
├── README.md
└── report.pdf

---

## Notes

- The dataset is preprocessed programmatically in `data_cleaning.py`
- All statistical outputs are generated in `analysis.py`
- Tests ensure results are reproducible and correct
- Visualizations are saved as image files during execution

---

## Reproducibility

To reproduce results:
1. Download the datasets
2. Place them in the project folder
3. Run `analysis.py`

---

## Authors

Sophia Patrin  
Melodie Nekoorad
