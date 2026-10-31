
# Food Delivery Time Prediction Using Machine Learning

An end-to-end Machine Learning project that predicts food delivery time in minutes using delivery distance, weather, traffic, preparation time, and courier experience. Developed as part of the SYBCA-AI (Semester III) coursework at Sutex Bank College of Computer Applications & Science.

## Team Members

- **Shyam Sitapara** — Roll No. 550
- **Dhruv Vora** — Roll No. 551
- **Faculty / Supervisor:** Hiral A. Patel

## Project Overview

The objective of this project is to estimate the total food delivery time in minutes using relevant delivery and order features.

- **Target variable:** `Delivery_Time_min`
- **Dataset:** Food Delivery Time Dataset
- **Models:** Linear Regression and Ridge Regression

## Dataset and Features

The dataset is stored in `Dataset/Food_Delivery_Time_Dataset.csv`.

Features used:

- `Distance_km`
- `Weather`
- `Traffic_Level`
- `Time_of_Day`
- `Vehicle_Type`
- `Preparation_Time_min`
- `Courier_Experience_yrs`

## Data Preprocessing

The project includes data cleaning and preprocessing steps such as:

- Imputing missing categorical values with the mode
- Imputing missing numerical values with the median
- Detecting and handling outliers using the Interquartile Range (IQR) method
- Applying One-Hot Encoding to categorical features

## Models and Evaluation

The models were evaluated using an 80:20 train-test split.

### Linear Regression

| Metric | Result |
|---|---:|
| MAE | 5.5792 |
| MSE | 73.9204 |
| RMSE | 8.5977 |
| R² Score | 0.8508 |
| Cross-validation Mean R² | 0.7895 |

### Ridge Regression

| Metric | Result |
|---|---:|
| MAE | 5.5773 |
| MSE | 73.8912 |
| RMSE | 8.5960 |
| R² Score | 0.8508 |

## Project Files

- `Dataset/Food_Delivery_Time_Dataset.csv` — Dataset
- `Python_Code/project.py` — Data preprocessing, analysis, model training, and evaluation
- `Reports/EDA_Report.html` — Exploratory Data Analysis report
- `Reports/Project_Report.pdf` — Project report

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- sweetviz
- scikit-learn

## How to Run the Project

Clone the repository:

```bash
git clone https://github.com/shyamsitapara07-tech/Food-Delivery-Time-Prediction.git
cd Food-Delivery-Time-Prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn sweetviz scikit-learn
```

Run the project script:

```bash
python Python_Code/project.py
```

## Repository

[Food Delivery Time Prediction on GitHub](https://github.com/shyamsitapara07-tech/Food-Delivery-Time-Prediction)
