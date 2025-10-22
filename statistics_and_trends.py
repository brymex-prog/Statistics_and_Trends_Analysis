"""
Statistics & Trends Assignment – COVID-19  Data
Author: Ibrahim Abdulsalam
Date: 22/10/2025

This script performs:
- Data preprocessing and inspection
- Relational, categorical and statistical plots
- Calculation of the 4 main statistical moments:
  mean, standard deviation, skewness, and excess kurtosis.
"""

from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """Create and save a relational plot (days_from_firstcase vs avgtemp)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x='days_from_firstcase', y='avgtemp',
                 hue='Country_Region', legend=False, ax=ax)
    ax.set_title('Days from First Case vs Average Temperature')
    ax.set_xlabel('Days from First Case')
    ax.set_ylabel('Average Temperature (°C)')
    plt.tight_layout() #adjust spacing
    plt.savefig('relational_plot.png')
    plt.close()  #closes the figure
    return


def plot_categorical_plot(df):
    """Create and save a categorical plot (Top 10 countries by avgtemp)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    top_countries = (
        df.groupby('Country_Region')['avgtemp']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    top_countries.plot(kind='barh', ax=ax, color='skyblue')
    ax.invert_yaxis()
    ax.set_title('Top 10 Countries by Average Temperature')
    ax.set_xlabel('Average Temperature (°C)')
    plt.tight_layout()   #adjust spacing
    plt.savefig('categorical_plot.png')
    plt.close()   #closes the figure
    return



def plot_statistical_plot(df):
    """Create and save a statistical plot (correlation heatmap)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = ['density', 'medianage', 'urbanpop', 'avgtemp', 'avghumidity']
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Between Key Numeric Features')
    plt.tight_layout()  #adjust spacing
    plt.savefig('statistical_plot.png')
    plt.close()  #closes the figure
    return

def statistical_analysis(df, col: str):
    """Compute mean, std, skewness, and excess kurtosis for a chosen column."""
    series = df[col].dropna().astype(float)
    mean = series.mean()
    stddev = series.std(ddof=1)
    skew = ss.skew(series, bias=False)
    kurtosis_val = ss.kurtosis(series, fisher=False, bias=False)
    excess_kurtosis = kurtosis_val - 3.0
    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    """Preprocess dataset and display quick EDA outputs."""
    print('\n Dataset Head ')
    print(df.head())

    print('\n Dataset Info ')
    print(df.info())

    print('\n Statistical Summary ')
    print(df.describe())

    # Drop rows with missing temperature data
    df = df.dropna(subset=['avgtemp']).copy()

    # Convert date to datetime if needed
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    print('\n Correlation Matrix ')
    num_cols = ['density', 'medianage', 'urbanpop', 'avgtemp', 'avghumidity']
    print(df[num_cols].corr())

    return df


def writing(moments, col):
    """Print interpretation of the four main statistical moments."""
    mean, stddev, skew, kurt = moments
    print(f'For the attribute {col}:')
    print(f'Mean = {mean:.2f}, '
          f'Standard Deviation = {stddev:.2f}, '
          f'Skewness = {skew:.2f}, '
          f'Excess Kurtosis = {kurt:.2f}.')

    # Interpret skewness
    if abs(skew) < 0.5:
        skew_desc = "approximately symmetric"
    elif skew > 0.5:
        skew_desc = "right (positively) skewed"
    else:
        skew_desc = "left (negatively) skewed"

    # Interpret kurtosis
    if kurt < -1:
        kurt_desc = "platykurtic (thin tails)"
    elif kurt > 1:
        kurt_desc = "leptokurtic (heavy tails)"
    else:
        kurt_desc = "mesokurtic (normal-like tails)"

    print(f'The data is {skew_desc} and {kurt_desc}.')
    return



def main():
    """Main function to execute full analysis."""
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    col = 'avgtemp'  # main column for statistical analysis

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
