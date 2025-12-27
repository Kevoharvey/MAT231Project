# Explanation of Rainfall Intensity Analysis Project

This document provides a detailed, cell-by-cell explanation of the code and logic used in the `FInal Statistical Model.ipynb` notebook.

---

## 1. Libraries and Setup (Cell 2)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
```

- **`pandas`**: Used for data manipulation and analysis (loading the CSV, calculating statistics).
- **`numpy`**: Provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.
- **`matplotlib.pyplot` & `seaborn`**: Used for creating the visualizations (histograms, line plots).
- **`scipy.stats`**: Contains a large number of probability distributions and statistical functions (fitting distributions, KS test).
- **`sns.set(style="whitegrid")`**: Configures the aesthetic style of the plots to be cleaner and more readable.

---

## 2. Data Loading and Preprocessing (Cell 3)

```python
df = pd.read_csv('/Users/kevinharvey/Desktop/Projects/Statistics Project/rainfall in india 1901-2015.csv')
df = df.dropna(subset=['ANNUAL'])
df[['ANNUAL']].describe()
```

- **`read_csv`**: Loads the dataset into a DataFrame called `df`.
- **`dropna(subset=['ANNUAL'])`**: This is a critical step. If the `ANNUAL` rainfall value is missing (`NaN`), it would break the statistical calculations. We remove these rows to ensure data integrity.
- **`describe()`**: Generates descriptive statistics like mean, standard deviation, min, and max for the annual rainfall.

---

## 3. Rainfall Intensity Probabilities (Cell 4)

### Intensity Categorization

```python
quantiles = df['ANNUAL'].quantile([0.2, 0.4, 0.6, 0.8])
```

Rainfall "intensity" is defined here using **quantiles**. We divide the data into 5 equal parts:
- **0.2 (20th percentile)**: Threshold for 'Very Low' vs 'Low'.
- **0.8 (80th percentile)**: Threshold for 'High' vs 'Extreme'.

### Empirical Probability

```python
intensity_probs = intensity_counts / len(df)
```

The **Empirical Probability** is the simplest form of prediction. It looks at historical frequencies.
- If "Extreme" rainfall happened in 20% of the recorded years, we predict a **0.2** probability of it happening again, assuming history repeats itself.

### Threshold Exceedance

```python
prob_exceed = (df['ANNUAL'] > 2000).mean()
```

This calculates the probability that in any given year, the total rainfall will be greater than 2000mm. It's the total number of years matching the criteria divided by the total number of years.

---

## 4. Comparison with Theoretical Distributions (Cell 5)

### Fitting the Normal Distribution

```python
mu, std = stats.norm.fit(annual_data)
```

A **Normal Distribution** (Gaussian) is often the starting point for statistics.
- **`mu`**: The mean (center) of the curve.
- **`std`**: The spread (standard deviation).

However, rainfall data is often **skewed** (heavy on one side), so the Normal distribution might not be the best fit.

### Visualization
The code plots the **Empirical Density** (the histogram of actual data) and overlays the **Probability Density Function (PDF)** of the Normal distribution. If the red line matches the blue bars, the model is accurate.

---

## 5. Law of Large Numbers (LLN) Validation (Cell 6)

### The Concept
The Law of Large Numbers (Weak LLN) states that the average of the results obtained from a large number of trials should be close to the expected value (the population mean).

### Implementation

```python
for n in sample_sizes:
    sample = np.random.choice(annual_data, size=n, replace=True)
    sample_means.append(sample.mean())
```

1. We calculate the **Population Mean** (the average of all 4115 records).
2. We take a tiny sample (e.g., 1 year), then a larger one (10 years), then 100, up to 2000.
3. We plot the "running average".
4. **Observation**: At small sample sizes, the mean fluctuates wildly. As $n$ grows, the blue line flattens and merges with the red line (the true mean). This visually proves the LLN.

---

## 6. Curve Fitting and Trend Analysis (Cell 7 & 8)

### Distribution Comparison
In addition to the Normal distribution, we now fit:
- **Gamma Distribution**: Often more accurate for rainfall as it handles the "heavy tail" of extreme events.
- **Lognormal Distribution**: Another excellent model for data that cannot be negative.

### Visual Comparison (ECDF)
We use the **Empirical Cumulative Distribution Function (ECDF)** to compare how well the theoretical models match reality. While the PDF (histogram) shows density, the ECDF shows how probabilities accumulate. A closer match in the ECDF plot indicates a better model.

### Long-Term Trend
We calculate a **Trend Curve** (Linear and Quadratic) over the historical years to see if there is a statistically significant change in rainfall intensity over the last century.

---

## 7. Rainfall Probability Equation Model (Cell 9)

### The Mathematical Equation
Based on our analysis, we select the **Gamma Distribution** as our primary prediction model. The probability of a certain rainfall amount is calculated using its Probability Density Function:

$$ f(x; a, \theta) = \frac{x^{a-1} e^{-x/\theta}}{\theta^a \Gamma(a)} $$

### Interactive Prediction
We implemented a function `get_rainfall_probability(value)` which allows anyone to:
1. Input a rainfall amount (e.g., 2500mm).
2. Get the probability of it happening (CDF).
3. Get the **Exceedance Probability** (the "Risk Score") which tells us the chance of a flood event occurring beyond that threshold.

---

## Summary of Results
- **Rainfall Prediction**: We now have a formal mathematical equation (Gamma Model) to predict future rainfall intensity.
- **Trend**: We can identify if rainfall is increasing or decreasing over time using our curve-fitting models.
- **Validation**: Use the LLN and ECDF plots to ensure our predictions are scientifically sound.
