

## ## Load Rainfall Data

### **Goal:**

Load the dataset, inspect it, and check for missing values + data types.

```python
import pandas as pd
```

Imports the **pandas** library for data manipulation & reading.

```python
df_rainfall = pd.read_csv('rainfall in india 1901-2015.csv')
```

Reads the CSV file and loads it into a pandas **DataFrame** named `df_rainfall`.

```python
print("First 5 rows of the DataFrame:")
print(df_rainfall.head())
```

Shows the first 5 rows so you can instantly see if the dataset loaded correctly.

```python
print("\nMissing values in each column:")
print(df_rainfall.isnull().sum())
```

Checks every column for missing values by counting all `NaN` entries.

```python
print("\nData types of each column:")
df_rainfall.info()
```

Displays **data types** (int, float, object…), memory usage, and non-null counts.

---

## ## Data Exploration and Preprocessing

### **Goal:**

Clean missing values, get descriptive stats, and plot a histogram.

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

Imports **matplotlib** for plotting and **seaborn** for prettier plots.

```python
df_rainfall_cleaned = df_rainfall.dropna(subset=['ANNUAL'])
```

Removes any row where the `ANNUAL` rainfall value is missing.

```python
print(f"\nNumber of rows before dropping NaNs: {len(df_rainfall)}")
print(f"Number of rows after dropping NaNs in 'ANNUAL': {len(df_rainfall_cleaned)}")
```

Prints the number of rows before/after cleaning so you see exactly how much was removed.

```python
print("\nDescriptive statistics for 'ANNUAL' rainfall:")
print(df_rainfall_cleaned['ANNUAL'].describe())
```

Displays summary stats such as mean, median, std, quartiles, min, max.

```python
plt.figure(figsize=(10, 6))
sns.histplot(df_rainfall_cleaned['ANNUAL'], bins=50, kde=True)
```

Creates a histogram of annual rainfall + a KDE (smooth density) curve.

```python
plt.title('Distribution of Annual Rainfall (1901-2015)')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

Adds labels, a grid, and displays the plot.

---

## ## Demonstrating the Law of Large Numbers (LLN)

### **Goal:**

Simulate coin flips to show that sample averages converge to the theoretical probability.

```python
np.random.seed(42)
```

Sets a seed so results are **repeatable**.

```python
num_tosses = 1000
```

Defines how many coin flips you simulate.

```python
results = np.random.binomial(1, 0.5, num_tosses)
```

Generates 1000 coin flips where:

* `1` = heads
* `0` = tails

```python
cumulative_heads = np.cumsum(results)
```

Keeps a running total of heads.

```python
cumulative_proportion = cumulative_heads / np.arange(1, num_tosses + 1)
```

Calculates the proportion of heads after each flip.

---

### Plotting LLN

```python
plt.figure(figsize=(12, 7))
sns.lineplot(x=np.arange(1, num_tosses + 1), y=cumulative_proportion, color='blue', label='Cumulative Proportion of Heads')
```

Plots the cumulative proportion of heads over time.

```python
plt.axhline(0.5, color='red', linestyle='--', label='Theoretical Probability (0.5)')
```

Draws the horizontal line showing the true probability.

```python
plt.xscale('log')
```

Uses a log scale to show early randomness vs later convergence.

```python
plt.title('Demonstration of the Law of Large Numbers (Coin Toss)')
plt.xlabel('Number of Coin Tosses')
plt.ylabel('Cumulative Proportion of Heads')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
```

Labels & displays the convergence plot.

---

## ## Fitting a Normal Distribution

```python
from scipy.stats import norm
```

Imports the Normal distribution functions.

```python
rainfall_data = df_rainfall_cleaned['ANNUAL'].dropna().values
```

Extracts rainfall values into a NumPy array.

```python
mu_norm, std_norm = norm.fit(rainfall_data)
```

Estimates:

* mean (μ)
* standard deviation (σ)

```python
print(f"\nNormal Distribution Parameters: mean = {mu_norm:.2f}, std = {std_norm:.2f}")
```

Prints μ and σ.

---

## ## Overlay Theoretical PDFs with Empirical Distribution

### **Goal:**

Plot fitted distribution curves on top of the real data histogram.

```python
x = np.linspace(min(rainfall_data), max(rainfall_data), 1000)
```

Creates a smooth range of x-values for plotting PDFs.

```python
pdf_norm = norm.pdf(x, mu_norm, std_norm)
```

Computes the normal distribution PDF at each x-value.

```python
plt.figure(figsize=(12, 7))
sns.histplot(rainfall_data, bins=50, kde=True, stat='density', color='skyblue', label='Empirical Data (KDE)')
```

Same histogram as before, normalized to density.

```python
plt.plot(x, pdf_norm, color='red', linestyle='-', label=f'Normal (mu={mu_norm:.2f}, std={std_norm:.2f})')
```

Overlays the Normal distribution curve.

```python
plt.legend()
plt.title('Comparison of Empirical Rainfall Distribution with Fitted Theoretical Distributions')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(min(rainfall_data) - 50, max(rainfall_data) + 50)
plt.show()
```

Final styling + display.

---

## ## Kolmogorov–Smirnov Goodness-of-Fit Test

### **Goal:**

Measure how well each theoretical distribution matches the actual data.

```python
from scipy.stats import kstest
```

Imports the KS test.

```python
rainfall_data_sorted = np.sort(rainfall_data)
```

Sorts data (not required, but doesn't hurt).

```python
ks_stat_norm, p_value_norm = kstest(rainfall_data_sorted, lambda x: norm.cdf(x, loc=mu_norm, scale=std_norm))
```

Runs KS test comparing:

* empirical CDF
* fitted Normal CDF

```python
print(f"Normal Distribution KS Test: Statistic = {ks_stat_norm:.4f}, p-value = {p_value_norm:.4f}")
```

Prints the KS statistic and p-value.

---

## Done. 🎉

This `.md` explanation gives your project the **clarity of a NASA engineering logbook** and the **vibes of a caffeinated AI student**.
If you want, I can also generate:

* PDF version
* README version
* Clean version for submission
* A GitHub-ready notebook

Just throw the next request and we blast. 🚀
