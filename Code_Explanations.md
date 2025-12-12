# Line-by-Line Code Explanation: Statistical Model Demo

This document provides a detailed, line-by-line explanation of the code.

## Section 1: Imports

```python
import numpy as np
```
- **Line 1**: Imports the NumPy library and gives it the alias `np`. NumPy is the fundamental package for scientific computing in Python, used here for generating random numbers and calculating means.

```python
import matplotlib.pyplot as plt
```
- **Line 2**: Imports the `pyplot` module from the Matplotlib library with the alias `plt`. This is used for creating the visualizations (histograms and line plots).

```python
from scipy.stats import norm
```
- **Line 3**: Imports the `norm` object from the `scipy.stats` module. This provides functionality related to the Normal (Gaussian) distribution, specifically its Probability Density Function (PDF).

```python
import os
```
- **Line 4**: Imports the `os` module, which provides a portable way of using operating system-dependent functionality, like creating directories.

---

## Section 2: Main Function and Parameters

```python
def main():
```
- **Line 6**: Defines the main function `main()`, which contains the core logic of the script.

```python
    # Parameters
    n_samples = 10000
    mu = 0
    sigma = 1
```
- **Lines 8-10**: Sets the parameters for the simulation:
    - `n_samples`: The number of random data points to generate (10,000 samples).
    - `mu`: The mean (average) of the normal distribution (0).
    - `sigma`: The standard deviation of the normal distribution (1).

```python
    print(f"Generating {n_samples} samples from Normal(mu={mu}, sigma={sigma})...")
```
- **Line 12**: Prints a formatted string to the console, informing the user about the generation process and the parameters being used.

---

## Section 3: Data Generation

```python
    # 1. Generate random data
    data = np.random.normal(mu, sigma, n_samples)
```
- **Line 15**: Generates an array of `n_samples` random numbers drawn from a normal distribution with mean `mu` and standard deviation `sigma`. This is our "empirical" data.

```python
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
```
- **Line 18**: Creates a directory named 'output' to store the generated plots. The `exist_ok=True` argument prevents an error if the directory already exists.

---

## Section 4: Comparing with Theoretical Distribution

```python
    # 2. Compare with theoretical distribution
    plt.figure(figsize=(10, 6))
```
- **Line 21**: Creates a new figure for the plot with a specific size of 10 inches wide by 6 inches tall.

```python
    # Plot histogram of empirical data
    count, bins, ignored = plt.hist(data, 30, density=True, alpha=0.6, color='b', label='Empirical Data')
```
- **Line 24**: Plots a histogram of the generated `data`.
    - `30`: The number of bins (vertical bars).
    - `density=True`: Normalizes the histogram so the total area under it equals 1 (making it a probability density).
    - `alpha=0.6`: Makes the bars semi-transparent (60% opacity).
    - `color='b'`: Sets the bar color to blue.
    - `label='Empirical Data'`: Adds a label for the legend.

```python
    # Plot theoretical PDF
    plt.plot(bins, norm.pdf(bins, mu, sigma), linewidth=2, color='r', label='Theoretical PDF')
```
- **Line 27**: Plots the theoretical Probability Density Function (PDF) curve.
    - `bins`: Uses the bin edges from the histogram as the x-axis values.
    - `norm.pdf(...)`: Calculates the theoretical probability density for each x-value given `mu` and `sigma`.
    - `linewidth=2`: Sets the line thickness.
    - `color='r'`: Sets the line color to red.
    - `label='Theoretical PDF'`: Adds a label for the legend.

```python
    plt.title('Comparison of Empirical Data with Theoretical Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
```
- **Lines 29-33**: Adds a title, x-axis label, y-axis label, displays the legend (using the labels defined earlier), and adds a faint grid to the plot for readability.

```python
    dist_plot_path = 'output/distribution_comparison.png'
    plt.savefig(dist_plot_path)
    print(f"Distribution comparison plot saved to {dist_plot_path}")
    plt.close()
```
- **Lines 35-38**: Defines the file path for the plot, saves the figure to that path, prints a confirmation message, and closes the plot to free up memory.

---

## Section 5: Law of Large Numbers (LLN) Demonstration

```python
    # 3. Law of Large Numbers (LLN) Demonstration
    # Calculate running averages
    running_means = [np.mean(data[:i+1]) for i in range(len(data))]
```
- **Line 42**: Calculates the running average (cumulative mean) of the data.
    - It uses a list comprehension to iterate from `i=0` to the end of the data.
    - For each `i`, it calculates the mean of the data subset `data[:i+1]` (from the start up to the current point).
    - This shows how the sample mean evolves as sample size increases.

```python
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_samples + 1), running_means, color='g', linewidth=1, label='Sample Mean')
```
- **Lines 44-45**: Creates a new figure and plots the running means.
    - x-axis: The sample size `n` (from 1 to `n_samples`).
    - y-axis: The calculated `running_means`.
    - `color='g'`: Sets the line color to green.

```python
    plt.axhline(y=mu, color='r', linestyle='--', linewidth=2, label='Expected Value (Population Mean)')
```
- **Line 46**: Adds a horizontal reference line at the true population mean (`mu`).
    - `linestyle='--'`: Makes it a dashed line.
    - This visualizes the value that the sample mean should converge to.

```python
    plt.title('Law of Large Numbers: Convergence of Sample Mean')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Sample Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
```
- **Lines 48-52**: Adds title, labels, legend, and grid, similar to the previous plot.

```python
    plt.ylim(mu - 0.5, mu + 0.5)  # Zoom in around the mean
```
- **Line 53**: Sets the limits of the y-axis to be centered around the mean (±0.5). This zooms in to clearly show the convergence behavior.

```python
    lln_plot_path = 'output/lln_convergence.png'
    plt.savefig(lln_plot_path)
    print(f"LLN convergence plot saved to {lln_plot_path}")
    plt.close()
```
- **Lines 55-58**: Saves the second plot to the output directory, prints a confirmation, and closes the figure.

---

## Section 6: Execution Entry Point

```python
if __name__ == "__main__":
    main()
```
- **Lines 60-61**: This is a standard Python conditional block.
    - It checks if the script is being run directly (not imported as a module).
    - If true, it calls the `main()` function to execute the program.

---

## Section 7: Visualizations

Here are the plots generated by the code:

### Distribution Comparison
![Distribution Comparison](output/distribution_comparison.png)
*Comparison of the empirical data histogram with the theoretical normal distribution PDF.*

### Law of Large Numbers Convergence
![LLN Convergence](output/lln_convergence.png)
*Demonstration of the sample mean converging to the population mean as sample size increases.*

---

## Appendix: Full Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def main():
    # Parameters
    n_samples = 10000
    mu = 0
    sigma = 1
    
    print(f"Generating {n_samples} samples from Normal(mu={mu}, sigma={sigma})...")
    
    # 1. Generate random data
    data = np.random.normal(mu, sigma, n_samples)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # 2. Compare with theoretical distribution
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of empirical data
    count, bins, ignored = plt.hist(data, 30, density=True, alpha=0.6, color='b', label='Empirical Data')
    
    # Plot theoretical PDF
    plt.plot(bins, norm.pdf(bins, mu, sigma), linewidth=2, color='r', label='Theoretical PDF')
    
    plt.title('Comparison of Empirical Data with Theoretical Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    dist_plot_path = 'output/distribution_comparison.png'
    plt.savefig(dist_plot_path)
    print(f"Distribution comparison plot saved to {dist_plot_path}")
    plt.close()
    
    # 3. Law of Large Numbers (LLN) Demonstration
    # Calculate running averages
    running_means = [np.mean(data[:i+1]) for i in range(len(data))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_samples + 1), running_means, color='g', linewidth=1, label='Sample Mean')
    plt.axhline(y=mu, color='r', linestyle='--', linewidth=2, label='Expected Value (Population Mean)')
    
    plt.title('Law of Large Numbers: Convergence of Sample Mean')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Sample Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(mu - 0.5, mu + 0.5)  # Zoom in around the mean
    
    lln_plot_path = 'output/lln_convergence.png'
    plt.savefig(lln_plot_path)
    print(f"LLN convergence plot saved to {lln_plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
```

