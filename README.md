# 📊 Time Series Analysis Application

A comprehensive Flask web application for analyzing financial time series data, specifically designed for Dow Jones Industrial Average (DJIA) index analysis.

**Live Application:** [https://kopisto.pythonanywhere.com/](https://kopisto.pythonanywhere.com/)

## 📋 Overview

This application performs detailed time series analysis on financial data, including trend analysis, seasonality detection, stochasticity evaluation, risk metrics (VaR), Hurst exponent calculation, autocorrelation analysis, and more.

**Dataset:** Dow Jones Industrial Average (DJIA) Index  
**Period:** 2015-01-04 to 2025-11-23

## ✨ Features

The application includes 18 comprehensive exercises covering various aspects of time series analysis:

### Exercise 1: Time Series Plot
Visualization of the price time series (P) over time.

### Exercise 2: Regression with R²
Linear regression analysis to determine the relationship between time and price, including R² coefficient calculation.

### Exercise 3: Trend Regression Plot
Visual representation of the regression line fitted to the time series data.

### Exercise 4: Seasonality (Cyclicality)
Detection and analysis of seasonal patterns using FFT (Fast Fourier Transform) to identify periodic components.

### Exercise 5: Stochasticity
Analysis of the stochastic (random) component of the time series.

### Exercise 6: Returns Histogram
Distribution analysis of percentage returns with histogram visualization.

### Exercise 7: Residual Regression
Regression analysis on the residuals (S_E) to evaluate stationarity.

### Exercise 8: S+E = P - P'
Decomposition showing the sum of seasonality and stochastic components.

### Exercise 9: Confidence Interval
95% confidence interval calculation for percentage differences/returns.

### Exercise 10: VaR and Percentiles
Value at Risk (VaR) calculation at multiple confidence levels (50%, 60%, 70%, 80%, 85%, 90%, 95%, 99%, 99.5%, 99.9%) and percentile analysis.

### Exercise 11: Hurst Index and Long-Term Memory
Calculation of the Hurst exponent to determine:
- **H > 0.55**: Upward trend (long-term memory)
- **0.45 ≤ H ≤ 0.55**: Random walk (no clear direction)
- **H < 0.45**: Mean-reverting behavior (downward trend)

### Exercise 12: Autocorrelation
Autocorrelation analysis for both the original time series and residuals to detect serial correlation.

### Exercise 13: Phase Diagrams
Visualization of phase space plots showing the relationship between consecutive values.

### Exercise 14: Phase Portrait
Scatter plot of residuals vs. first differences to analyze system dynamics.

### Exercise 15: Moving Average Differences
Stationarity analysis using differences of moving averages.

### Exercise 16: Moving Averages
Visualization of 20-period and 50-period moving averages.

### Exercise 17: ATR and Stability
Average True Range (ATR) calculation and stability assessment based on coefficient of variation.

### Exercise 18: Comprehensive Conclusions
Summary analysis combining all previous exercises with detailed interpretations and conclusions.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Required Packages

Install the required dependencies:

```bash
pip install flask pandas numpy matplotlib scikit-learn statsmodels scipy
```

Or use a requirements file:

```bash
pip install -r requirements.txt
```

### Required Python Packages

- `flask` - Web framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `scikit-learn` - Machine learning (LinearRegression, r2_score)
- `statsmodels` - Statistical modeling (adfuller)
- `scipy` - Scientific computing (stats)

## 📁 Project Structure

```
pyflask/
├── app_flask (1).py    # Main Flask application
├── dataset_processed.csv          # Main dataset (required)
├── dataset_processed_hlc.csv      # High-Low-Close data (optional)
├── templates/                      # HTML templates (required)
│   ├── index.html
│   ├── exercise1.html
│   ├── exercise2.html
│   └── ... (other exercise templates)
└── README.md
```

## 🔧 Configuration

### Data File Paths

The application expects data files at:
- Main dataset: `/home/kopisto/pythonexample/dataset_processed.csv` (PythonAnywhere)
- HLC dataset: `dataset_processed_hlc.csv` (optional, for Exercise 17)

**Note:** For local development, update the file paths in the code to match your local directory structure.

### Required CSV Columns

The main dataset (`dataset_processed.csv`) should contain:
- `t` - Time index
- `P` - Price values
- `P'` - Predicted/trend values (optional, will be computed if missing)
- `S_E` - Seasonality + Error component (optional, will be computed if missing)
- `S` - Seasonality component (optional)
- `PD` - Percentage differences (optional, will be computed if missing)

## 🖥️ Local Development

1. **Clone or download the project**

2. **Install dependencies:**
   ```bash
   pip install flask pandas numpy matplotlib scikit-learn statsmodels scipy
   ```

3. **Prepare your data:**
   - Place `dataset_processed.csv` in the project directory
   - Optionally include `dataset_processed_hlc.csv` for Exercise 17

4. **Update file paths:**
   - Modify the CSV file paths in `app_flask (1).py` to match your local setup
   - Change `/home/kopisto/pythonexample/dataset_processed.csv` to your local path

5. **Run the application:**
   ```bash
   python app_flask\ \(1\).py
   ```

6. **Access the application:**
   - Open your browser and navigate to: `http://127.0.0.1:5000`

## 🌐 Deployment on PythonAnywhere

The application is currently deployed on PythonAnywhere at:
**https://kopisto.pythonanywhere.com/**

### Deployment Steps:

1. **Upload files to PythonAnywhere:**
   - Upload `app_flask (1).py` to your PythonAnywhere account
   - Upload your CSV data files
   - Upload HTML templates to the `templates/` directory

2. **Configure WSGI file:**
   - Create/edit your WSGI file to point to the Flask app
   - Example WSGI configuration:
   ```python
   import sys
   path = '/home/kopisto/pythonexample'
   if path not in sys.path:
       sys.path.append(path)
   
   from app_flask import app as application
   ```

3. **Install packages:**
   - Use PythonAnywhere's Bash console to install required packages
   ```bash
   pip3.10 install --user flask pandas numpy matplotlib scikit-learn statsmodels scipy
   ```

4. **Reload the web app:**
   - Click "Reload" in the PythonAnywhere web app configuration

## 📊 Usage

1. **Main Page (`/`):**
   - Displays a summary of all 18 exercises
   - Shows key metrics and visualizations

2. **Individual Exercises (`/exercise/1` to `/exercise/18`):**
   - Access specific exercises via their routes
   - Each exercise provides detailed analysis and visualizations

## 🔍 Key Features Explained

### Trend Analysis
Uses linear regression to identify and quantify trends in the time series data.

### Seasonality Detection
Employs FFT to detect periodic patterns and seasonal components in the data.

### Risk Metrics
Calculates Value at Risk (VaR) at multiple confidence levels to assess potential losses.

### Stationarity Testing
Uses Augmented Dickey-Fuller test and regression analysis to determine if the time series is stationary.

### Memory Analysis
Hurst exponent calculation to identify long-term memory and trend persistence.

## 🛠️ Technical Details

- **Backend:** Flask (Python web framework)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib (converted to base64 for web display)
- **Statistical Analysis:** scikit-learn, statsmodels, scipy
- **Plot Format:** PNG images embedded as base64 strings

## 📝 Notes

- The application automatically computes missing columns (like `S_E`) if they're not present in the CSV
- All plots are generated dynamically and embedded in HTML as base64 images
- The application uses non-interactive matplotlib backend (`Agg`) for server-side rendering

## 👤 Author

PistolasDev

## 📄 License

This project is for educational and analysis purposes.

## 🔗 Links

- **Live Application:** [https://kopisto.pythonanywhere.com/](https://kopisto.pythonanywhere.com/)
- **PythonAnywhere:** [https://www.pythonanywhere.com/](https://www.pythonanywhere.com/)

---

© 2026, PistolasDev

