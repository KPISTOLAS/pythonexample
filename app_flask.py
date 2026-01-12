from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from math import log

app = Flask(__name__)

def plot_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

@app.route('/')
def index():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    results = {}

    # Exercise 1 - Time series plot
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df['t'], df['P'], marker='o', linestyle='-', markersize=2)
    ax1.set_title("Χρονοσειρά της Τιμής (P)")
    ax1.set_xlabel("Χρόνος (t)")
    ax1.set_ylabel("Τιμή (P)")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    results['exercise1_plot'] = plot_to_base64(fig1)
    plt.close(fig1)

    # Exercise 2 & 3 - Regression
    t = df['t'].values.reshape(-1, 1)
    P = df['P'].values
    regression = LinearRegression()
    regression.fit(t, P)
    P_pred = regression.predict(t)
    results['exercise2'] = {
        'a': f"{regression.intercept_:.2f}",
        'b': f"{regression.coef_[0]:.2f}",
        'r2': f"{r2_score(P, P_pred):.4f}",
        'correlation': "R² > 0.5, έχουν συσχέτιση." if r2_score(P, P_pred) > 0.5 else "R² ≤ 0.5, δεν έχουν συσχέτιση.",
        'stationarity': "Η χρονοσειρά δεν είναι στάσιμη." if regression.coef_[0] != 0 else "Η χρονοσειρά είναι στάσιμη."
    }
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.scatter(t, P, alpha=0.6, color='blue', s=20)
    ax3.plot(t, P_pred, color='red', linewidth=2, label=f'P = {results["exercise2"]["a"]} + {results["exercise2"]["b"]}*t')
    ax3.set_title(f"Παλινδρόμηση Τάσης (R² = {results['exercise2']['r2']})")
    ax3.set_xlabel("Χρόνος (t)")
    ax3.set_ylabel("Τιμή (P)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    results['exercise3_plot'] = plot_to_base64(fig3)
    plt.close(fig3)

    # Exercise 4 - Seasonality
    t_vals = df['t'].values
    P_vals = df['P'].values
    n = len(t_vals)
    S_E_values = df["S_E"].values if "S_E" in df.columns else P_vals
    S_E_fft = S_E_values[-512:] if len(S_E_values) >= 512 else S_E_values
    n_fft = len(S_E_fft)
    fa_result = np.fft.fft(S_E_fft)
    amplitudes = np.abs(fa_result) / n_fft
    peaks = []
    for i in range(1, min(n_fft//2, 100)):
        if amplitudes[i] > 0:
            period_candidate = n_fft / i if i > 0 else 1
            peaks.append((i, amplitudes[i], period_candidate))
    peaks.sort(key=lambda x: x[1], reverse=True)
    MIN_PERIOD = 2
    MAX_PERIOD = 52
    peak_index = None
    max_amplitude = 0
    for p_idx, amp, period in peaks[:20]:
        if MIN_PERIOD <= period <= MAX_PERIOD:
            peak_index = p_idx
            max_amplitude = amp
            break
    if peak_index is None:
        PERIOD = 6
        peak_index = int(n_fft / PERIOD) if n_fft >= PERIOD else 1
        max_amplitude = amplitudes[peak_index] if peak_index < len(amplitudes) else 0
    else:
        PERIOD = max(int(n_fft / peak_index), MIN_PERIOD) if peak_index > 0 else MIN_PERIOD
        PERIOD = min(max(PERIOD, MIN_PERIOD), MAX_PERIOD)
    MAX_VALUES = 20
    seasonal_indices = []
    for pos in range(PERIOD):
        positions = []
        for idx in range(pos, n, PERIOD):
            positions.append(idx)
        if positions:
            positions_to_use = positions[:MAX_VALUES]
            values_sum = sum(S_E_values[idx] if idx < n else 0 for idx in positions_to_use)
            avg_value = values_sum / MAX_VALUES
            seasonal_indices.append(avg_value)
        else:
            seasonal_indices.append(0)
    S = []
    for idx in range(n):
        pos_in_cycle = idx % PERIOD
        S.append(seasonal_indices[pos_in_cycle])
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(df['t'], S, label='S')
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('S')
    ax4.set_title('Kiklikotita (Seasonality)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    plt.tight_layout()
    results['exercise4'] = {
        'period': PERIOD,
        'peak_index': peak_index,
        'max_amplitude': f"{max_amplitude:.2f}",
        'plot': plot_to_base64(fig4)
    }
    plt.close(fig4)

    # Exercise 5 - Stochasticity
    T = df['t'].values
    S_vals = df['S'].values if 'S' in df.columns else np.zeros(len(T))
    S_E_vals = df['S_E'].values if 'S_E' in df.columns else df['P'].values
    E = S_E_vals - S_vals
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    ax5.plot(T, E)
    ax5.set_xlabel('Time (t)')
    ax5.set_ylabel('E')
    ax5.set_title('Stoxastkotita')
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    results['exercise5_plot'] = plot_to_base64(fig5)
    plt.close(fig5)

    # Exercise 6 - Histogram of Returns
    p_diffl = ((df["P"] - df["P"].shift(1)) / df["P"].shift(1)) * 100
    p_diffl_clean = p_diffl.dropna()
    pososta = [-np.inf, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf]
    counts, bins = np.histogram(p_diffl_clean, bins=pososta)
    intervals = ['(...-5)', '(-4,-3)', '(-3,-2)', '(-2,-1)', '(-1,0)', '(0,0...1)', '(1...2)', '(2...3)', '(3...4)', '(4...)']
    if len(counts) == 11 and len(intervals) == 10:
        counts_selected = np.concatenate([counts[0:1], counts[2:]])
        counts = counts_selected
    intervals_8 = intervals[1:9]
    midpoints = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    counts_8 = counts[1:9]
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    ax6.bar(midpoints, counts_8, width=0.8, edgecolor='black', alpha=0.7, color='skyblue', label='Returns')
    ax6.set_xlabel('Returns (Midpoints)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Histogram of Returns')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    results['exercise6_plot'] = plot_to_base64(fig6)
    plt.close(fig6)

    # Exercise 7 - Regression on residuals
    residuals = df['S_E'].values if 'S_E' in df.columns else df['P'].values
    regression_res = LinearRegression()
    regression_res.fit(t, residuals)
    P_pred_res = regression_res.predict(t)
    r2_res = r2_score(residuals, P_pred_res)
    results['exercise7'] = {
        'a': f"{regression_res.intercept_:.2f}",
        'b': f"{regression_res.coef_[0]:.2f}",
        'r2': f"{r2_res:.4f}",
        'correlation': "R² > 0.5, έχουν συσχέτιση." if r2_res > 0.5 else "R² ≤ 0.5, δεν έχουν συσχέτιση.",
        'stationarity': "Η νέα χρονοσειρά (στοχαστικότητα) είναι στάσιμη." if regression_res.coef_[0] != 0 else "Η νέα χρονοσειρά (στοχαστικότητα) δεν είναι στάσιμη."
    }

    # Exercise 8 - S+E plot
    P_trend = df["P'"].values if "P'" in df.columns else P
    df['S_E'] = P - P_trend
    fig8, ax8 = plt.subplots(figsize=(12, 5))
    ax8.plot(df['t'], df['S_E'])
    ax8.set_xlabel('Time (t)')
    ax8.set_ylabel('S+E = P - P\'')
    ax8.set_title('S+E = P - P\'')
    ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    results['exercise8_plot'] = plot_to_base64(fig8)
    plt.close(fig8)

    # Exercise 9 - Confidence Interval
    if 'PD' in df.columns:
        DP = df['PD'].dropna().values
    else:
        p_diffl = ((df["P"] - df["P"].shift(1)) / df["P"].shift(1)) * 100
        p_diffl_clean = p_diffl.dropna()
        DP = p_diffl_clean.values
    if len(DP) > 0 and not np.all(np.isnan(DP)):
        DP = DP[~np.isnan(DP)]
        if len(DP) > 0:
            mean = np.mean(DP)
            std = np.std(DP, ddof=0)
            n_dp = len(DP)
            confidence_level = 0.95
            t1 = std / np.sqrt(n_dp)
            t_value = stats.t.ppf((1 + confidence_level) / 2, df=n_dp-1)
            margin_error = t_value * t1
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
            results['exercise9'] = {
                'mean': f"{mean:.2f}",
                'std': f"{std:.2f}",
                'ci_lower': f"{ci_lower:.2f}",
                'ci_upper': f"{ci_upper:.2f}"
            }
        else:
            results['exercise9'] = {'mean': "N/A", 'std': "N/A", 'ci_lower': "N/A", 'ci_upper': "N/A"}
    else:
        results['exercise9'] = {'mean': "N/A", 'std': "N/A", 'ci_lower': "N/A", 'ci_upper': "N/A"}

    # Exercise 10 - VaR
    df_returns = df.copy()
    df_returns['Returns'] = df_returns['P'].pct_change() * 100
    df_returns = df_returns.dropna()
    confidence_levels = [50, 60, 70, 80, 85, 90, 95, 99, 99.5, 99.9]
    returns = df_returns['Returns'].values
    var_results = {}
    for conf in confidence_levels:
        percentile = 100 - conf
        var = -np.percentile(returns, percentile)
        var_results[conf] = var
    percentile_mapping = {80: 80, 90: 90, 110: 10, 120: 20}
    percentile_values = {}
    for label, p in percentile_mapping.items():
        percentile_values[label] = np.percentile(returns, p)
    fig10, ax10 = plt.subplots(figsize=(14, 7))
    ax10.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='skyblue', label='Αποδόσεις')
    percentile_colors = ['green', 'blue', 'purple', 'orange']
    for i, (label, value) in enumerate(percentile_values.items()):
        actual_p = percentile_mapping[label]
        if label > 100:
            ax10.axvline(x=value, color=percentile_colors[i], linestyle='-',
                      linewidth=2.5, label=f'Percentile {label} ({actual_p}th) = {value:.2f}%', alpha=0.8)
        else:
            ax10.axvline(x=value, color=percentile_colors[i], linestyle='-',
                      linewidth=2.5, label=f'Percentile {label} = {value:.2f}%', alpha=0.8)
    ax10.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax10.set_title("Κατανομή Αποδόσεων και Percentiles (80, 90, 110, 120)", fontsize=14, fontweight='bold')
    ax10.set_xlabel("Απόδοση (%)", fontsize=12)
    ax10.set_ylabel("Συχνότητα", fontsize=12)
    ax10.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax10.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    results['exercise10'] = {
        'plot': plot_to_base64(fig10),
        'var_results': var_results
    }
    plt.close(fig10)

    # Exercise 11 - Hurst Index
    P_hurst = df['P'].values
    n_hurst = len(P_hurst)
    if n_hurst >= 567:
        P_hurst = P_hurst[-567:]
        n_hurst = 567
    P_mean_hurst = np.mean(P_hurst)
    Deviations_hurst = P_hurst - P_mean_hurst
    R_hurst = np.max(Deviations_hurst) - np.min(Deviations_hurst)
    S_hurst = np.std(Deviations_hurst)
    RS_hurst = R_hurst / S_hurst if S_hurst > 0 else 0
    segment_size = n_hurst // 3
    Aggregate_Deviations0 = []
    Aggregate_Deviations1 = []
    Aggregate_Deviations2 = []
    for i in range(segment_size):
        if i == 0:
            Aggregate_Deviations0.append(Deviations_hurst[i])
        else:
            Aggregate_Deviations0.append(Deviations_hurst[i] + Aggregate_Deviations0[i-1])
    for i in range(segment_size, 2*segment_size):
        idx = i - segment_size
        if idx == 0:
            Aggregate_Deviations1.append(Deviations_hurst[i])
        else:
            Aggregate_Deviations1.append(Deviations_hurst[i] + Aggregate_Deviations1[idx-1])
    for i in range(2*segment_size, n_hurst):
        idx = i - 2*segment_size
        if idx == 0:
            Aggregate_Deviations2.append(Deviations_hurst[i])
        else:
            Aggregate_Deviations2.append(Deviations_hurst[i] + Aggregate_Deviations2[idx-1])
    R0_hurst = np.max(Aggregate_Deviations0) - np.min(Aggregate_Deviations0)
    S0_hurst = np.std(Aggregate_Deviations0) if len(Aggregate_Deviations0) > 1 else 1
    RS0_hurst = R0_hurst / S0_hurst if S0_hurst > 0 else 0
    R1_hurst = np.max(Aggregate_Deviations1) - np.min(Aggregate_Deviations1)
    S1_hurst = np.std(Aggregate_Deviations1) if len(Aggregate_Deviations1) > 1 else 1
    RS1_hurst = R1_hurst / S1_hurst if S1_hurst > 0 else 0
    R2_hurst = np.max(Aggregate_Deviations2) - np.min(Aggregate_Deviations2)
    S2_hurst = np.std(Aggregate_Deviations2) if len(Aggregate_Deviations2) > 1 else 1
    RS2_hurst = R2_hurst / S2_hurst if S2_hurst > 0 else 0
    RS_mean_hurst = (RS0_hurst + RS1_hurst + RS2_hurst) / 3 if (RS0_hurst > 0 and RS1_hurst > 0 and RS2_hurst > 0) else RS_hurst
    val1_hurst = 1
    val2_hurst = log(n_hurst)
    val3_hurst = log(RS_mean_hurst) if RS_mean_hurst > 0 else log(RS_hurst) if RS_hurst > 0 else 0
    val4_hurst = log(RS_hurst) if RS_hurst > 0 else 0
    if val2_hurst > val1_hurst and val4_hurst > 0:
        H_hurst = (val4_hurst - val3_hurst) / (val2_hurst - val1_hurst)
    else:
        H_hurst = 0.5
    if 0.45 <= H_hurst <= 0.55:
        interpretation_hurst = f"H = {H_hurst:.4f} (περίπου 0.45-0.55): Ο Hurst είναι ΑΔΙΑΦΟΡΟΣ. Η χρονοσειρά ακολουθεί random walk. Δεν υπάρχει σαφής κατεύθυνση."
        memory_type_hurst = "ΑΔΙΑΦΟΡΟΣ"
    elif H_hurst < 0.45:
        interpretation_hurst = f"H = {H_hurst:.4f} < 0.45: Η χρονοσειρά έχει ΚΑΘΟΔΙΚΗ ΠΟΡΕΙΑ. Υπάρχει mean-reverting συμπεριφορά. Οι τιμές τείνουν να επιστρέφουν στον μέσο όρο."
        memory_type_hurst = "ΚΑΘΟΔΙΚΗ ΠΟΡΕΙΑ"
    else:
        interpretation_hurst = f"H = {H_hurst:.4f} > 0.55: Η χρονοσειρά έχει ΑΝΟΔΙΚΗ ΠΟΡΕΙΑ. Υπάρχει μακροπρόθεσμη μνήμη. Οι τάσεις τείνουν να συνεχίζονται."
        memory_type_hurst = "ΑΝΟΔΙΚΗ ΠΟΡΕΙΑ (ΜΑΚΡΟΠΡΟΘΕΣΜΗ ΜΝΗΜΗ)"
    results['exercise11'] = {
        'H': f"{H_hurst:.4f}",
        'RS0': f"{RS0_hurst:.4f}",
        'RS1': f"{RS1_hurst:.4f}",
        'RS2': f"{RS2_hurst:.4f}",
        'RS_mean': f"{RS_mean_hurst:.4f}",
        'RS': f"{RS_hurst:.4f}",
        'interpretation': interpretation_hurst,
        'memory_type': memory_type_hurst
    }

    # Exercise 12 - Autocorrelation
    def autocorr(x, max_lag=50):
        n = len(x)
        mean = np.mean(x)
        autocorrs = []
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorrs.append(1.0)
            else:
                numerator = np.sum((x[lag:] - mean) * (x[:-lag] - mean))
                denominator = np.sum((x - mean) ** 2)
                if denominator > 0:
                    autocorrs.append(numerator / denominator)
                else:
                    autocorrs.append(0.0)
        return np.array(autocorrs)
    P_autocorr = df['P'].values
    autocorr_P = autocorr(P_autocorr, max_lag=50)
    t = df['t'].values.reshape(-1, 1)
    P = df['P'].values
    reg = LinearRegression()
    reg.fit(t, P)
    P_pred = reg.predict(t)
    df['S_E'] = P - P_pred
    residuals_autocorr = df['S_E'].values
    autocorr_residuals = autocorr(residuals_autocorr, max_lag=50)
    lags = np.arange(len(autocorr_P))
    fig12, axes12 = plt.subplots(2, 1, figsize=(10, 10))
    axes12[0].plot(lags, autocorr_P, marker='o', linestyle='-', markersize=3, color='blue')
    axes12[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes12[0].set_title("Αυτοσυσχέτιση Αρχικής Χρονοσειράς (P)")
    axes12[0].set_xlabel("Lag")
    axes12[0].set_ylabel("Αυτοσυσχέτιση")
    axes12[0].grid(True, alpha=0.3)
    axes12[0].set_xlim(0, 50)
    axes12[1].plot(lags, autocorr_residuals, marker='o', linestyle='-', markersize=3, color='red')
    axes12[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes12[1].set_title("Αυτοσυσχέτιση Χρονοσειράς Στοχαστικότητας (Κατάλοιπα)")
    axes12[1].set_xlabel("Lag")
    axes12[1].set_ylabel("Αυτοσυσχέτιση")
    axes12[1].grid(True, alpha=0.3)
    axes12[1].set_xlim(0, 50)
    plt.tight_layout()
    results['exercise12'] = {
        'ac_original_lag1': f"{autocorr_P[1]:.4f}",
        'ac_original_max': f"{np.max(autocorr_P[1:]):.4f}",
        'ac_residuals_lag1': f"{autocorr_residuals[1]:.4f}",
        'ac_residuals_max': f"{np.max(autocorr_residuals[1:]):.4f}",
        'plot': plot_to_base64(fig12)
    }
    plt.close(fig12)

    # Exercise 13 - Phase plots
    s_e_vals = df['S_E'].values if 'S_E' in df.columns else df['P'].values
    diff_S_E = np.diff(s_e_vals)
    X_n = s_e_vals[:-1]
    X_n1 = s_e_vals[1:]
    fig13, axes13 = plt.subplots(3, 1, figsize=(10, 12))
    axes13[0].plot(df['t'][:-1], X_n, marker='o', linestyle='-', markersize=3, alpha=0.7, color='blue')
    axes13[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes13[0].set_title("E_n (Χρονοσειρά Καταλοίπων)")
    axes13[0].set_xlabel("Χρόνος (t)")
    axes13[0].set_ylabel("X_n")
    axes13[0].grid(True, alpha=0.3)
    axes13[1].plot(df['t'][1:], X_n1, marker='o', linestyle='-', markersize=3, alpha=0.7, color='green')
    axes13[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes13[1].set_title("E_{n+1} (Χρονοσειρά Καταλοίπων - Επόμενη Περίοδος)")
    axes13[1].set_xlabel("Χρόνος (t)")
    axes13[1].set_ylabel("X_{n+1}")
    axes13[1].grid(True, alpha=0.3)
    axes13[2].plot(df['t'][1:], diff_S_E, marker='o', linestyle='-', markersize=3, alpha=0.7, color='red')
    axes13[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes13[2].set_title("X_{n+1} - X_n (Πρώτες Διαφορές Καταλοίπων)")
    axes13[2].set_xlabel("Χρόνος (t)")
    axes13[2].set_ylabel("X_{n+1} - X_n")
    axes13[2].grid(True, alpha=0.3)
    plt.tight_layout()
    results['exercise13_plot'] = plot_to_base64(fig13)
    plt.close(fig13)

    # Exercise 14 - Phase portrait
    diff_residuals = np.diff(s_e_vals)
    X_n_14 = s_e_vals[:-1]
    X_diff = diff_residuals
    fig14, ax14 = plt.subplots(figsize=(10, 8))
    ax14.scatter(X_n_14, X_diff, alpha=0.5, s=10, color='blue')
    ax14.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax14.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax14.set_xlabel("Χρονοσειρά βήματος 5 (X_n = Κατάλοιπα/S_E)", fontsize=12)
    ax14.set_ylabel("Πρώτες Διαφορές (X_{n+1} - X_n)", fontsize=12)
    ax14.set_title("Φασικό Πορτραίτο: Κατάλοιπα vs Πρώτες Διαφορές", fontsize=14, fontweight='bold')
    ax14.grid(True, alpha=0.3)
    plt.tight_layout()
    results['exercise14_plot'] = plot_to_base64(fig14)
    plt.close(fig14)

    # Exercise 15 - Moving average differences
    df_ma = df.copy()
    df_ma['MA_20'] = df_ma['P'].rolling(window=20, min_periods=1).mean()
    df_ma['dMA_20'] = df_ma['MA_20'].diff()
    df_clean_ma = df_ma[['t', 'dMA_20']].dropna()
    t_ma = df_clean_ma['t'].values.reshape(-1, 1)
    regression_ma = LinearRegression()
    regression_ma.fit(t_ma, df_clean_ma['dMA_20'])
    results['exercise15'] = {
        'a': f"{regression_ma.intercept_:.2f}",
        'b': f"{regression_ma.coef_[0]:.2f}",
        'stationarity': "Η χρονοσειρά δεν είναι στάσιμη." if regression_ma.coef_[0] != 0 else "Η χρονοσειρά είναι στάσιμη."
    }

    # Exercise 16 - Moving averages
    t_16 = df['t'].values
    df_16 = df.copy()
    df_16['MO_20'] = df_16['P'].rolling(window=20, min_periods=1).mean()
    df_16['MO_50'] = df_16['P'].rolling(window=50, min_periods=1).mean()
    fig16, axs16 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs16[0].plot(t_16, df_16['MO_20'], label='MO_20')
    axs16[0].set_ylabel('MO_20')
    axs16[0].set_title('Kinitos Mesos Oro 20')
    axs16[0].grid(True, alpha=0.3)
    axs16[0].legend()
    axs16[1].plot(t_16, df_16['MO_50'], label='MO_50', color='orange')
    axs16[1].set_xlabel('Time (t)')
    axs16[1].set_ylabel('MO_50')
    axs16[1].set_title('Kinitos Mesos Oro 50')
    axs16[1].grid(True, alpha=0.3)
    axs16[1].legend()
    plt.tight_layout()
    results['exercise16_plot'] = plot_to_base64(fig16)
    plt.close(fig16)

    # Exercise 17 - ATR and stability
    import os
    hlc_path = "../dataset_processed_hlc.csv"
    if not os.path.exists(hlc_path):
        hlc_path = "dataset_processed_hlc.csv"
    if os.path.exists(hlc_path):
        df2 = pd.read_csv(hlc_path)
    else:
        df2 = df.copy()
        df2['Close'] = df['P'].values
        df2['High'] = df['P'].values * 1.01
        df2['Low'] = df['P'].values * 0.99
    P_17 = df['P'].values
    cl = df2['Close'].values[-500:] if len(df2) >= 500 else df2['Close'].values
    hi = df2['High'].values[-500:] if len(df2) >= 500 else df2['High'].values
    lo = df2['Low'].values[-500:] if len(df2) >= 500 else df2['Low'].values
    n2 = len(cl)
    TR = []
    period_17 = 50
    for i in range(n2):
        if i == 0:
            TR.append(hi[i]-lo[i])
        else:
            x1 = abs(hi[i]-lo[i])
            x2 = abs(hi[i]-cl[i-1])
            x3 = abs(lo[i]-cl[i-1])
            TR.append(max(x1, x2, x3))
    ATR = np.mean(TR)
    mean_all = np.mean(P_17)
    std_all = np.std(P_17)
    all_cv = std_all / mean_all
    if all_cv < 0.3:
        stability = "Η χρονοσειρά είναι σταθερή."
    elif all_cv < 0.5:
        stability = "Η χρονοσειρά είναι ασθενώς ασταθής."
    else:
        stability = "Η χρονοσειρά είναι ασταθής."
    results['exercise17'] = {
        'atr': f"{ATR:.2f}",
        'stability': stability
    }

    return render_template('index.html', results=results)

@app.route('/exercise/1')
def exercise1():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['t'], df['P'], marker='o', linestyle='-', markersize=2)
    ax.set_title("Χρονοσειρά της Τιμής (P)")
    ax.set_xlabel("Χρόνος (t)")
    ax.set_ylabel("Τιμή (P)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)
    return render_template('exercise1.html', plot=img)

@app.route('/exercise/2')
def exercise2():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    t = df['t'].values.reshape(-1, 1)
    P = df['P'].values

    regression = LinearRegression()
    regression.fit(t, P)
    P_pred = regression.predict(t)

    a = f"{regression.intercept_:.2f}"
    b = f"{regression.coef_[0]:.2f}"
    r2 = r2_score(P, P_pred)
    r2_str = f"{r2:.4f}"

    correlation = "R² > 0.5, έχουν συσχέτιση." if r2 > 0.5 else "R² ≤ 0.5, δεν έχουν συσχέτιση."
    stationarity = "Η χρονοσειρά δεν είναι στάσιμη." if regression.coef_[0] != 0 else "Η χρονοσειρά είναι στάσιμη."

    return render_template('exercise2.html', a=a, b=b, r2=r2_str, correlation=correlation, stationarity=stationarity)

@app.route('/exercise/3')
def exercise3():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    t = df['t'].values.reshape(-1, 1)
    P = df['P'].values

    regression = LinearRegression()
    regression.fit(t, P)
    P_pred = regression.predict(t)

    a = f"{regression.intercept_:.2f}"
    b = f"{regression.coef_[0]:.2f}"
    r2 = f"{r2_score(P, P_pred):.4f}"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(t, P, alpha=0.6, color='blue', s=20)
    ax.plot(t, P_pred, color='red', linewidth=2, label=f'P = {a} + {b}*t')
    ax.set_title(f"Παλινδρόμηση Τάσης (R² = {r2})")
    ax.set_xlabel("Χρόνος (t)")
    ax.set_ylabel("Τιμή (P)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)
    return render_template('exercise3.html', a=a, b=b, r2=r2, plot=img)

@app.route('/exercise/4')
def exercise4():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    t = df['t'].values
    P = df['P'].values
    n = len(t)
    S_E_values = df["S_E"].values if "S_E" in df.columns else P

    S_E_fft = S_E_values[-512:] if len(S_E_values) >= 512 else S_E_values
    n_fft = len(S_E_fft)
    fa_result = np.fft.fft(S_E_fft)
    amplitudes = np.abs(fa_result) / n_fft

    peaks = []
    for i in range(1, min(n_fft//2, 100)):
        if amplitudes[i] > 0:
            period_candidate = n_fft / i if i > 0 else 1
            peaks.append((i, amplitudes[i], period_candidate))

    peaks.sort(key=lambda x: x[1], reverse=True)

    MIN_PERIOD = 2
    MAX_PERIOD = 52
    peak_index = None
    max_amplitude = 0

    for p_idx, amp, period in peaks[:20]:
        if MIN_PERIOD <= period <= MAX_PERIOD:
            peak_index = p_idx
            max_amplitude = amp
            break

    if peak_index is None:
        PERIOD = 6
        peak_index = int(n_fft / PERIOD) if n_fft >= PERIOD else 1
        max_amplitude = amplitudes[peak_index] if peak_index < len(amplitudes) else 0
    else:
        PERIOD = max(int(n_fft / peak_index), MIN_PERIOD) if peak_index > 0 else MIN_PERIOD
        PERIOD = min(max(PERIOD, MIN_PERIOD), MAX_PERIOD)

    MAX_VALUES = 20
    seasonal_indices = []
    for pos in range(PERIOD):
        positions = []
        for idx in range(pos, n, PERIOD):
            positions.append(idx)
        if positions:
            positions_to_use = positions[:MAX_VALUES]
            values_sum = sum(S_E_values[idx] if idx < n else 0 for idx in positions_to_use)
            avg_value = values_sum / MAX_VALUES
            seasonal_indices.append(avg_value)
        else:
            seasonal_indices.append(0)

    S = []
    for idx in range(n):
        pos_in_cycle = idx % PERIOD
        S.append(seasonal_indices[pos_in_cycle])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['t'], S, label='S')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('S')
    ax.set_title('Kiklikotita (Seasonality)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)

    return render_template('exercise4.html', period=PERIOD, peak_index=peak_index, max_amplitude=f"{max_amplitude:.2f}", plot=img)

@app.route('/exercise/5')
def exercise5():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    T = df['t'].values
    S = df['S'].values if 'S' in df.columns else np.zeros(len(T))
    S_E = df['S_E'].values if 'S_E' in df.columns else df['P'].values

    E = S_E - S

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(T, E)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('E')
    ax.set_title('Stoxastkotita')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)
    return render_template('exercise5.html', plot=img)

@app.route('/exercise/6')
def exercise6():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    p_diffl = ((df["P"] - df["P"].shift(1)) / df["P"].shift(1)) * 100
    p_diffl_clean = p_diffl.dropna()

    pososta = [-np.inf, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf]
    counts, bins = np.histogram(p_diffl_clean, bins=pososta)

    intervals = ['(...-5)', '(-4,-3)', '(-3,-2)', '(-2,-1)', '(-1,0)', '(0,0...1)', '(1...2)', '(2...3)', '(3...4)', '(4...)']

    if len(counts) == 11 and len(intervals) == 10:
        counts_selected = np.concatenate([counts[0:1], counts[2:]])
        counts = counts_selected

    intervals_8 = intervals[1:9]
    midpoints = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    counts_8 = counts[1:9]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(midpoints, counts_8, width=0.8, edgecolor='black', alpha=0.7, color='skyblue', label='Returns')
    ax.set_xlabel('Returns (Midpoints)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Returns')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)
    return render_template('exercise6.html', plot=img)

@app.route('/exercise/7')
def exercise7():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    residuals = df['S_E'].values if 'S_E' in df.columns else df['P'].values
    t = df['t'].values.reshape(-1, 1)

    regression = LinearRegression()
    regression.fit(t, residuals)
    P_pred = regression.predict(t)

    a = f"{regression.intercept_:.2f}"
    b = f"{regression.coef_[0]:.2f}"
    r2 = r2_score(residuals, P_pred)
    r2_str = f"{r2:.4f}"

    correlation = "R² > 0.5, έχουν συσχέτιση." if r2 > 0.5 else "R² ≤ 0.5, δεν έχουν συσχέτιση."
    stationarity = "Η νέα χρονοσειρά (στοχαστικότητα) είναι στάσιμη." if regression.coef_[0] != 0 else "Η νέα χρονοσειρά (στοχαστικότητα) δεν είναι στάσιμη."

    return render_template('exercise7.html', a=a, b=b, r2=r2_str, correlation=correlation, stationarity=stationarity)

@app.route('/exercise/8')
def exercise8():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    t = df['t'].values.reshape(-1, 1)
    P = df['P'].values
    P_trend = df["P'"].values if "P'" in df.columns else P

    df['S_E'] = P - P_trend

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['t'], df['S_E'])
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('S+E = P - P\'')
    ax.set_title('S+E = P - P\'')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)
    return render_template('exercise8.html', plot=img)

@app.route('/exercise/9')
def exercise9():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")

    # ypologizoume tis posostiakes prwtes diaforeseis
    if 'PD' in df.columns:
        DP = df['PD'].dropna().values
    else:
        # Calculate percentage differences like in askisi6.py
        p_diffl = ((df["P"] - df["P"].shift(1)) / df["P"].shift(1)) * 100
        p_diffl_clean = p_diffl.dropna()
        DP = p_diffl_clean.values

    # Check if we have valid data
    if len(DP) == 0 or np.all(np.isnan(DP)):
        return render_template('exercise9.html',
                             mean="N/A",
                             std="N/A",
                             ci_lower="N/A",
                             ci_upper="N/A")

    # Remove any remaining NaN values
    DP = DP[~np.isnan(DP)]

    if len(DP) == 0:
        return render_template('exercise9.html',
                             mean="N/A",
                             std="N/A",
                             ci_lower="N/A",
                             ci_upper="N/A")

    mean = np.mean(DP)
    std = np.std(DP, ddof=0)
    n = len(DP)

    confidence_level = 0.95
    t1 = std / np.sqrt(n)
    t_value = stats.t.ppf((1 + confidence_level) / 2, df=n-1)

    margin_error = t_value * t1
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error

    return render_template('exercise9.html',
                         mean=f"{mean:.2f}",
                         std=f"{std:.2f}",
                         ci_lower=f"{ci_lower:.2f}",
                         ci_upper=f"{ci_upper:.2f}")

@app.route('/exercise/10')
def exercise10():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    df['Returns'] = df['P'].pct_change() * 100
    df = df.dropna()

    confidence_levels = [50, 60, 70, 80, 85, 90, 95, 99, 99.5, 99.9]
    returns = df['Returns'].values

    var_results = {}
    for conf in confidence_levels:
        percentile = 100 - conf
        var = -np.percentile(returns, percentile)
        var_results[conf] = var

    percentile_mapping = {80: 80, 90: 90, 110: 10, 120: 20}
    percentile_values = {}
    for label, p in percentile_mapping.items():
        percentile_values[label] = np.percentile(returns, p)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='skyblue', label='Αποδόσεις')

    percentile_colors = ['green', 'blue', 'purple', 'orange']
    for i, (label, value) in enumerate(percentile_values.items()):
        actual_p = percentile_mapping[label]
        if label > 100:
            ax.axvline(x=value, color=percentile_colors[i], linestyle='-',
                      linewidth=2.5, label=f'Percentile {label} ({actual_p}th) = {value:.2f}%', alpha=0.8)
        else:
            ax.axvline(x=value, color=percentile_colors[i], linestyle='-',
                      linewidth=2.5, label=f'Percentile {label} = {value:.2f}%', alpha=0.8)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_title("Κατανομή Αποδόσεων και Percentiles (80, 90, 110, 120)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Απόδοση (%)", fontsize=12)
    ax.set_ylabel("Συχνότητα", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)
    return render_template('exercise10.html', plot=img, var_results=var_results)

@app.route('/exercise/11')
def exercise11():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    P = df['P'].values
    n = len(P)

    # Use last 567 points if available, otherwise use all
    if n >= 567:
        P = P[-567:]
        n = 567

    P_mean = np.mean(P)
    Deviations = P - P_mean
    R = np.max(Deviations) - np.min(Deviations)
    S = np.std(Deviations)
    RS = R / S

    # Split into 3 segments
    segment_size = n // 3
    Aggregate_Deviations0 = []
    Aggregate_Deviations1 = []
    Aggregate_Deviations2 = []

    # Segment 0
    for i in range(segment_size):
        if i == 0:
            Aggregate_Deviations0.append(Deviations[i])
        else:
            Aggregate_Deviations0.append(Deviations[i] + Aggregate_Deviations0[i-1])

    # Segment 1
    for i in range(segment_size, 2*segment_size):
        idx = i - segment_size
        if idx == 0:
            Aggregate_Deviations1.append(Deviations[i])
        else:
            Aggregate_Deviations1.append(Deviations[i] + Aggregate_Deviations1[idx-1])

    # Segment 2
    for i in range(2*segment_size, n):
        idx = i - 2*segment_size
        if idx == 0:
            Aggregate_Deviations2.append(Deviations[i])
        else:
            Aggregate_Deviations2.append(Deviations[i] + Aggregate_Deviations2[idx-1])

    # Calculate R/S for each segment
    R0 = np.max(Aggregate_Deviations0) - np.min(Aggregate_Deviations0)
    S0 = np.std(Aggregate_Deviations0) if len(Aggregate_Deviations0) > 1 else 1
    RS0 = R0 / S0 if S0 > 0 else 0

    R1 = np.max(Aggregate_Deviations1) - np.min(Aggregate_Deviations1)
    S1 = np.std(Aggregate_Deviations1) if len(Aggregate_Deviations1) > 1 else 1
    RS1 = R1 / S1 if S1 > 0 else 0

    R2 = np.max(Aggregate_Deviations2) - np.min(Aggregate_Deviations2)
    S2 = np.std(Aggregate_Deviations2) if len(Aggregate_Deviations2) > 1 else 1
    RS2 = R2 / S2 if S2 > 0 else 0

    RS_mean = (RS0 + RS1 + RS2) / 3 if (RS0 > 0 and RS1 > 0 and RS2 > 0) else RS

    # Calculate Hurst exponent
    val1 = 1
    val2 = log(n)
    val3 = log(RS_mean) if RS_mean > 0 else log(RS)
    val4 = log(RS) if RS > 0 else 0

    if val2 > val1 and val4 > 0:
        H = (val4 - val3) / (val2 - val1)
    else:
        H = 0.5

    # Interpretation
    if 0.45 <= H <= 0.55:
        interpretation = f"H = {H:.4f} (περίπου 0.45-0.55): Ο Hurst είναι ΑΔΙΑΦΟΡΟΣ. Η χρονοσειρά ακολουθεί random walk. Δεν υπάρχει σαφής κατεύθυνση."
        memory_type = "ΑΔΙΑΦΟΡΟΣ"
    elif H < 0.45:
        interpretation = f"H = {H:.4f} < 0.45: Η χρονοσειρά έχει ΚΑΘΟΔΙΚΗ ΠΟΡΕΙΑ. Υπάρχει mean-reverting συμπεριφορά. Οι τιμές τείνουν να επιστρέφουν στον μέσο όρο."
        memory_type = "ΚΑΘΟΔΙΚΗ ΠΟΡΕΙΑ"
    else:  # H > 0.55
        interpretation = f"H = {H:.4f} > 0.55: Η χρονοσειρά έχει ΑΝΟΔΙΚΗ ΠΟΡΕΙΑ. Υπάρχει μακροπρόθεσμη μνήμη. Οι τάσεις τείνουν να συνεχίζονται."
        memory_type = "ΑΝΟΔΙΚΗ ΠΟΡΕΙΑ (ΜΑΚΡΟΠΡΟΘΕΣΜΗ ΜΝΗΜΗ)"

    return render_template('exercise11.html',
                         H=f"{H:.4f}",
                         RS0=f"{RS0:.4f}",
                         RS1=f"{RS1:.4f}",
                         RS2=f"{RS2:.4f}",
                         RS_mean=f"{RS_mean:.4f}",
                         RS=f"{RS:.4f}",
                         interpretation=interpretation,
                         memory_type=memory_type)

@app.route('/exercise/12')
def exercise12():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")

    def autocorr(x, max_lag=50):
        n = len(x)
        mean = np.mean(x)
        autocorrs = []
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorrs.append(1.0)
            else:
                numerator = np.sum((x[lag:] - mean) * (x[:-lag] - mean))
                denominator = np.sum((x - mean) ** 2)
                if denominator > 0:
                    autocorrs.append(numerator / denominator)
                else:
                    autocorrs.append(0.0)
        return np.array(autocorrs)

    # arxiki timeseira (P)
    P = df['P'].values
    # ypologizoume to autocorr
    autocorr_P = autocorr(P, max_lag=50)

    # timeseira apo to step 5 - compute S_E if it doesn't exist
    if 'S_E' not in df.columns:
        t = df['t'].values.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(t, P)
        P_pred = reg.predict(t)
        df['S_E'] = P - P_pred
    residuals = df['S_E'].values
    autocorr_residuals = autocorr(residuals, max_lag=50)

    lags = np.arange(len(autocorr_P))

    # Γράφημα σύγκρισης
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Αρχική χρονοσειρά
    axes[0].plot(lags, autocorr_P, marker='o', linestyle='-', markersize=3, color='blue')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0].set_title("Αυτοσυσχέτιση Αρχικής Χρονοσειράς (P)")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Αυτοσυσχέτιση")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 50)

    # Χρονοσειρά στοχαστικότητας
    axes[1].plot(lags, autocorr_residuals, marker='o', linestyle='-', markersize=3, color='red')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title("Αυτοσυσχέτιση Χρονοσειράς Στοχαστικότητας (Κατάλοιπα)")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Αυτοσυσχέτιση")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 50)

    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)

    return render_template('exercise12.html',
                         ac_original_lag1=f"{autocorr_P[1]:.4f}",
                         ac_original_max=f"{np.max(autocorr_P[1:]):.4f}",
                         ac_residuals_lag1=f"{autocorr_residuals[1]:.4f}",
                         ac_residuals_max=f"{np.max(autocorr_residuals[1:]):.4f}",
                         plot=img)

@app.route('/exercise/13')
def exercise13():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    s_e_values = df['S_E'].values if 'S_E' in df.columns else df['P'].values

    diff_S_E = np.diff(s_e_values)
    X_n = s_e_values[:-1]
    X_n1 = s_e_values[1:]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].plot(df['t'][:-1], X_n, marker='o', linestyle='-', markersize=3, alpha=0.7, color='blue')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title("E_n (Χρονοσειρά Καταλοίπων)")
    axes[0].set_xlabel("Χρόνος (t)")
    axes[0].set_ylabel("X_n")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['t'][1:], X_n1, marker='o', linestyle='-', markersize=3, alpha=0.7, color='green')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title("E_{n+1} (Χρονοσειρά Καταλοίπων - Επόμενη Περίοδος)")
    axes[1].set_xlabel("Χρόνος (t)")
    axes[1].set_ylabel("X_{n+1}")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df['t'][1:], diff_S_E, marker='o', linestyle='-', markersize=3, alpha=0.7, color='red')
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[2].set_title("X_{n+1} - X_n (Πρώτες Διαφορές Καταλοίπων)")
    axes[2].set_xlabel("Χρόνος (t)")
    axes[2].set_ylabel("X_{n+1} - X_n")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)
    return render_template('exercise13.html', plot=img)

@app.route('/exercise/14')
def exercise14():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    s_e_values = df['S_E'].values if 'S_E' in df.columns else df['P'].values

    diff_residuals = np.diff(s_e_values)
    X_n = s_e_values[:-1]
    X_diff = diff_residuals

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_n, X_diff, alpha=0.5, s=10, color='blue')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Χρονοσειρά βήματος 5 (X_n = Κατάλοιπα/S_E)", fontsize=12)
    ax.set_ylabel("Πρώτες Διαφορές (X_{n+1} - X_n)", fontsize=12)
    ax.set_title("Φασικό Πορτραίτο: Κατάλοιπα vs Πρώτες Διαφορές", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)
    return render_template('exercise14.html', plot=img)

@app.route('/exercise/15')
def exercise15():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    df['MA_20'] = df['P'].rolling(window=20, min_periods=1).mean()
    df['dMA_20'] = df['MA_20'].diff()
    df_clean = df[['t', 'dMA_20']].dropna()
    t = df_clean['t'].values.reshape(-1, 1)

    regression = LinearRegression()
    regression.fit(t, df_clean['dMA_20'])

    a = f"{regression.intercept_:.2f}"
    b = f"{regression.coef_[0]:.2f}"
    stationarity = "Η χρονοσειρά δεν είναι στάσιμη." if regression.coef_[0] != 0 else "Η χρονοσειρά είναι στάσιμη."

    return render_template('exercise15.html', a=a, b=b, stationarity=stationarity)

@app.route('/exercise/16')
def exercise16():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    t = df['t'].values
    df['MO_20'] = df['P'].rolling(window=20, min_periods=1).mean()
    df['MO_50'] = df['P'].rolling(window=50, min_periods=1).mean()

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, df['MO_20'], label='MO_20')
    axs[0].set_ylabel('MO_20')
    axs[0].set_title('Kinitos Mesos Oro 20')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(t, df['MO_50'], label='MO_50', color='orange')
    axs[1].set_xlabel('Time (t)')
    axs[1].set_ylabel('MO_50')
    axs[1].set_title('Kinitos Mesos Oro 50')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    img = plot_to_base64(fig)
    plt.close(fig)
    return render_template('exercise16.html', plot=img)

@app.route('/exercise/17')
def exercise17():
    import os
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    hlc_path = "../dataset_processed_hlc.csv"
    if not os.path.exists(hlc_path):
        hlc_path = "dataset_processed_hlc.csv"
    if os.path.exists(hlc_path):
        df2 = pd.read_csv(hlc_path)
    else:
        # Fallback if file doesn't exist
        df2 = df.copy()
        df2['Close'] = df['P'].values
        df2['High'] = df['P'].values * 1.01
        df2['Low'] = df['P'].values * 0.99

    P = df['P'].values
    cl = df2['Close'].values[-500:] if len(df2) >= 500 else df2['Close'].values
    hi = df2['High'].values[-500:] if len(df2) >= 500 else df2['High'].values
    lo = df2['Low'].values[-500:] if len(df2) >= 500 else df2['Low'].values
    n2 = len(cl)

    TR = []
    period = 50
    for i in range(n2):
        if i == 0:
            TR.append(hi[i]-lo[i])
        else:
            x1 = abs(hi[i]-lo[i])
            x2 = abs(hi[i]-cl[i-1])
            x3 = abs(lo[i]-cl[i-1])
            TR.append(max(x1, x2, x3))

    ATR = np.mean(TR)

    mean_all = np.mean(P)
    std_all = np.std(P)
    all_cv = std_all / mean_all

    if all_cv < 0.3:
        stability = "Η χρονοσειρά είναι σταθερή."
    elif all_cv < 0.5:
        stability = "Η χρονοσειρά είναι ασθενώς ασταθής."
    else:
        stability = "Η χρονοσειρά είναι ασταθής."

    return render_template('exercise17.html', atr=f"{ATR:.2f}", stability=stability)

@app.route('/exercise/18')
def exercise18():
    df = pd.read_csv("/home/kopisto/pythonexample/dataset_processed.csv")
    P = df['P'].values
    t = df['t'].values.reshape(-1, 1)
    n = len(P)

    # Basic characteristics
    mean_P = np.mean(P)
    std_P = np.std(P)
    min_P = np.min(P)
    max_P = np.max(P)

    # Trend analysis
    regression = LinearRegression()
    regression.fit(t, P)
    P_pred = regression.predict(t)
    a = regression.intercept_
    b = regression.coef_[0]
    r2 = r2_score(P, P_pred)
    has_trend = b != 0
    trend_direction = "ΑΝΟΔΙΚΗ" if b > 0 else "ΚΑΘΟΔΙΚΗ" if b < 0 else "ΟΥΔΕΤΕΡΗ"
    trend_strength = "ΙΣΧΥΡΗ" if abs(r2) > 0.7 else "ΜΕΤΡΙΑ" if abs(r2) > 0.3 else "ΑΣΘΕΝΗΣ"

    # Stationarity of original series
    result_original = adfuller(P, autolag='AIC')
    is_original_stationary = result_original[1] < 0.05

    # Residuals analysis
    residuals = P - P_pred
    result_residuals = adfuller(residuals, autolag='AIC')
    is_residuals_stationary = result_residuals[1] < 0.05

    # Hurst index
    P_hurst = P[-567:] if len(P) >= 567 else P
    n_hurst = len(P_hurst)
    P_mean_hurst = np.mean(P_hurst)
    Deviations_hurst = P_hurst - P_mean_hurst
    R_hurst = np.max(Deviations_hurst) - np.min(Deviations_hurst)
    S_hurst = np.std(Deviations_hurst)
    RS_hurst = R_hurst / S_hurst if S_hurst > 0 else 0
    segment_size = n_hurst // 3
    Aggregate_Deviations0 = []
    Aggregate_Deviations1 = []
    Aggregate_Deviations2 = []
    for i in range(segment_size):
        if i == 0:
            Aggregate_Deviations0.append(Deviations_hurst[i])
        else:
            Aggregate_Deviations0.append(Deviations_hurst[i] + Aggregate_Deviations0[i-1])
    for i in range(segment_size, 2*segment_size):
        idx = i - segment_size
        if idx == 0:
            Aggregate_Deviations1.append(Deviations_hurst[i])
        else:
            Aggregate_Deviations1.append(Deviations_hurst[i] + Aggregate_Deviations1[idx-1])
    for i in range(2*segment_size, n_hurst):
        idx = i - 2*segment_size
        if idx == 0:
            Aggregate_Deviations2.append(Deviations_hurst[i])
        else:
            Aggregate_Deviations2.append(Deviations_hurst[i] + Aggregate_Deviations2[idx-1])
    R0_hurst = np.max(Aggregate_Deviations0) - np.min(Aggregate_Deviations0)
    S0_hurst = np.std(Aggregate_Deviations0) if len(Aggregate_Deviations0) > 1 else 1
    RS0_hurst = R0_hurst / S0_hurst if S0_hurst > 0 else 0
    R1_hurst = np.max(Aggregate_Deviations1) - np.min(Aggregate_Deviations1)
    S1_hurst = np.std(Aggregate_Deviations1) if len(Aggregate_Deviations1) > 1 else 1
    RS1_hurst = R1_hurst / S1_hurst if S1_hurst > 0 else 0
    R2_hurst = np.max(Aggregate_Deviations2) - np.min(Aggregate_Deviations2)
    S2_hurst = np.std(Aggregate_Deviations2) if len(Aggregate_Deviations2) > 1 else 1
    RS2_hurst = R2_hurst / S2_hurst if S2_hurst > 0 else 0
    RS_mean_hurst = (RS0_hurst + RS1_hurst + RS2_hurst) / 3 if (RS0_hurst > 0 and RS1_hurst > 0 and RS2_hurst > 0) else RS_hurst
    val1_hurst = 1
    val2_hurst = log(n_hurst)
    val3_hurst = log(RS_mean_hurst) if RS_mean_hurst > 0 else log(RS_hurst) if RS_hurst > 0 else 0
    val4_hurst = log(RS_hurst) if RS_hurst > 0 else 0
    if val2_hurst > val1_hurst and val4_hurst > 0:
        H = (val4_hurst - val3_hurst) / (val2_hurst - val1_hurst)
    else:
        H = 0.5
    has_long_memory = H > 0.55
    memory_type = "ΑΝΟΔΙΚΗ ΠΟΡΕΙΑ (ΜΑΚΡΟΠΡΟΘΕΣΜΗ ΜΝΗΜΗ)" if H > 0.55 else "ΚΑΘΟΔΙΚΗ ΠΟΡΕΙΑ" if H < 0.45 else "ΑΔΙΑΦΟΡΟΣ"

    # Autocorrelation
    def autocorr(x, lag=1):
        n_ac = len(x)
        mean_ac = np.mean(x)
        numerator = np.sum((x[lag:] - mean_ac) * (x[:-lag] - mean_ac))
        denominator = np.sum((x - mean_ac) ** 2)
        return numerator / denominator if denominator > 0 else 0
    ac_original = autocorr(P, lag=1)
    ac_residuals = autocorr(residuals, lag=1)
    has_high_autocorr = abs(ac_original) > 0.5

    # Instability
    mid_point = n // 2
    first_half = P[:mid_point]
    second_half = P[mid_point:]
    mean_first = np.mean(first_half)
    mean_second = np.mean(second_half)
    change_pct = abs(mean_second - mean_first) / mean_first * 100 if mean_first > 0 else 0
    has_instability = change_pct > 10

    # Stability from exercise 17
    import os
    hlc_path = "../dataset_processed_hlc.csv"
    if not os.path.exists(hlc_path):
        hlc_path = "dataset_processed_hlc.csv"
    if os.path.exists(hlc_path):
        df2 = pd.read_csv(hlc_path)
    else:
        df2 = df.copy()
        df2['Close'] = df['P'].values
        df2['High'] = df['P'].values * 1.01
        df2['Low'] = df['P'].values * 0.99
    cl = df2['Close'].values[-500:] if len(df2) >= 500 else df2['Close'].values
    hi = df2['High'].values[-500:] if len(df2) >= 500 else df2['High'].values
    lo = df2['Low'].values[-500:] if len(df2) >= 500 else df2['Low'].values
    n2 = len(cl)
    TR = []
    for i in range(n2):
        if i == 0:
            TR.append(hi[i]-lo[i])
        else:
            x1 = abs(hi[i]-lo[i])
            x2 = abs(hi[i]-cl[i-1])
            x3 = abs(lo[i]-cl[i-1])
            TR.append(max(x1, x2, x3))
    ATR = np.mean(TR)
    all_cv = std_P / mean_P if mean_P > 0 else 0
    if all_cv < 0.3:
        stability_status = "ΣΤΑΘΕΡΗ"
    elif all_cv < 0.5:
        stability_status = "ΑΣΘΕΝΩΣ ΑΣΤΑΘΗΣ"
    else:
        stability_status = "ΑΣΤΑΘΗΣ"

    # Create summary plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].plot(df['t'], P, label='Χρονοσειρά (P)', color='blue', alpha=0.6, linewidth=1)
    axes[0].plot(df['t'], P_pred, label=f'Τάση (R² = {r2:.4f})', color='red', linewidth=2)
    axes[0].set_title("Χρονοσειρά και Τάση")
    axes[0].set_xlabel("Χρόνος (t)")
    axes[0].set_ylabel("Τιμή (P)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df['t'], residuals, label='Κατάλοιπα (Στοχαστικότητα)', color='green', alpha=0.7, linewidth=1)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title("Κατάλοιπα (Στάσιμα)" if is_residuals_stationary else "Κατάλοιπα")
    axes[1].set_xlabel("Χρόνος (t)")
    axes[1].set_ylabel("Κατάλοιπα")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    summary_plot = plot_to_base64(fig)
    plt.close(fig)

    return render_template('exercise18.html',
                         n=n,
                         mean_P=f"{mean_P:.2f}",
                         std_P=f"{std_P:.2f}",
                         min_P=f"{min_P:.2f}",
                         max_P=f"{max_P:.2f}",
                         r2=f"{r2:.4f}",
                         trend_direction=trend_direction,
                         trend_strength=trend_strength,
                         is_original_stationary=is_original_stationary,
                         is_residuals_stationary=is_residuals_stationary,
                         H=f"{H:.4f}",
                         has_long_memory=has_long_memory,
                         memory_type=memory_type,
                         ac_original=f"{ac_original:.4f}",
                         ac_residuals=f"{ac_residuals:.4f}",
                         has_high_autocorr=has_high_autocorr,
                         change_pct=f"{change_pct:.2f}",
                         has_instability=has_instability,
                         stability_status=stability_status,
                         ATR=f"{ATR:.2f}",
                         summary_plot=summary_plot)

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
