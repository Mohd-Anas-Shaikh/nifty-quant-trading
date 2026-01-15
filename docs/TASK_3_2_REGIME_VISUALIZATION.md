# Task 3.2: Regime Visualization

## Overview

This document describes the visualizations created for regime analysis. These plots help understand market regime patterns, transitions, and characteristics.

---

## Visualizations Created

### 1. Price Chart with Regime Overlay

**File**: `regime_price_overlay.png`

**Description**: NIFTY 50 price chart with color-coded background showing the detected regime at each point in time.

**Color Coding**:
- 🟢 **Green**: Uptrend (+1)
- 🟠 **Orange**: Sideways (0)
- 🔴 **Red**: Downtrend (-1)

**Interpretation**:
- Visual confirmation of regime detection accuracy
- Shows how regimes align with actual price movements
- Identifies regime persistence and transition points

---

### 2. Transition Matrix Heatmap

**File**: `transition_matrix_heatmap.png`

**Description**: Heatmap showing the probability of transitioning from one regime to another.

**How to Read**:
- Rows = "From" regime
- Columns = "To" regime
- Values = Probability (0 to 1)
- Darker colors = Higher probability

**Key Insights**:
```
                    To:
From:        Downtrend   Sideways   Uptrend
Downtrend      96.1%       3.9%       0.0%
Sideways        0.0%      98.7%       1.3%
Uptrend         2.8%       0.0%      97.2%
```

- **High diagonal values**: Regimes are persistent (stay in same state)
- **Off-diagonal values**: Show transition patterns
- **Downtrend → Sideways**: 3.9% (recovery through consolidation)
- **Uptrend → Downtrend**: 2.8% (direct reversal possible)

---

### 3. Regime Statistics (Box Plots)

**File**: `regime_statistics.png`

**Description**: Box plots showing the distribution of key features in each regime.

**Features Analyzed**:
1. **Average IV**: Volatility expectation
2. **IV Spread**: Call-Put IV difference
3. **PCR (OI)**: Put-Call Ratio
4. **ATM Call Delta**: Price sensitivity
5. **ATM Call Gamma**: Delta sensitivity
6. **ATM Call Vega**: Volatility sensitivity

**What to Look For**:
- **Median line**: Center of distribution
- **Box**: Middle 50% of data (IQR)
- **Whiskers**: Range of typical values
- **Dots**: Outliers

**Expected Patterns**:
| Feature | Uptrend | Sideways | Downtrend |
|---------|---------|----------|-----------|
| Avg IV | Lower | Medium | Higher |
| PCR | Lower | Medium | Higher |
| Delta | Higher | ~0.5 | Lower |

---

### 4. Duration Histogram

**File**: `regime_duration_histogram.png`

**Description**: Histograms showing how long each regime typically lasts.

**Metrics Shown**:
- **Mean**: Average duration
- **Median**: Middle value
- **Max**: Longest observed duration

**Interpretation**:
- **Right-skewed**: Most regimes are short, few are very long
- **High mean vs median**: Presence of long-lasting regimes
- **Compare across regimes**: Which regime persists longest?

---

### 5. Regime Pie Chart (Bonus)

**File**: `regime_pie_chart.png`

**Description**: Pie chart showing overall regime distribution.

**Results**:
- **Uptrend**: 25.32% (396 observations)
- **Sideways**: 55.31% (865 observations)
- **Downtrend**: 19.37% (303 observations)

**Insight**: Market spends most time in sideways regime, with roughly equal time in uptrend and downtrend.

---

## Output Files

| Visualization | File | Location |
|--------------|------|----------|
| Price Overlay | `regime_price_overlay.png` | `data/processed/visualizations/` |
| Transition Matrix | `transition_matrix_heatmap.png` | `data/processed/visualizations/` |
| Regime Statistics | `regime_statistics.png` | `data/processed/visualizations/` |
| Duration Histogram | `regime_duration_histogram.png` | `data/processed/visualizations/` |
| Pie Chart | `regime_pie_chart.png` | `data/processed/visualizations/` |

---

## How to Use

### Viewing Plots

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load and display a plot
img = Image.open('data/processed/visualizations/regime_price_overlay.png')
plt.figure(figsize=(16, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

### Regenerating Visualizations

```bash
python run_regime_viz.py
```

### Custom Visualizations

```python
from src.regime.regime_visualization import RegimeVisualizer
import pandas as pd

df = pd.read_csv('data/processed/nifty_features_5min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

visualizer = RegimeVisualizer()

# Create individual plots
visualizer.plot_price_with_regimes(df)
visualizer.plot_transition_matrix()
visualizer.plot_regime_statistics(df)
visualizer.plot_duration_histogram(df)
```

---

## Interpretation Guide

### Price Overlay Analysis

1. **Regime Alignment**: Do colored regions match price direction?
2. **Transition Points**: Where do colors change?
3. **False Signals**: Any misclassified regions?

### Transition Matrix Analysis

1. **Persistence**: High diagonal = stable regimes
2. **Common Transitions**: Which regime changes are most likely?
3. **Rare Transitions**: Which changes almost never happen?

### Statistics Analysis

1. **Separation**: Are distributions different across regimes?
2. **Overlap**: How much do distributions overlap?
3. **Outliers**: Any unusual values in specific regimes?

### Duration Analysis

1. **Typical Duration**: What's the most common regime length?
2. **Long Regimes**: How often do extended regimes occur?
3. **Trading Implications**: How long to hold positions?

---

## Trading Applications

### Regime-Based Position Sizing

```python
# Larger positions in confident regimes
if regime_prob > 0.8:
    position_size = base_size * 1.5
else:
    position_size = base_size * 0.5
```

### Regime-Based Strategy Selection

```python
# Different strategies for different regimes
strategies = {
    1: 'trend_following_long',
    0: 'mean_reversion',
    -1: 'trend_following_short'
}
current_strategy = strategies[current_regime]
```

### Duration-Based Exit

```python
# Exit if regime has lasted longer than typical
if regime_duration > median_duration * 2:
    consider_exit = True
```

---

## Summary

The visualizations provide:

1. **Visual Validation**: Confirm regime detection makes sense
2. **Transition Understanding**: Know how regimes change
3. **Feature Insights**: Understand what drives each regime
4. **Duration Patterns**: Plan holding periods

These insights are crucial for:
- Strategy development
- Risk management
- Position sizing
- Entry/exit timing
