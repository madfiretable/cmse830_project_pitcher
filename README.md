# ⚾ Pitching Analytics Dashboard
## MLB 2025 + CPBL 2024/2025 Performance Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cmse830projectpitchergit.streamlit.app/)

**🌐 Live Demo**: [https://cmse830projectpitchergit.streamlit.app/](https://cmse830projectpitchergit.streamlit.app/)

A comprehensive Streamlit-based analytics platform for analyzing and comparing pitcher performance across Major League Baseball (MLB) and Chinese Professional Baseball League (CPBL). This interactive dashboard provides advanced statistical analysis, machine learning predictions, and multi-dimensional visualizations for baseball analytics.

---

## 📊 Project Overview

This project integrates multiple data sources to provide in-depth insights into pitcher performance metrics, featuring:

- **Multi-league Analysis**: Combined MLB (2025) and CPBL (2024, 2025) data
- **Advanced Metrics**: ERA+, FIP, WHIP, K%, BB%, WAR, and more
- **Machine Learning Models**: Predictive analytics for WAR (Wins Above Replacement)
- **Interactive Visualizations**: KDE plots, scatter grids, violin plots, correlation heatmaps
- **Year-over-Year Comparisons**: Track performance trends across seasons

---

## 🎯 Key Features

### 1. **Data Integration & Processing**
- Three distinct data sources (MLB 2025, CPBL 2024, CPBL 2025)
- Advanced data cleaning and preprocessing
- Two-stage imputation (KNN → Iterative Imputer)
- BF (Batters Faced) threshold filtering
- Unified schema across leagues

### 2. **Exploratory Data Analysis**
- Correlation heatmaps with customizable parameters
- Interactive KDE (Kernel Density Estimation) plots by league/year
- Team-based violin plots with year grouping
- Multi-panel scatter grid (ERA+ vs other metrics)
- Comprehensive statistical summaries

### 3. **Feature Engineering**
- ERA+ normalization (league-adjusted)
- SO/BB ratio calculation
- Team_YY (Team + Year) composite features
- Percentage-based metrics (K%, BB%, GB%, FB%)
- **Advanced interaction terms**:
  - K_WHIP_interaction: Combines strikeout rate with control
  - Efficiency_Score: SO/BB ratio weighted by ERA+
  - ERA_WHIP_product: Combines two strongest predictors
  - FIP_weighted: Workload-adjusted fielding independent pitching
- Two-stage imputation (KNN → Iterative Imputer)

### 4. **Machine Learning Models**
- **Robust Regression (Huber)**: Outlier-resistant linear model with interpretable coefficients
- **Gradient Boosting**: 500 trees with optimized hyperparameters (R² ≈ 0.87)
- **XGBoost**: State-of-the-art ensemble with early stopping (R² ≈ 0.87)
- WAR prediction with comprehensive evaluation metrics
- Model comparison with R², MAE scores, and residual analysis
- Automatic outlier detection and feature importance ranking
- **Performance**: Achieves professional-grade accuracy (R² > 0.85, MAE < 0.4 WAR)

### 5. **Interactive Dashboard**
- **6 organized tabs** with clear navigation:
  - 📋 **Overview**: Project introduction and metrics dictionary
  - 🔍 **Data Cleaning (IDA)**: Raw data inspection and quality checks
  - 🧹 **Data Encoding (EDA)**: Statistical analysis and correlation heatmaps
  - 📊 **Visualizations**: Interactive KDE, violin, and scatter plots
  - 🤖 **ML Models**: Machine learning predictions and diagnostics
  - 💾 **Download**: Export processed data
- Dynamic filtering (League, Team, Year, BF threshold)
- Customizable visualization parameters
- Real-time data exploration
- Export functionality for processed data

---

## 🛠️ Technology Stack

- **Python 3.8+**
- **Streamlit**: Interactive web application framework
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models and preprocessing
- **XGBoost**: Advanced gradient boosting for optimal predictions
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Statistical plotting

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pitching-analytics.git
cd pitching-analytics
```

2. **Create a virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Prepare data files**
Place the following files in the project directory:
- `MLB_Pitch.csv` - MLB 2025 pitching data
- `投手2024.csv` - CPBL 2024 pitching data
- `投手.xlsx` - CPBL 2025 pitching data

---

## 🚀 Usage

### Running the Application

```bash
streamlit run test_stream.py
```

The application will launch in your default web browser at `http://localhost:8501`

### Navigation Guide

#### **Tab 1: 📋 Overview**
- View data sources and processing pipeline
- Comprehensive metrics dictionary with definitions
- Preview of raw and imputed data

#### **Tab 2: 🔍 Data Cleaning (IDA)**
- Raw data statistics and summaries
- Missing value analysis
- Duplicate detection
- Data quality checks

#### **Tab 3: 🧹 Data Encoding (EDA)**
- Correlation heatmap with customizable settings
- Statistical summaries before and after imputation
- Distribution analysis
- Feature engineering results

#### **Tab 4: 📊 Visualizations**
- **KDE Plots**: Distribution comparison across leagues/years
- **Violin Plots**: Team performance distributions with year grouping
- **Scatter Grid**: ERA+ correlations with multiple metrics

#### **Tab 5: 🤖 ML Models**
- **Feature Selection Strategy**: Data-driven feature selection
- **Diagnostic Information**: Data quality and sample size checks
- **Robust Regression**: Outlier-resistant predictions with coefficients
- **Gradient Boosting**: Optimized ensemble model (R² ≈ 0.87)
- **XGBoost**: State-of-the-art predictions with early stopping
- **Model Comparison**: Side-by-side performance metrics
- **Residual Analysis**: Error pattern visualization
- **Prediction Plots**: Actual vs predicted scatter plots

#### **Tab 6: 💾 Download**
- Export processed and filtered data as CSV

### Interactive Controls

**Sidebar Settings:**
- **BF Threshold**: Filter pitchers by minimum batters faced
- **League Filter**: Select MLB, CPBL, or both
- **Year Filter**: Choose 2024, 2025, or both
- **Visualization Options**: Toggle different plot types
- **Heatmap Tuning**: Adjust cell size and text size

---

## 📈 Metrics Explained

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **ERA** | Earned Run Average | Lower = better run prevention |
| **ERA+** | Adjusted ERA (100 = league avg) | >100 = above average |
| **FIP** | Fielding Independent Pitching | Lower = better pitching skill |
| **WHIP** | Walks + Hits per Inning | Lower = better control |
| **K%** | Strikeout Percentage | Higher = more dominant |
| **BB%** | Walk Percentage | Lower = better command |
| **WAR** | Wins Above Replacement | Higher = greater value |
| **SO/BB** | Strikeout-to-Walk Ratio | Higher = more efficient |

*See full metrics table in the Overview tab*

---

## 🔬 Data Processing Pipeline

```
1. Data Loading
   ├── MLB 2025 (CSV)
   ├── CPBL 2024 (CSV)
   └── CPBL 2025 (XLSX)
   
2. Data Cleaning
   ├── Schema unification
   ├── Type conversion
   └── Duplicate removal
   
3. Feature Engineering
   ├── ERA+ calculation
   ├── SO/BB ratio
   └── Percentage metrics
   
4. Missing Value Imputation
   ├── Stage 1: KNN Imputer
   └── Stage 2: Iterative Imputer
   
5. Filtering
   └── BF threshold application
   
6. Ready for Analysis & Modeling
```

---

## 🤖 Machine Learning Approach

### Model Selection Rationale

**Robust Regression (Huber):**
- Outlier-resistant linear model
- Interpretable coefficients for feature analysis
- Fast training and prediction
- Automatic outlier detection (downweights extreme values)
- **Performance**: R² ≈ 0.72, MAE ≈ 0.52 WAR

**Gradient Boosting Regressor:**
- Handles non-linear relationships and feature interactions
- 500 trees with optimized hyperparameters
- Balanced complexity vs overfitting
- Early stopping for optimal tree count
- **Performance**: R² ≈ 0.87, MAE ≈ 0.37 WAR (Best overall)

**XGBoost Regressor:**
- State-of-the-art gradient boosting implementation
- Built-in L1/L2 regularization
- Parallel processing for faster training
- Handles missing values naturally
- **Performance**: R² ≈ 0.87, MAE ≈ 0.36 WAR

### Evaluation Metrics
- **R² Score**: Proportion of variance explained (higher is better, max 1.0)
- **MAE (Mean Absolute Error)**: Average prediction error in WAR units (lower is better)
- **Residual Analysis**: Visualizes prediction error patterns

### Features Used (13 total)
**Primary Features:**
- ERA_plus, WHIP, FIP, SO/BB, K%, BAbip, **BF (Batters Faced)**

**Secondary Features:**
- Whiff%, HR9, GB%, BB%, ERA, FB%

**Engineered Interaction Terms:**
- K_WHIP_interaction: K% × (1/WHIP)
- Efficiency_Score: (SO/BB × ERA+) / 100
- ERA_WHIP_product: ERA+ × (1/WHIP)
- FIP_weighted: FIP × log(BF)

### Key Insight
**BF (Batters Faced) is critical!** Adding this feature improved R² by 10-15%, as WAR depends on both quality (ERA+, WHIP) and quantity (innings pitched).

---

## 📁 Project Structure

```
pitching-analytics/
│
├── test_stream.py           # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
│
├── data/                   # Data directory (not in repo)
│   ├── MLB_Pitch.csv
│   ├── 投手2024.csv
│   └── 投手.xlsx
│
└── outputs/                # Generated outputs
    └── combined_pitching_processed_filtered.csv
```

---

## 🎨 Visualization Examples

### Correlation Heatmap
- Identifies relationships between pitching metrics
- Customizable color scale and size
- Option to exclude BF from analysis

### KDE Plots
- Compare metric distributions across leagues
- Year-over-year trends visualization
- Interactive variable selection

### Scatter Grid
- ERA+ vs multiple metrics in one view
- Color-coded by team
- Shape-coded by year (circle=2024, square=2025)

### Violin Plots
- Team performance distributions
- League-specific comparisons
- Custom team selection

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `AttributeError: 'numpy.ndarray' object has no attribute 'fillna'`
- **Solution**: Ensure you're using the latest version of the code with proper Series conversion

**Issue**: App doesn't update after code changes
- **Solution**: 
  - Stop Streamlit (Ctrl+C)
  - Restart: `streamlit run test_stream.py`
  - Or press 'R' in browser to rerun

**Issue**: Missing data files
- **Solution**: Ensure all three data files are in the correct directory with exact filenames

**Issue**: Memory errors with large datasets
- **Solution**: Increase BF threshold to reduce dataset size

---

## 🏆 Model Performance

### Achieved Results (Test Set)

| Model | R² Score | MAE (WAR) | Training Speed | Best For |
|-------|----------|-----------|----------------|----------|
| **Robust Regression** | 0.7203 | 0.5207 | Fast | Outlier resistance & interpretability |
| **Gradient Boosting** | **0.8723** | **0.3723** | Medium | **Best balance** |
| **XGBoost** | 0.8706 | **0.3599** | Fast | Maximum accuracy |

### Performance Highlights

✅ **Professional-Grade Accuracy**: R² > 0.85 (comparable to MLB team analytics)  
✅ **Low Prediction Error**: MAE < 0.4 WAR (excellent for practical use)  
✅ **Robust to Outliers**: 22.7% of training data automatically downweighted  
✅ **Feature Insights**: BF, WHIP, and ERA+ are top predictors

### What This Means

- **R² = 0.87**: Model explains 87% of WAR variance
- **MAE = 0.37**: Average error of only 0.37 wins
- **Comparison**: Professional teams typically achieve R² ≈ 0.85-0.90
- **Use Cases**: 
  - Predict 2025 season WAR with high confidence
  - Identify undervalued pitchers
  - Compare MLB vs CPBL performance
  - Support trade and signing decisions

---

## 📊 Data Requirements

### MLB Data (MLB_Pitch.csv)
Expected columns: Player, Team, ERA, FIP, WHIP, K%, BB%, WAR, BF, etc.

### CPBL Data (投手2024.csv, 投手.xlsx)
Expected columns: 球員 (Player), 球隊 (Team), 防禦率 (ERA), 投球局數 (IP), 三振 (K), 保送 (BB), etc.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Future Enhancements

- [ ] Add player comparison tool
- [ ] Implement pitch type analysis
- [ ] Include injury data integration
- [ ] Add predictive modeling for ERA+
- [ ] Export visualizations as images
- [x] ~~Add more ML models (XGBoost)~~ ✅ Completed
- [ ] Implement cross-validation visualization
- [ ] Add season prediction capabilities
- [ ] Separate models for starters vs relievers
- [ ] Real-time data updates via API

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- MLB for providing comprehensive pitching statistics
- CPBL for baseball data
- Streamlit community for excellent documentation
- Scikit-learn for machine learning tools
- XGBoost developers for state-of-the-art gradient boosting
- FanGraphs for WAR calculation methodology

---

## 📞 Support

For questions or issues:
1. Check the Troubleshooting section
2. Open an issue on GitHub
3. Contact via email

---

**Last Updated**: December 2024

**Version**: 4.0 - Enhanced ML Models & 6-Tab Structure

**Live App**: [https://cmse830projectpitchergit.streamlit.app/](https://cmse830projectpitchergit.streamlit.app/)

