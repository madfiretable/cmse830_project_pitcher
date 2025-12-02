# ⚾ Pitching Analytics Dashboard
## MLB 2025 + CPBL 2024/2025 Performance Analysis

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
- Advanced missing value handling

### 4. **Machine Learning Models**
- **Linear Regression**: Feature coefficient analysis
- **Random Forest Regressor**: Feature importance ranking
- WAR prediction with comprehensive evaluation metrics
- Model comparison with R² and MAE scores

### 5. **Interactive Dashboard**
- 5 organized tabs (Overview, IDA, EDA, Visualizations, Download)
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

#### **Tab 1: Overview**
- View data sources and processing pipeline
- Terms explanation for all metrics
- Preview of raw and imputed data

#### **Tab 2: IDA (Initial Data Analysis)**
- Raw data statistics and summaries
- Missing value analysis
- Data quality checks

#### **Tab 3: EDA (Exploratory Data Analysis)**
- Correlation heatmap with customizable settings
- Statistical summaries by league
- Distribution analysis

#### **Tab 4: Visualizations**
- **KDE Plots**: Distribution comparison across leagues/years
- **Violin Plots**: Team performance distributions
- **Scatter Grid**: ERA+ correlations with other metrics
- **WAR Prediction Models**: ML model results and insights

#### **Tab 5: Download**
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

**Linear Regression:**
- Interpretable coefficients
- Fast training and prediction
- Good baseline for feature importance

**Random Forest Regressor:**
- Handles non-linear relationships
- Robust to outliers
- Feature importance ranking
- Better predictive performance

### Evaluation Metrics
- **R² Score**: Proportion of variance explained
- **MAE (Mean Absolute Error)**: Average prediction error

### Features Used
- ERA, ERA+, FIP, WHIP
- K%, BB%, SO/BB
- HR9, BAbip, GB%, FB%

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
- [ ] Add more ML models (XGBoost, Neural Networks)
- [ ] Implement cross-validation visualization
- [ ] Add season prediction capabilities

---

## 🙏 Acknowledgments

- MLB for providing comprehensive pitching statistics
- CPBL for baseball data
- Streamlit community for excellent documentation
- Scikit-learn for machine learning tools

---

