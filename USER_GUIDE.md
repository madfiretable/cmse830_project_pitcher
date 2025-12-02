# User Guide - Pitching Analytics Dashboard

## Quick Start Guide

### 🚀 Getting Started in 5 Minutes

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Data Files**
   - `MLB_Pitch.csv`
   - `投手2024.csv`
   - `投手.xlsx`

3. **Run Application**
   ```bash
   streamlit run test_stream.py
   ```

4. **Open Browser**
   - Navigate to `http://localhost:8501`

---

## 📊 Tab-by-Tab Guide

### Tab 1: Overview
**Purpose**: Understand the data and metrics

**What You'll See**:
- Data sources and processing pipeline
- Complete metrics dictionary
- Sample of raw and imputed data

**Use This Tab When**:
- First time using the app
- Need to understand what metrics mean
- Want to see data quality

**Tips**:
- Bookmark the metrics table for reference
- Compare "Before" vs "After" imputation tables

---

### Tab 2: IDA (Initial Data Analysis)
**Purpose**: Explore raw data characteristics

**What You'll See**:
- Descriptive statistics by league
- Missing value patterns
- Data distribution summaries

**Use This Tab When**:
- Checking data quality
- Understanding league differences
- Identifying data issues

**Key Questions to Answer**:
- How much data is missing?
- Are MLB and CPBL comparable?
- What's the typical range for each metric?

**Tips**:
- Compare mean values between leagues
- Look for suspicious outliers
- Note missing value percentages

---

### Tab 3: EDA (Exploratory Data Analysis)
**Purpose**: Deep dive into relationships and patterns

**What You'll See**:
- **Correlation Heatmap**: Shows relationships between metrics
- Statistical summaries post-imputation
- League-specific statistics

**Use This Tab When**:
- Looking for metric relationships
- Understanding what correlates with WAR
- Comparing leagues statistically

**Interactive Controls**:
- Toggle heatmap on/off
- Exclude BF from analysis
- Adjust cell size (0.6-1.8)
- Adjust text size (6-18)

**Key Insights**:
- Strong correlations: ERA+ ↔ WAR, K% ↔ Whiff%
- Negative correlations: ERA ↔ ERA+, BB% ↔ SO/BB
- Look for unexpected patterns

**Tips**:
- Red = positive correlation
- Blue = negative correlation
- Larger numbers = stronger relationship
- Values close to 1 or -1 are very strong

---

### Tab 4: Visualizations
**Purpose**: Interactive visual exploration

#### 🎵 KDE Plots (Kernel Density Estimation)

**What It Shows**:
- Distribution shapes for selected metrics
- Comparison across league/year combinations

**How to Use**:
1. Select metrics from dropdown (default: ERA, FIP, K%)
2. View overlapping distributions
3. Compare "MLB 25" vs "CPBL 24" etc.

**Interpretation**:
- Peak = most common value
- Width = variability
- Multiple peaks = subgroups
- Shift = league differences

**Example Use Cases**:
- "Is CPBL ERA lower than MLB?" → Check ERA plot
- "Do 2025 pitchers strike out more?" → Compare K% curves
- "League difficulty difference?" → Compare ERA+ distributions

---

#### 🎻 Violin Plots

**What It Shows**:
- Team-by-team metric distributions
- Year-specific patterns (Team_YY format)

**How to Use**:
1. Select metric (e.g., WAR, ERA+)
2. Choose specific teams OR view all by league
3. Examine shape and spread

**Interpretation**:
- Width = frequency (wider = more players at that value)
- White dot = median
- Thick black bar = interquartile range
- Thin line = min/max range

**Example Questions**:
- "Which team has the best pitchers?" → Compare WAR medians
- "Most consistent team?" → Look for narrow violins
- "Year-over-year improvement?" → Compare "MIA 24" vs "MIA 25"

**Tips**:
- Use "Custom Teams" to focus on specific clubs
- Compare shapes, not just medians
- Look for outliers (extreme values)

---

#### 📈 Scatter Grid (ERA+ vs Other Metrics)

**What It Shows**:
- ERA+ correlations with multiple metrics simultaneously
- 4-column grid layout
- Color = Team, Shape = Year

**How to Use**:
1. Select variables for x-axis (default: top metrics)
2. Each subplot shows ERA+ (y-axis) vs selected metric
3. Hover for player details

**Symbols**:
- ⚪ Circle = 2024 season
- ⬜ Square = 2025 season

**Color Coding**:
- Each team has unique color
- Easy to spot team clusters

**Example Insights**:
- "Strong K% improves ERA+?" → Check K% subplot
- "Is high WHIP bad?" → See WHIP correlation
- "Team effect on ERA+?" → Spot color clusters

**Tips**:
- Look for tight linear relationships
- Outliers = unusual players
- Compare 2024 vs 2025 patterns

---

#### 🤖 WAR Prediction Models

**What It Shows**:
- Machine learning model results
- Feature importance rankings
- Model performance metrics

**Models**:
1. **Linear Regression**: Simple, interpretable
2. **Random Forest**: Complex, more accurate

**Key Metrics**:
- **R² Score**: % of variance explained (higher = better)
  - 0.80 = good
  - 0.90 = excellent
- **MAE**: Average error in WAR units (lower = better)
  - 0.5 = half a win error (acceptable)

**How to Interpret**:

**Linear Regression Coefficients**:
- Positive = increases WAR
- Negative = decreases WAR
- Larger magnitude = stronger effect

**Random Forest Importances**:
- Values sum to 1.0
- Higher = more important for predictions
- Typical top features: ERA+, FIP, K%

**Use Cases**:
- "What matters most for WAR?" → Check importances
- "Is model reliable?" → Check R² and MAE
- "Which model is better?" → Compare metrics

---

### Tab 5: Download
**Purpose**: Export processed data

**What You Get**:
- CSV file with all filters applied
- Imputed values included
- Ready for external analysis

**Use This When**:
- Need data for Excel/Python/R
- Want to share filtered dataset
- Building custom models

**Filename**: `combined_pitching_processed_filtered.csv`

---

## 🎛️ Sidebar Controls

### Global Settings

#### BF Threshold
- **Range**: 0-1000+ batters faced
- **Default**: 70
- **Purpose**: Filter out pitchers with minimal work
- **Recommendations**:
  - 70 = Include most pitchers
  - 100 = Quality starters only
  - 200 = Workhorse starters

#### League Filter
- **Options**: MLB, CPBL, or both
- **Use Case**: 
  - Compare leagues → Select both
  - Focus on one → Select single league

#### Year Filter
- **Options**: 2024, 2025, or both
- **Use Case**:
  - Year-over-year → Select both
  - Single season → Select one

#### Team Filter
- **Dynamic**: Updates based on league selection
- **Multi-select**: Choose multiple teams
- **Use Case**: Focus on specific franchises

---

### Visualization Toggles

#### Show Correlation Heatmap
- ✅ On: Display heatmap in EDA tab
- ⬜ Off: Hide for faster loading

#### Exclude BF from Heatmap
- ✅ On: Remove BF (reduces noise)
- ⬜ Off: Include BF

#### Show Heatmap Variable List
- Displays exact variables used
- Useful for documentation

#### Show KDE Plots
- Toggle kernel density plots
- Saves time if not needed

#### Show Violin Plots
- Toggle team violin plots
- Resource-intensive with many teams

#### Show Scatter Grid
- Toggle ERA+ scatter analysis
- Large plot, may slow performance

---

### Heatmap Tuning

#### Cell Size Scale (0.6 - 1.8)
- **Smaller**: Compact heatmap
- **Larger**: Easier to read
- **Default**: 1.2

#### Text Size (6 - 18)
- **Smaller**: More data visible
- **Larger**: Better readability
- **Default**: 11

---

## 🎯 Common Use Cases

### Use Case 1: "Find the Best Pitcher in 2025"
1. **Tab 3 → EDA**: Sort by WAR (descending)
2. **Tab 4 → Scatter**: Look for top-right outliers
3. Check: High ERA+, High K%, Low WHIP

### Use Case 2: "Compare My Team to League Average"
1. **Sidebar**: Select your team only
2. **Tab 4 → Violin**: Select WAR or ERA+
3. Compare violin shape to league distribution

### Use Case 3: "Identify Breakout Candidates"
1. **Tab 4 → Scatter**: Look at K% vs ERA+
2. Find: High K% but lower ERA+ (2024)
3. Check if improved in 2025

### Use Case 4: "Evaluate Trade Target"
1. **Tab 1**: Understand metrics
2. **Tab 3**: Check correlations with WAR
3. **Tab 4**: Visual comparisons
4. **Tab 4 → Models**: See predicted WAR

### Use Case 5: "League Comparison Study"
1. **Sidebar**: Select both MLB & CPBL
2. **Tab 2**: Compare descriptive statistics
3. **Tab 4 → KDE**: Overlay distributions
4. Look for systematic differences

### Use Case 6: "Year-over-Year Team Improvement"
1. **Sidebar**: Select team, both years
2. **Tab 4 → Violin**: Choose ERA+ or WAR
3. Compare "Team 24" vs "Team 25"
4. Look for median shift upward

---

## 💡 Pro Tips

### Performance Optimization
- Increase BF threshold if app is slow
- Disable unused visualizations
- Select fewer teams/leagues
- Use "Custom Teams" for focused analysis

### Best Practices
- Always check Tab 1 metrics first
- Use multiple visualizations for insights
- Compare relative performance (ERA+) over absolute (ERA)
- Consider sample size (BF) when evaluating

### Interpretation Guidelines
- **Single metric never tells full story**
- ERA+ > 120 = ace-level
- K% > 25% = dominant
- BB% < 7% = excellent control
- WAR > 3.0 = All-Star caliber

### Common Pitfalls to Avoid
- ❌ Comparing ERA across leagues (use ERA+ instead)
- ❌ Judging relievers by starter metrics
- ❌ Ignoring BF/IP (sample size matters)
- ❌ Overfitting to single season

---

## 🐛 Troubleshooting

### Problem: "No data to plot"
**Solution**: 
- Lower BF threshold
- Select more leagues/years
- Check team filter not too restrictive

### Problem: "App is slow"
**Solution**:
- Increase BF threshold to 150+
- Disable scatter grid
- Reduce heatmap size
- Select fewer teams

### Problem: "Metrics don't make sense"
**Solution**:
- Check Tab 1 for definitions
- Verify league is selected
- Ensure year filter is correct
- Read DATA_DICTIONARY.md

### Problem: "Can't see player name"
**Solution**:
- Hover over data points (scatter/violin)
- Check Tab 2/3 raw tables
- Use Download tab → open in Excel

---

## 📚 Additional Resources

### Documentation Files
- `README.md` - Full project documentation
- `DATA_DICTIONARY.md` - Metric definitions
- `MODELING_APPROACH.md` - ML methodology
- `requirements.txt` - Dependencies

### External Resources
- **FanGraphs**: https://www.fangraphs.com/library/
- **Baseball Reference**: https://www.baseball-reference.com/
- **CPBL Official**: https://www.cpbl.com.tw/

---

## 🆘 Getting Help

### In-App Help
- Metric definitions → Tab 1
- Hover tooltips on interactive elements
- Sidebar expanders for context

### External Support
1. Check troubleshooting section above
2. Review documentation files
3. Open GitHub issue

---

**Last Updated**: December 2025  
**App Version**: 3.0
