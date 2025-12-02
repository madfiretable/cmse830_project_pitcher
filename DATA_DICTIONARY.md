# Data Dictionary

## Overview
This document provides detailed definitions and specifications for all variables used in the Pitching Analytics Dashboard.

---

## Dataset Sources

### 1. MLB 2025 Data (`MLB_Pitch.csv`)
- **Source**: Major League Baseball official statistics
- **Year**: 2025 season
- **Records**: ~800 pitchers
- **Update Frequency**: Season-to-date

### 2. CPBL 2024 Data (`投手2024.csv`)
- **Source**: Chinese Professional Baseball League
- **Year**: 2024 complete season
- **Records**: ~100 pitchers
- **Language**: Traditional Chinese column names

### 3. CPBL 2025 Data (`投手.xlsx`)
- **Source**: Chinese Professional Baseball League
- **Year**: 2025 season (in progress)
- **Records**: ~100 pitchers
- **Format**: Excel spreadsheet

---

## Variable Definitions

### Core Identification Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| **Player** | String | Player name | "Sandy Alcantara" |
| **Team** | String | Team name/abbreviation | "MIA", "富邦" |
| **League** | String | League identifier | "MLB", "CPBL" |
| **Year** | Integer | Season year | 2024, 2025 |
| **Team_YY** | String | Combined team and year | "MIA 25", "富邦 24" |

---

### Workload Metrics

| Variable | Type | Description | Range | Interpretation |
|----------|------|-------------|-------|----------------|
| **BF** | Integer | Batters Faced | 0-1000+ | Total hitters faced; indicator of workload |
| **IP** | Float | Innings Pitched | 0-250+ | Total innings pitched |
| **G** | Integer | Games Pitched | 0-80+ | Number of game appearances |
| **GS** | Integer | Games Started | 0-35+ | Number of starts (starting pitchers) |

---

### Run Prevention Metrics

| Variable | Type | Description | Range | Interpretation |
|----------|------|-------------|-------|----------------|
| **ERA** | Float | Earned Run Average | 0-10+ | Earned runs per 9 innings; **Lower is better** |
| **ERA+** | Integer | Adjusted ERA | 0-200+ | League/park adjusted ERA (100=avg); **Higher is better** |
| **FIP** | Float | Fielding Independent Pitching | 2-6+ | Defense-independent ERA estimate; **Lower is better** |
| **RA9** | Float | Runs Allowed per 9 innings | 0-12+ | Total runs (earned + unearned) per 9 IP |

---

### Control & Command Metrics

| Variable | Type | Description | Range | Interpretation |
|----------|------|-------------|-------|----------------|
| **WHIP** | Float | Walks + Hits per IP | 0.8-2.0 | Baserunners allowed per inning; **Lower is better** |
| **BB%** | Float | Walk Percentage | 0-20% | Walks ÷ BF × 100; **Lower is better** |
| **K%** | Float | Strikeout Percentage | 0-40% | Strikeouts ÷ BF × 100; **Higher is better** |
| **SO/BB** | Float | Strikeout-to-Walk Ratio | 0-10+ | K ÷ BB; **Higher is better** (3.0+ is excellent) |
| **BB/9** | Float | Walks per 9 innings | 0-6+ | Free passes per 9 IP; **Lower is better** |
| **K/9** | Float | Strikeouts per 9 innings | 0-15+ | Strikeouts per 9 IP; **Higher is better** |

---

### Contact Quality Metrics

| Variable | Type | Description | Range | Interpretation |
|----------|------|-------------|-------|----------------|
| **BABIP** | Float | Batting Avg on Balls in Play | .200-.400 | Excludes HR & K; .300 is typical |
| **GB%** | Float | Ground Ball Percentage | 30-60% | % of batted balls on ground; **Higher = fewer HR** |
| **FB%** | Float | Fly Ball Percentage | 20-50% | % of batted balls in air; High = HR risk |
| **LD%** | Float | Line Drive Percentage | 15-25% | % of hard-hit line drives |
| **HR/9** | Float | Home Runs per 9 IP | 0-2.5 | Long balls allowed per 9 IP; **Lower is better** |
| **HR/FB** | Float | HR per Fly Ball | 5-20% | % of fly balls leaving park |

---

### Advanced Metrics

| Variable | Type | Description | Range | Interpretation |
|----------|------|-------------|-------|----------------|
| **WAR** | Float | Wins Above Replacement | -2 to 10+ | Total value vs. replacement player; **Higher is better** |
| **xFIP** | Float | Expected FIP | 2-6 | FIP with normalized HR rate |
| **SIERA** | Float | Skill-Interactive ERA | 2-6 | Advanced ERA estimator |
| **WPA** | Float | Win Probability Added | -5 to +5 | Contribution to team wins |

---

### Pitch-Level Metrics (MLB Only)

| Variable | Type | Description | Range | Interpretation |
|----------|------|-------------|-------|----------------|
| **Whiff%** | Float | Swinging Strike Rate | 10-35% | % of swings resulting in miss; **Higher is better** |
| **Swing%** | Float | Overall Swing Rate | 35-55% | % of pitches swung at |
| **SwStr%** | Float | Swinging Strike % | 8-18% | Swinging strikes ÷ total pitches |
| **O-Swing%** | Float | Outside Swing % | 20-40% | % of swings on pitches outside zone |
| **Z-Swing%** | Float | Zone Swing % | 55-75% | % of swings on pitches in zone |
| **Contact%** | Float | Contact Rate | 65-85% | % of swings making contact |
| **PutAway%** | Float | Putaway Rate | 15-30% | % of 2-strike ABs ending in K |

---

### Count & Outcome Variables

| Variable | Type | Description | Range | Interpretation |
|----------|------|-------------|-------|----------------|
| **W** | Integer | Wins | 0-25 | Games won as pitcher of record |
| **L** | Integer | Losses | 0-20 | Games lost as pitcher of record |
| **SV** | Integer | Saves | 0-50+ | Games saved (closers) |
| **BS** | Integer | Blown Saves | 0-15 | Save opportunities failed |
| **HLD** | Integer | Holds | 0-40 | Hold situations converted |
| **CG** | Integer | Complete Games | 0-10 | Games pitched entirely |
| **SHO** | Integer | Shutouts | 0-5 | Complete games with 0 runs |
| **QS** | Integer | Quality Starts | 0-25 | Starts with 6+ IP and ≤3 ER |

---

## Derived Variables (Calculated in Pipeline)

### ERA+ Calculation
```
ERA+ = (League ERA / Player ERA) × 100
```
- Accounts for league average and ballpark factors
- 100 = league average
- 120 = 20% better than league average
- 80 = 20% worse than league average

### SO/BB Calculation
```
SO/BB = Strikeouts / Walks
```
- Measures efficiency and dominance
- 2.0 = good
- 3.0 = very good
- 4.0+ = elite

### Team_YY Calculation
```
Team_YY = Team + " " + (Year % 100)
```
- Examples: "MIA 25", "富邦 24"
- Used for year-aware grouping in visualizations

---

## Data Types & Constraints

### Integer Variables
- **Range**: 0 to practical maximum
- **Missing Values**: Represented as NaN
- **Imputation**: KNN → Iterative Imputer

### Float Variables
- **Precision**: 2-3 decimal places typical
- **Missing Values**: Handled via imputation pipeline
- **Outliers**: Retained for model robustness

### String Variables
- **Encoding**: UTF-8 (supports Chinese characters)
- **Case**: Preserved as in source
- **Standardization**: Minimal (team names may vary)

---

## Missing Value Strategy

### Imputation Pipeline (Two-Stage)

**Stage 1: KNN Imputer**
- **Method**: K-Nearest Neighbors
- **k value**: 5 neighbors
- **Use case**: Initial fill for sparse data

**Stage 2: Iterative Imputer**
- **Method**: MICE (Multiple Imputation by Chained Equations)
- **Max iterations**: 10
- **Use case**: Refined imputation considering relationships

### Variables with High Missing Rates
- Advanced pitch metrics (MLB only)
- Some CPBL-specific stats
- New 2025 season data (in progress)

---

## Data Quality Notes

### Known Issues
1. **CPBL Team Names**: May vary in romanization
2. **2025 Data**: Incomplete season (updated regularly)
3. **League Differences**: CPBL has fewer games/pitchers
4. **Metric Availability**: Not all metrics available for both leagues

### Data Validation Rules
- BF ≥ 70 (configurable threshold)
- ERA ≥ 0
- IP ≥ 0
- Percentage metrics: 0-100%
- Ratios: Non-negative

---

## Usage Examples

### Filtering High-Quality Pitchers
```python
qualified_pitchers = df[df['BF'] >= 200]
```

### League Comparison
```python
mlb_avg_era = df[df['League'] == 'MLB']['ERA'].mean()
cpbl_avg_era = df[df['League'] == 'CPBL']['ERA'].mean()
```

### Year-over-Year Analysis
```python
player_2024 = df[(df['Player'] == 'Name') & (df['Year'] == 2024)]
player_2025 = df[(df['Player'] == 'Name') & (df['Year'] == 2025)]
```

---

## Column Name Mappings (CPBL → English)

| CPBL Column (Chinese) | English Equivalent | Variable Name |
|-----------------------|-------------------|---------------|
| 球員 | Player | Player |
| 球隊 | Team | Team |
| 防禦率 | ERA | ERA |
| 投球局數 | Innings Pitched | IP |
| 三振 | Strikeouts | K |
| 保送 | Walks | BB |
| 被安打 | Hits Allowed | H |
| 責失分 | Earned Runs | ER |
| 勝場 | Wins | W |
| 敗場 | Losses | L |
| 救援成功 | Saves | SV |

---

## References

- **MLB Stats**: [MLB.com](https://www.mlb.com/stats)
- **FanGraphs Glossary**: [fangraphs.com/glossary](https://www.fangraphs.com/library/pitching/)
- **CPBL Official**: [cpbl.com.tw](https://www.cpbl.com.tw)

---

**Last Updated**: December 2024  
**Version**: 1.0
