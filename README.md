# cmse830_project_pitcher

An interactive Streamlit app for **exploratory data analysis (EDA)** and **preprocessing** of pitching stats from **MLB** and **CPBL (Chinese Professional Baseball League)**.

> 📌 This project is part of a data science course requirement and demonstrates IDA/EDA and preprocessing steps through a Streamlit dashboard.

---

## Data Sources

**MLB (2025):**
- [Standard Pitching](https://www.baseball-reference.com/leagues/majors/2025-standard-pitching.shtml)  
- [Advanced Pitching](https://www.baseball-reference.com/leagues/majors/2025-advanced-pitching.shtml)

**CPBL (2025 JO):**
- [Brothers](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/Kae1X-%E4%B8%AD%E4%BF%A1%E5%85%84%E5%BC%9F?tab=pitching)  
- [Hawks](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/t6zJf-%E5%8F%B0%E9%8B%BC%E9%9B%84%E9%B7%B9?tab=pitching)  
- [Dragons](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/R2VRh-%E5%91%B3%E5%85%A8%E9%BE%8D?tab=pitching)  
- [Guardians](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/wi4T3-%E5%AF%8C%E9%82%A6%E6%82%8D%E5%B0%87?tab=pitching)  
- [Monkeys](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/WyADE-%E6%A8%82%E5%A4%A9%E6%A1%83%E7%8C%BF?tab=pitching)  
- [Lions](https://www.rebas.tw/tournament/CPBL-2025-JO/firstbase/Xs1sP-%E7%B5%B1%E4%B8%807-ELEVEn%E7%8D%85?tab=pitching)

---

## What the App Demonstrates

- **EDA**
  - League/team selection; quick stat summaries (ERA, WHIP, K/BB, etc.)
  - Distributions & comparisons (histograms, box plots)
  - Correlation heatmaps for numeric columns
- **Preprocessing**
  - Encoding: league & team → numeric (`OneHotEncoder`)
  - Scaling: standardize numeric columns (`StandardScaler`)
  - Imputation: handle missing values (`SimpleImputer`)
  - Downloadable transformed dataset

---

## Progress Summary

### **Why I Chose My Dataset**
I chose to work with **pitching statistics from both Major League Baseball (MLB)** and **the Chinese Professional Baseball League (CPBL)** because they represent two different competitive environments and levels of play.  
Comparing these leagues allows me to explore how pitcher performance metrics vary across different systems, player pools, and conditions. This makes the project meaningful for sports analytics and cross-league data analysis.

---

### **What I’ve Learned from IDA/EDA**
Through exploratory data analysis, I found that:
- **ERA, WHIP, and K/BB ratio** are key performance indicators that show clear variation between MLB and CPBL.
- **CPBL pitchers** tend to have more variance in earned run averages, while **MLB pitchers** cluster more tightly.
- Visual tools like **histograms, box plots, and correlation heatmaps** helped me spot potential multicollinearity (e.g., between innings pitched and strikeouts).
- Integrating data from different sources taught me practical skills in **data cleaning, renaming inconsistent columns, and handling missing values.**

---

### **What Preprocessing Steps I’ve Completed**
The preprocessing workflow includes:
- **Imputation:** Replacing missing numeric values using `SimpleImputer(strategy="median")`
- **Encoding:** Converting categorical variables (`league`, `team`) with `OneHotEncoder`
- **Scaling:** Standardizing numeric columns such as ERA, WHIP, SO, BB, and IP using `StandardScaler`
- All steps are combined using a **`ColumnTransformer` pipeline**, and the app allows users to **download the transformed dataset**.

---

### **What I’ve Tried with Streamlit So Far**
The Streamlit dashboard currently supports:
- **Automatic data loading** (from CSV or live scraping via `pandas.read_html`)
- **Interactive controls** for selecting datasets and metrics
- **Dynamic visualizations** with Plotly (histogram, box plot, heatmap)
- **Preprocessing demo** with live preview of encoded and scaled features
- **Download button** for exporting the cleaned and transformed dataset

---

## Acknowledgements
- Baseball-Reference and Rebas.tw for public pitching statistics  
- Streamlit, Plotly, and scikit-learn for data visualization and preprocessing tools

---

## Streamlit link

https://cmse830projectpitchergit-sz7pq7q3u9wdu8m3hfqhgq.streamlit.app/
