# Pedestrian Injury Risk Predictor

# Table of Contents

1. [Overview](#overview)
2. [Stakeholder Context](#stakeholder-context)

   * [Business Question](#business-question)
   * [Project Purpose](#project-purpose)
3. [Dataset Description](#dataset-description)
4. [Reproduction Steps](#reproduction-steps)
5. [Exploratory Data Analysis](#exploratory-data-analysis-decision-oriented-summary)

   * [Time-of-Day Patterns](#time-of-day-patterns)
   * [Borough × Time Risk Patterns](#borough--time-risk-patterns)
   * [Vehicle Type Patterns](#vehicle-type-patterns)
   * [Contributing Factors](#contributing-factors)
   * [Weekday vs Weekend Patterns](#weekday-vs-weekend-patterns)
6. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)

   * [Major Cleaning Steps](#major-cleaning-steps)
   * [Borough Imputation (GeoPandas)](#borough-imputation-with-geopandas)
7. [Modeling Approach](#modeling-approach)

   * [Model A – Baseline](#model-a--global-mean-baseline)
   * [Model B – Simple Logistic Regression](#model-b--simple-logistic-regression)
   * [Model C – Final Calibrated Logistic Regression](#model-c--final-calibrated-logistic-regression)
   * [Calibration Explanation](#what-is-a-calibrated-model)
8. [Assumptions, Limitations, and Ethical Considerations](#assumptions-limitations-and-ethical-considerations)
9. [Connection to the Business Question](#connection-to-business-question)
10. [Future Work](#future-work)
11. [Streamlit App Overview](#streamlit-app-overview)

* [What the App Does](#what-the-app-does)
* [How to Use It](#how-to-use-it)

## PM board and Slide Deck
- [Project Management Board](https://www.notion.so/2c4bea2d956480f3b248ee6d00b69f3a?v=2c4bea2d9564808784c4000cc71a9540&source=copy_link)
- [Slide Deck](https://docs.google.com/presentation/d/1q9BKlU-QY4dN1jVR5D0fCWjn0GbfX5mxRmK1R1gakGs/edit?usp=sharing)
## Overview

This project supports **NYC Vision Zero**, the city’s multi-agency initiative dedicated to eliminating traffic deaths and serious injuries. Using NYC’s Motor Vehicle Collision dataset, the project builds a predictive model that identifies conditions under which pedestrian injuries are most likely to occur. The tool is designed to inform safety interventions, enforcement prioritization, and data-driven street redesign strategies.


## Stakeholder Context

**Primary Stakeholder:** NYC Vision Zero Initiative
Vision Zero uses crash data to understand where injuries happen and to guide decisions about infrastructure improvements, targeted enforcement, and pedestrian-safety programs. Their mission is to make streets safer for people who walk, bike, or use micromobility.

## Business Question:
**What factors increase the likelihood that a pedestrian will be injured in a crash, and how can Vision Zero target these risks?**

## Project Purpose:
Transform raw collision records into actionable insights that can help Vision Zero predict when and where pedestrian injuries are more likely, enabling smarter and more timely safety interventions.



## Dataset Description

**Source:** [NYPD Motor Vehicle Collisions – Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)

### Key Fields

* **hour:** Hour of day when the crash occurred
* **BoroName:** NYC borough where crash occurred
* **veh_group:** Simplified vehicle categories (sedan, SUV, truck, motorcycle, other)
* **cf1_clean:** Cleaned primary contributing factor (e.g., distraction, unsafe speed, failure to yield)
* **ped_injury:** Target variable (1 = pedestrian injured, 0 = not injured)


## Reproduction Steps

1. Clone the repository
2. Download the dataset from NYC Open Data
3. Place the CSV in the `/data` folder
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
5. Run the cleaning and modeling notebooks
6. Launch the application:

   ```bash
   streamlit run app.py
   ```



# Exploratory Data Analysis (Decision-Oriented Summary)

## Time-of-Day Patterns

**Pedestrian injury risk varies substantially by time of day.**

![Injuries by Hour](images/ped_inj_hour.png)
### Hourly Trends

* **1–5 AM:** Low injuries (200–400 per hour)
* **7–9 AM:** Sharp rise; ~1,800 injuries at 8 AM
* **10 AM–3 PM:** Steady mid-day plateau (1,200–1,500)
* **4–6 PM:** Highest-risk period; 5–6 PM exceeds 2,500 injuries
* **8 PM–12 AM:** Declines but remains above early-morning levels

**Key takeaway:**
The *evening commute* (4–6 PM) is the most dangerous period citywide. This is when pedestrian volumes and traffic volumes are simultaneously high.

### Binned Time-of-Day Patterns

![Hourly Bins](images/bins.png)

* **Highest injury counts:** Midday (10–15) and Evening (16–19)
* **Moderate risk:** Morning (6–9) and Night (20–23)
* **Lowest risk:** Overnight (0–5)

**Implication for Vision Zero:**
Focus enforcement and safety operations during peak evening hours.



## Borough × Time Risk Patterns

![Borough x Time Heatmap](images/heatmap.png)

The heatmap shows strong spatial and temporal differences across boroughs.

### Key Insights

* **Brooklyn has the highest pedestrian injury counts across nearly all hours.**
  Likely due to dense commercial corridors, wide avenues, and very high foot traffic.

* **Evening commute peaks (16–19) occur in every borough**, especially Brooklyn and Queens.

### Suggested Interventions

* Traffic calming on high-risk corridors
* Automated enforcement during PM peaks
* Visibility improvements (daylighting, signal timing upgrades)
* Turn-calming and pedestrian-first intersection designs



## Vehicle Type Patterns

![Vehicle Heatmap](images/heatmap_v_b.png)

### 1. Sedans and SUVs account for the majority of pedestrian injuries

Examples:

* Brooklyn: 4,135 sedan injuries; 3,860 SUV injuries
* Queens: 3,344 sedan injuries; 2,987 SUV injuries

**Interpretation:**
Everyday passenger vehicles pose the largest injury burden.

### 2. Borough-Level Differences

* **Brooklyn:** Highest volumes of sedan and SUV injuries
* **Queens:** Elevated SUV and truck injuries on wide arterials
* **Manhattan:** High sedan involvement due to dense traffic
* **Bronx & Staten Island:** Lower counts but similar patterns

**Main takeaway:**
Interventions should be borough-specific.



## Contributing Factors

![Behavior](images/topcontributing.png)

Pedestrian injuries are overwhelmingly driven by **preventable driver behaviors**.

### Top Contributors

1. **Driver Inattention/Distraction**
   Over 8,000 injuries; #1 cause citywide.

2. **Failure to Yield Right-of-Way**
   Nearly as significant as distraction.
   Common in intersection and turning conflicts.

3. **Aggressive/unsafe maneuvers:**

   * Following too closely
   * Improper lane use
   * Backing unsafely
   * Unsafe speed

### Vision Zero Implications

* Automated enforcement
* Turn-calming infrastructure
* Leading Pedestrian Intervals (LPIs)
* Public education on distracted driving
* Visibility and crosswalk redesigns



## Weekday vs. Weekend Patterns

![Weekday Patterns](images/weekda.png)

* **Wednesday–Friday**: Highest injury counts (> 5,400 each)
* **Friday**: Highest-risk day overall
* **Sunday**: Lowest (~3,400)

**Drivers of weekday risk:**

* Commuting
* School dismissals
* Commercial activity
* Higher vehicle and pedestrian volumes



# Data Cleaning and Preprocessing

### Major Cleaning Steps

* Removed rows with missing pedestrian injury values
* Consolidated contributing factors into interpretable categories
* Simplified vehicle types (sedan, SUV, truck, etc.)
* Converted crash dates to datetime
* Created new features: hour, weekday
* Removed irrelevant columns and noise
* Filtered to records with at least one vehicle and a defined borough

### Borough Imputation with GeoPandas

1. Remove rows with missing coordinates
2. Convert to GeoDataFrame using Shapely points (WGS84)
3. Load official NYC borough boundary shapefiles
4. Spatial join to assign borough from latitude/longitude

This improved completeness and preserved valuable records.



# Modeling Approach

## Overview

Multiple models were developed to evaluate which crash characteristics predict pedestrian injuries, with recall prioritized due to the safety-critical nature of the problem.

### Models Tested



### **Model A – Global Mean Baseline**

* Predicts injury probability equal to the dataset’s average (~9%)
* Establishes benchmark MAE/MSE
* No feature usage; only for comparison



### **Model B – Simple Logistic Regression**

Features:

* `hour`
* `veh_group` (one-hot encoded)

Improved performance over baseline but limited recall.



### **Model C – Final Calibrated Logistic Regression**

**Features Used:**

* `cf1_clean`
* `hour`
* `veh_group`
* `BoroName`

**Pipeline:**

* StandardScaler for hour
* One-hot encoding for categorical fields
* LogisticRegression(max_iter=2000, class_weight='balanced')
* GridSearchCV tuned `C ∈ {0.01, 0.1, 3, 5}` with scoring='recall'
* Best `C = 3`
* CalibratedClassifierCV (sigmoid, cv=3)
* Decision threshold = **0.07** (safety-first)

**Test Set Performance (threshold = 0.07):**

* **Precision:** ~0.16
* **Recall:** ~0.84
* **F1:** ~0.27
* **AUC:** ~0.79



## What Is a Calibrated Model?

Calibration adjusts raw model scores so predicted probabilities more accurately reflect true risk. The logistic regression model was poorly calibrated due to class imbalance. After calibration:

* Probabilities became more realistic
* Threshold selection became meaningful
* Recall improved substantially

This makes the model suitable for Vision Zero safety screening.



# Assumptions, Limitations, and Ethical Considerations

## Why Recall Was Prioritized

Missing a true pedestrian-injury case is far more harmful than flagging a false positive.
A recall of **0.84** reflects this safety-first approach.

## Model Purpose and Assumptions

* Designed as a **risk screening tool**, not a precision classifier
* Uses four key features that carry reliable predictive signal
* Assumes historical patterns reflect future risks

## Key Limitations

* Limited feature set (no speed limits, weather, road geometry, etc.)
* Highly imbalanced data
* Increased false positives due to low threshold
* One-hot encoding reduces intuitive interpretability
* Historical relationships may shift over time

## Ethical Considerations

* Avoid reinforcing policing disparities through over-prediction in certain areas
* Ensure transparency: results guide planning, not enforcement penalties
* Focus on systemic safety—not attributing blame to individuals



# Connection to Business Question

The analysis directly answers:

**“What factors increase the likelihood that a crash results in a pedestrian injury?”**

Results identify:

* High-risk times (4–6 PM)
* High-risk boroughs (Brooklyn, Queens)
* High-risk vehicle types (sedans, SUVs)
* High-risk behaviors (distraction, failing to yield)

This supports:

* Infrastructure planning
* Enforcement strategy
* Targeted safety campaigns
* Risk-informed resource allocation



# Future Work

* Add roadway, weather, speed, lighting, and traffic volume features
* Explore tree-based and boosting models
* Conduct fairness and equity audits
* Retrain with updated annual crash data
* Implement time-varying thresholds or dynamic calibration
* Add explainability tools (e.g., SHAP) for stakeholder transparency



# Streamlit App Overview

## What the App Does

The Streamlit app enables users to input:

* Primary contributing factor
* Hour of day
* Vehicle type
* Borough

The app returns:

* A calibrated injury probability
* A risk classification
* A threshold explanation

The interface is intentionally simple for non-technical users.

## How to Use It

1. Select crash conditions in the left sidebar
2. Click **Predict**
3. View:
   * Injury probability
   * High vs. low risk indicator
4. Adjust inputs to explore different scenarios



---

## Project Structure

```
.
├── Data/
│   ├── nybb_25d/
│   ├── .gitignore
│   └── Motor_Vehicle_Collisions_-_Crashes_20251209.csv
│
├── app/
│   ├── .venv/
│   ├── app.py
│   └── calibrated_model.pkl
│
├── images/
│   ├── bins.png
│   ├── heatmap.png
│   ├── heatmap_v_b.png
│   ├── ped_inj_hour.png
│   ├── topcontributing.png
│   └── weekda.png
│
├── models/
│   ├── modelA.ipynb
│   ├── modelB.ipynb
│   └── modelC.ipynb
│
├── notebooks/
│   ├── clean.ipynb
│   └── eda.ipynb
│
├── README.md
└── requirements.txt
```


