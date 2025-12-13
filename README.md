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

## Overview and Key takeaway
- This project supports **NYC Vision Zero**, the city’s multi-agency initiative dedicated to eliminating traffic deaths and serious injuries. Using NYC’s Motor Vehicle Collision dataset, the project builds a predictive model that identifies conditions under which pedestrian injuries are most likely to occur. The tool is designed to inform safety interventions, enforcement prioritization, and data-driven street redesign strategies.
- Using NYC crash data, this project identifies that pedestrian injury risk is highest during the evening commute (4–6 PM), in Brooklyn and Queens, and in crashes involving sedans or SUVs. A calibrated logistic regression model captures these patterns and correctly flags approximately 84% of pedestrian-injury crashes, enabling Vision Zero to proactively target high-risk conditions rather than react after harm occurs.

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

---------

## Modeling Approach
- **Key Takeaway:** Using a calibrated logistic regression model optimized for recall, we are able to correctly flag approximately 84% of pedestrian-injury crashes while maintaining an AUC of ~0.79, making the model well-suited for safety screening and early risk identification rather than punishment or individual prediction.

### Why a Predictive Model Was Needed
- Exploratory analysis revealed clear patterns in pedestrian injury risk by time of day, borough, vehicle type, and driver behavior, but Vision Zero decisions require more than descriptive trends. A predictive model allows us to combine these factors and estimate risk under specific conditions, such as:
  - “If a crash occurs in Brooklyn during the evening commute involving a sedan and driver distraction, how likely is a pedestrian injury?”
- Because pedestrian injuries are relatively rare (about 9% of crashes), the modeling challenge is not accuracy alone, but identifying high-risk cases without missing true injuries.

### Models Evaluated
- We tested three models of increasing complexity to balance interpretability, performance, and safety impact.

#### Model A – Global Mean Baseline
- What it does: Predicts the same injury probability for every crash equal to the dataset average (~9%).
- Why it matters: This establishes a minimum performance benchmark. Any useful model must outperform this baseline.
- Limitation: No ability to distinguish high-risk from low-risk situations.

#### Model B – Simple Logistic Regression
- Features used:
  - `Hour of day`
  - `Vehicle type`
- What improved: This model captured basic temporal and vehicle-related risk patterns and outperformed the baseline.
- Why it wasn’t enough: Recall remained limited, meaning too many pedestrian injury cases were missed. For a safety-critical application, this is unacceptable.


#### Model C – Final Calibrated Logistic Regression (Selected Model)
- This model was chosen because it provides the best balance between recall, interpretability, and responsible deployment.
- Features Used:
  - `cf1_clean` (primary contributing factor)
  - `hour`
  - `veh_group`
  - `BoroName`
- These features were selected because they:
  - Are consistently available and reliable
  - Show strong signal in exploratory analysis
  - Align with real-world intervention levers (time, location, behavior, vehicle type)
 
##### Model C's Pipeline
  - StandardScaler applied to numeric features to normalize scale
  - One-Hot Encoding for categorical features
  - Logistic Regression with class weighting to address imbalance
  - GridSearchCV used to tune the regularization parameter C, optimizing for recall
  - Best-performing value: C = 3
  - CalibratedClassifierCV (sigmoid) applied to correct probability distortion
  - Final decision threshold set to 0.07 to prioritize safety

##### Why Recall Was Prioritized
- In this context, false negatives are more harmful than false positives.
- A false negative means failing to flag a situation where a pedestrian injury is likely, potentially missing an opportunity for safety intervention.
- A false positive may lead to extra attention or resources, which can be reviewed and adjusted by human decision-makers.
  - Because of this, recall was chosen as the primary optimization metric.

##### Model Performance (Test Set)
- At a safety-optimized threshold of 0.07:
  - Recall: ~0.84
    - The model correctly identifies 84% of pedestrian-injury crashes
  - Precision: ~0.16
    - Many flagged cases will not result in injury, which is expected and acceptable for screening
  - F1 Score: ~0.27
    - Reflects the intentional tradeoff between recall and precision
  - ROC-AUC: ~0.79
    - Indicates strong overall ability to distinguish between injury and non-injury cases across thresholds
  - Interpretation:
    - The model is effective at ranking risk and identifying dangerous conditions, even when using a conservative threshold.

##### Why Calibration Was Necessary
- Before calibration, the model’s predicted probabilities were systematically misaligned with actual injury rates due to class imbalance. For example, a predicted probability of 0.30 did not correspond to a true 30% injury rate.
- After calibration:
  - Predicted probabilities became more realistic
  - Threshold selection became meaningful
  - Risk scores could be interpreted consistently across scenarios
  - This step was essential for deploying the model as a decision-support tool rather than a black-box classifier.



## Assumptions, Limitations, and Ethical Considerations
- Key Assumptions:
  - Historical crash patterns reflect near-term future risk
  - Selected features capture meaningful signal
  - The model is used to support, not replace, human judgment
- Limitations:
  - No roadway design, speed, weather, or lighting data
  - Class imbalance increases false positives
  - Relationships may shift over time (model drift)
  - Predictions indicate correlation, not causation
- Ethical Considerations
  - Crash data reflects reporting and enforcement bias
  - Over-prediction could reinforce inequitable enforcement
  - False negatives could miss emerging risk areas
- **The model is intended to guide safety design and planning, not punishment**
- Core principle: The goal is to anticipate danger, not assign **blame.**

## Connection to Business Question
- The analysis directly answers: **“What factors increase the likelihood that a crash results in a pedestrian injury?”**
- The model identifies clear risk drivers:
  - High-risk times (evening commute)
  - High-risk boroughs (Brooklyn, Queens)
  - High-risk vehicle types (sedans, SUVs)
  - High-risk behaviors (distraction, failure to yield)
  - These insights support Vision Zero’s ability to prioritize interventions where they can have the greatest safety impact.
- Future Work
  - Add roadway geometry, speed limits, weather, lighting, and traffic volume
  - Test tree-based and ensemble models
  - Conduct formal fairness audits
  - Retrain regularly with updated data
  - Explore dynamic thresholds by time or location

#### Streamlit App Overvie
- What the App Does
  - The app allows users to input crash conditions and returns:
    - A calibrated injury probability
    - A safety-focused risk classification
    - A clear explanation of the decision threshold
  - How to Use It
    - Select crash details
    - Click Predict
    - Interpret the risk score as a screening signal


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


