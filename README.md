
# Investigating the Role of Social Determinants in Shaping Perceptions of Health Status Across California Counties: A Machine Learning Approach

**Author**: Kat Limqueco, University of California, Los Angeles (UCLA)

> **Keywords**: Multidimensional Population Health, Machine Learning in Public Health, Health Determinants, Predictive Modeling 

## Abstract

This research investigates the role of social determinants in shaping perceptions of health status across California counties, focusing on a myriad of factors including socioeconomic conditions, demographic indicators, environmental conditions, and health status variables. Utilizing the County Health Rankings (CHR) California dataset from 2020 to 2022, the study employs advanced machine learning techniques, both supervised and unsupervised, to explore and categorize health outcomes. The findings reveal fifteen significant predictors of health perceptions, with physical distress emerging as a central determinant. K-Means clustering uncovers three distinct health profiles among the counties, demonstrating enduring patterns in health outcomes and behaviors. The study underscores the multidimensional nature of health perceptions and highlights the potential of machine learning in public health research, offering insights for targeted health policies and interventions.

## Introduction

Health is a multifaceted construct, influenced by individual characteristics, environmental conditions, and a range of social determinants. The World Health Organization emphasizes the impact of social determinants such as socioeconomic status, education, employment, and social support networks on health outcomes and disparities. A comprehensive understanding of these determinants is crucial for effective public health planning and intervention. The advent of machine learning offers a promising avenue for exploring health outcomes, providing advanced computational methods for tasks like disease prediction, health care service optimization, and health behavior analysis. However, the potential of these techniques in categorizing regions based on health status similarities remains largely unexplored, necessitating a detailed investigation into the specific factors predicting self-perceived poor health status in California counties.

## Objectives and Scope of Study

1. **Multidimensional Analysis:**
   - To conduct a comprehensive analysis considering multiple dimensions of population health, such as health-related quality of life and length of life, to provide a more holistic view[^1^].
   
2. **Advanced Machine Learning Techniques:**
   - To leverage advanced, data-driven multivariate statistical learning approaches, including linear and non-linear ensemble tree-based models, to capture the complex, nonlinear relationships in population health[^1^].

3. **Comprehensive Insight Generation:**
   - To interpret the results from the machine learning models to understand the significant predictors and provide insights for more targeted and effective public health strategies and interventions[^1^].

## Data

The primary dataset utilized is the CHR California dataset, spanning the years 2020 to 2022[^3^]. This dataset is a comprehensive collection of health-related indicators, including health behaviors, clinical care, social and economic factors, and physical environment, providing a holistic view of the health landscape in California counties[^3^].

## Methodology

1. **Exploratory Data Analysis (EDA):**
   - Conducting descriptive statistics, visualizations, and correlation analysis to understand the characteristics and relationships within the dataset[^1^].
   
2. **Ethical Feature Selection:**
   - Addressing confounding features and avoiding the blind incorporation of sensitive attributes to prevent the embedding of biases in the algorithms used for clinical care[^2^].
   
3. **Advanced Machine Learning Modeling:**
   - Implementing a suite of statistical learning models including linear regression and non-linear ensemble tree-based models to evaluate population health[^1^].
   
4. **Result Interpretation:**
   - Presenting visualization tools including a variable importance heat-map and partial dependence plots of the key predictors to explain the underlying relationships of the important variables with the population health outcomes[^1^].

## Results

### Model Performance

#### Linear Regression
- **Cross-Validation Score:** 0.944
- **Mean Squared Errors:** Training: 0.564, Testing: 1.324
- **R^2 Scores:** Training: 0.969, Testing: 0.930
- **Most Important Feature:** 'pct_freq_phys_distress' (2.798)

#### Support Vector Regression
- **Cross-Validation Score:** 0.940
- **Mean Squared Errors:** Training: 1.515, Testing: 4.131
- **R^2 Scores:** Training: 0.916, Testing: 0.780
- **Most Important Feature:** 'pct_freq_phys_distress' (2.806)

#### Decision Tree
- **Cross-Validation Score:** 0.806
- **Mean Squared Errors:** Training: 0.620, Testing: 3.645
- **R^2 Scores:** Training: 0.965, Testing: 0.806
- **Most Important Feature:** 'pct_freq_phys_distress' (0.701)

#### Random Forest
- **Cross-Validation Score:** 0.890
- **Mean Squared Errors:** Training: 0.255, Testing: 2.284
- **R^2 Scores:** Training: 0.986, Testing: 0.879
- **Most Important Feature:** 'pct_freq_phys_distress' (0.669)

#### XGBoost
- **Cross-Validation Score:** 0.935
- **Mean Squared Errors:** Training: 0.001, Testing: 1.674
- **R^2 Scores:** Training: 1.000, Testing: 0.911
- **Most Important Feature:** 'pct_freq_phys_distress' (0.652)

### Insights and Implications
All models performed well, with Linear Regression, Support Vector Regression, and XGBoost yielding particularly high cross-validation scores. The feature `pct_freq_phys_distress` emerged as the most important feature across all models, highlighting the significance of addressing physical distress in public health interventions. The variability in feature importance across models underscores the value of employing multiple machine learning models to gain a comprehensive understanding of the data.

### Cluster Analysis
#### 2020-2022 Clusters
Clusters identified over three years consistently represented counties with varying health outcomes, demographic characteristics, and health behaviors. The enduring patterns in health outcomes and behaviors across California have significant implications for designing and implementing health policies and interventions, reflecting the dynamic nature of public health.

## Recommendations for Future Research

1. **Addressing Ethical Considerations:**
   - Future studies should consider ethical implications in feature selection to avoid unintended and permanent embedding of biases in algorithms[^2^].
   
2. **In-depth Analysis of Socioeconomic Factors:**
   - Delving deeper into the impact of individual socioeconomic factors on health outcomes to understand the underlying mechanisms and to identify potential intervention points[^1^].

3. **Incorporation of Additional Datasets:**
   - Integrating the CHR dataset with other relevant datasets, such as healthcare utilization and access, can enrich the analysis and provide a more comprehensive view of the health landscape in California[^1^].

## Repository Structure

```bash 
├── data # data used in the project
│   ├── processed # processed data for modeling
│   │   ├── ca-counties.geojson # geojson file for California counties
│   │   ├── county-health-data-processed.csv # processed dataset for modeling
│   │   ├── county-health-data-summary.csv # summary statistics for dataset
│   │   ├── county-health-final.csv # final dataset for clustering
│   │   └── county-health-rank.csv # county rankings for health outcomes
│   └── raw # raw data for project  
│       └── county-health-data.csv # raw dataset from County Health Rankings
├── docs # documentation files
├── notebooks # notebooks for data analysis and modeling
│   └── health-perceptions.ipynb 
└── src # source code for project
    ├── __init__.py
    ├── app.py # dash app for visualizations
    ├── supervised # scripts for supervised learning models
    │   ├── __init__.py
    │   └── train_supervised.py
    └── unsupervised # scripts for unsupervised learning models
        ├── __init__.py
        └── train_unsupervised.py
```


## References

[^1^]: Wei, Z., Narin, A. B., & Mukherjee, S. (2022). Multidimensional population health modeling: A data-driven multivariate statistical learning approach. *IEEE Access, 10*, 22737-22755.
[^2^]: Chen, I. Y., Pierson, E., Rose, S., Joshi, S., Ferryman, K., & Ghassemi, M. (2021). Ethical machine learning in healthcare. *Annual review of biomedical data science, 4*, 123-144.

[^3^]: County Health Rankings & Roadmaps. (n.d.). *Methods*. Retrieved from [County Health Rankings Website](https://www.countyhealthrankings.org/explore-health-rankings/methods)

