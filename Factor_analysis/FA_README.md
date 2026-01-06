# Factor Analysis - Complete Notebook Collection

A comprehensive set of Jupyter notebooks demonstrating Factor Analysis across five carefully curated datasets, each highlighting different applications and aspects of FA.

## 📚 Notebook Overview

### 1. **Airline Passenger Satisfaction** - `01_airline_satisfaction_fa.ipynb`
**Focus:** Exploratory Factor Analysis (EFA) for Service Quality Dimensions

- **Dataset:** 100,000+ passenger survey responses, 14 Likert-scale variables
- **Expected Factors:** Digital Convenience, On-board Comfort, Service Quality
- **Why it's excellent for FA:**
  - Large sample size ensures stable factor structure
  - Matrix of correlated Likert-scale items (designed for FA)
  - Clear conceptual groupings of service attributes
  - High practical relevance for business decisions

**What you'll learn:**
- Data suitability testing (Bartlett's test, KMO)
- Factor extraction methods (Kaiser, scree plot, parallel analysis)
- Rotation techniques (Varimax, Promax, Oblimin)
- Factor interpretation and business insights
- Creating actionable service improvement strategies

---

### 2. **Big Five Personality Test (IPIP-FFM)** - `02_big_five_personality_fa.ipynb`
**Focus:** Confirmatory Factor Analysis (CFA) of Personality Structure

- **Dataset:** 50 personality questions, 10,000+ responses
- **Expected Factors:** Extroversion, Neuroticism, Agreeableness, Conscientiousness, Openness
- **Why it's excellent for FA:**
  - Textbook example of CFA (testing theoretical structure)
  - Questions explicitly designed to load on 5 factors
  - Demonstrates convergent and discriminant validity
  - Perfect for learning about cross-loadings

**What you'll learn:**
- Reverse-scoring of items
- Confirmatory vs exploratory approaches
- Detecting and interpreting cross-loadings
- Scale reliability and internal consistency
- Comparing empirical structure to theoretical model

---

### 3. **ACX Survey Results** - `03_acx_survey_fa.ipynb`
**Focus:** Uncovering Latent Worldview Dimensions

- **Dataset:** Annual blog reader survey with demographic and attitudinal variables
- **Expected Factors:** Political orientation, Social beliefs, Psychological traits
- **Why it's excellent for FA:**
  - Mix of psychological and attitudinal scales
  - Natural clustering of belief systems
  - Real-world social science application
  - Demonstrates FA in opinion research

**What you'll learn:**
- Handling mixed variable types
- Identifying belief clusters
- Worldview factor interpretation
- Social science applications of FA
- Complex multidimensional attitude structures

---

### 4. **Motivational State Questionnaire (MSQ)** - `04_msq_emotional_states_fa.ipynb`
**Focus:** Common Factor Modeling for Emotional Dimensions

- **Dataset:** 75 emotional state variables (active, alert, angry, anxious, etc.)
- **Expected Factors:** Positive Affect, Negative Affect, Energy, Tension
- **Why it's excellent for FA:**
  - Large number of variables (75) ideal for FA
  - Clear bipolar structure (positive vs negative emotions)
  - Demonstrates common factor modeling
  - Psychological measurement application

**What you'll learn:**
- Working with high-dimensional data (75 variables)
- Identifying bipolar factors
- Affect structure analysis
- Mood dimension interpretation
- Creating composite mood scales

---

### 5. **World Values Survey (WVS)** - `05_world_values_survey_fa.ipynb`
**Focus:** Cross-Cultural Value Dimensions

- **Dataset:** Global dataset covering 50+ variables across dozens of countries
- **Expected Factors:** Post-Materialism, Social Trust, Traditional vs Secular values
- **Why it's excellent for FA:**
  - High volume of correlated variables
  - Cross-cultural comparisons
  - Complex sociological structures
  - Real research-grade dataset

**What you'll learn:**
- Complex dimensionality reduction (50+ variables)
- Cross-cultural factor analysis
- Sociological value dimensions
- Multi-level factor structures
- International comparative research

---

## 🎯 Learning Progression

The notebooks build progressively:

1. **Airline** → EFA fundamentals with clear business application
2. **Big Five** → CFA and testing theoretical models
3. **ACX Survey** → Complex belief structures
4. **MSQ** → High-dimensional emotional data
5. **World Values** → Cross-cultural and multi-level analysis

## 🔧 Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy factor-analyzer
```

**Required libraries:**
```python
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.6.0
factor-analyzer>=0.4.0
```

## 🚀 Getting Started

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start with Airline dataset** for FA fundamentals

3. **Progress to Big Five** to learn CFA

4. **Explore remaining notebooks** based on interest

Each notebook includes:
- Synthetic data generation (works without downloading real datasets)
- Step-by-step FA workflow
- Comprehensive visualizations
- Interpretation guidelines
- Practical insights

## 📊 What Each Notebook Covers

### Common Topics Across All Notebooks:

1. **Data Preparation**
   - Missing value handling
   - Standardization
   - Correlation analysis

2. **Suitability Assessment**
   - Bartlett's Test of Sphericity
   - Kaiser-Meyer-Olkin (KMO) measure
   - Sample adequacy

3. **Factor Number Determination**
   - Kaiser criterion (eigenvalue > 1)
   - Scree plot analysis
   - Parallel analysis
   - Cumulative variance explained

4. **Factor Extraction**
   - Principal axis factoring
   - Maximum likelihood
   - Rotation methods (Varimax, Promax, Oblimin, Quartimax)

5. **Interpretation**
   - Factor loadings matrix
   - Communalities and uniqueness
   - Factor naming and labeling
   - High-loading variable identification

6. **Validation**
   - Model fit assessment (RMSR)
   - Residual analysis
   - Reliability checks

7. **Application**
   - Factor score calculation
   - Using factors in downstream analysis
   - Business/research recommendations

### Advanced Topics by Notebook:

| Notebook | Unique Features |
|----------|----------------|
| **Airline** | Business metrics, service improvement prioritization, customer segmentation |
| **Big Five** | Reverse scoring, cross-loadings, comparing to theoretical model |
| **ACX** | Mixed data types, belief clustering, worldview analysis |
| **MSQ** | High-dimensional data, bipolar factors, affect structure |
| **World Values** | Cross-cultural analysis, multi-level factors, international comparisons |

## 🎓 Key Concepts Demonstrated

### 1. **Mathematical Foundations**
- Factor model equation: X = ΛF + ε
- Common variance vs unique variance
- Factor loading interpretation
- Eigenvalue decomposition

### 2. **EFA vs CFA**
- **Exploratory (EFA):** Discover underlying structure
- **Confirmatory (CFA):** Test theoretical model
- When to use each approach

### 3. **Rotation Methods**
- **Orthogonal (Varimax, Quartimax):** Uncorrelated factors
- **Oblique (Promax, Oblimin):** Allows factor correlation
- Choosing appropriate rotation

### 4. **Model Assessment**
- Goodness of fit indices
- Communality thresholds
- Residual diagnostics
- Cross-validation approaches

## 📈 Factor Analysis Best Practices

### When to Use Factor Analysis:

✅ **Good situations:**
- Many correlated variables (minimum 3-5 per factor)
- Sample size > 100 (preferably 200+)
- Ratio of cases to variables > 5:1
- Variables measured on continuous scale
- Theoretical expectation of underlying factors

❌ **Not recommended:**
- Small sample sizes (n < 100)
- Low inter-correlations among variables
- Purely exploratory with no theory
- Binary variables without proper handling

### Quality Indicators:

**Bartlett's Test:**
- Tests if correlation matrix differs from identity
- p < 0.05 indicates suitable data

**KMO Measure:**
- Overall KMO: measure of sampling adequacy
- 0.90+: Marvelous
- 0.80-0.89: Meritorious
- 0.70-0.79: Middling
- 0.60-0.69: Mediocre
- 0.50-0.59: Miserable
- <0.50: Unacceptable

**Communalities:**
- Proportion of variance explained
- Low (<0.3): variable doesn't fit well
- High (>0.7): well explained by factors

**Factor Loadings:**
- >0.7: Excellent
- 0.5-0.7: Good
- 0.3-0.5: Marginal
- <0.3: Usually not interpreted

## 🔍 Common Issues and Solutions

### Issue 1: Too Many/Few Factors
**Solution:** Use parallel analysis, consider scree plot elbow, check cumulative variance (aim for 60-70%)

### Issue 2: Variables Don't Load Cleanly
**Solution:** Try different rotations, consider oblique rotation, check for complex items

### Issue 3: Low Communalities
**Solution:** Remove problematic variables, check data quality, consider more factors

### Issue 4: Heywood Cases
**Solution:** Reduce factors, check for outliers, try different extraction method

### Issue 5: Factor Interpretation Unclear
**Solution:** Try different rotations, consult domain knowledge, check cross-loadings

## 💡 Tips for Success

1. **Sample Size Matters:** Aim for 200+ cases, minimum 5:1 ratio of cases to variables

2. **Check Assumptions:** Continuous variables, linear relationships, adequate correlations

3. **Use Multiple Criteria:** Don't rely on one method for factor number determination

4. **Rotate Thoughtfully:** Varimax for orthogonal, Promax for oblique if factors should correlate

5. **Interpret Carefully:** Consider both statistical loadings and theoretical meaning

6. **Validate:** Use split-sample or cross-validation when possible

7. **Report Completely:** Include all fit indices, loadings, communalities, variance explained

## 📝 Dataset Sources and Citations

- **Airline Passenger Satisfaction:** Kaggle (https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- **Big Five Personality:** Kaggle (https://www.kaggle.com/datasets/tunguz/big-five-personality-test)
- **ACX Survey:** Astral Codex Ten (https://astralcodexten.substack.com)
- **MSQ:** R psych package / FAVis
- **World Values Survey:** Official WVS (https://www.worldvaluessurvey.org)

## 📚 Further Reading

### Books:
- *Factor Analysis in R* by Beaujean (2014)
- *Exploratory Factor Analysis* by Fabrigar & Wegener (2012)
- *Applied Multivariate Statistical Analysis* by Johnson & Wichern

### Papers:
- Cattell (1966): The Scree Test
- Horn (1965): Parallel Analysis
- Kaiser (1960): The Varimax Rotation
- Costello & Osborne (2005): Best Practices in EFA

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional datasets
- Advanced rotation methods
- Multilevel factor analysis
- Bayesian factor analysis
- SEM integration

## 📄 License

Educational use. Please cite appropriately if used in research.

---

## 🎯 Learning Outcomes

After completing these notebooks, you will:

1. ✅ Understand when to use Factor Analysis
2. ✅ Assess data suitability (Bartlett's, KMO)
3. ✅ Determine optimal number of factors
4. ✅ Extract and rotate factors appropriately
5. ✅ Interpret factor loadings and structure
6. ✅ Calculate and use factor scores
7. ✅ Evaluate model adequacy
8. ✅ Apply FA in different domains (business, psychology, social science)
9. ✅ Distinguish between EFA and CFA
10. ✅ Communicate FA results effectively

---

## 🌟 Why These Datasets?

Each dataset was specifically chosen to teach different aspects of FA:

1. **Airline:** Practical business application with clear factors
2. **Big Five:** Theoretical validation and confirmatory analysis
3. **ACX:** Complex attitude structures and belief systems
4. **MSQ:** High-dimensional emotional data
5. **World Values:** Cross-cultural and sociological applications

Together, they provide comprehensive coverage of FA techniques and applications!

---

**Happy Factor Analyzing!** 🚀

For questions or feedback, please open an issue or reach out to maintainers.
