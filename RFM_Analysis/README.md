# RFM Analysis Notebooks Collection

A comprehensive collection of Jupyter notebooks for learning and implementing RFM (Recency, Frequency, Monetary) analysis across different datasets and complexity levels.

## 📚 Notebooks Overview

| Notebook | Dataset | Complexity | Focus Areas | Best For |
|----------|---------|------------|-------------|----------|
| **01_online_retail_uci_rfm.ipynb** | Online Retail (UCI) | Medium | Standard RFM, Segmentation | General practice, E-commerce |
| **02_online_retail_ii_clv_rfm.ipynb** | Online Retail II | Medium-High | CLV prediction, Cohort analysis, Churn risk | Advanced analysis, CLV focus |
| **03_olist_brazil_rfm.ipynb** | Olist (Brazil) | High | Multi-table integration, Geographic, Reviews | Real-world complexity |
| **04_flo_shoes_omnichannel_rfm.ipynb** | FLO Shoes | Medium | Omni-channel behavior, Cross-channel | Retail with multiple channels |
| **05_cdnow_beginner_rfm.ipynb** | CDNOW | Low | RFM basics, Quick tests | Learning fundamentals |

---

## 🎯 Quick Start Guide

### For Beginners
Start with **05_cdnow_beginner_rfm.ipynb**
- Clean, small dataset
- Step-by-step RFM calculation
- Simple visualizations
- Clear business insights

### For Intermediate Users
Try **01_online_retail_uci_rfm.ipynb** or **04_flo_shoes_omnichannel_rfm.ipynb**
- Real-world datasets
- Multiple analysis techniques
- Comprehensive segmentation
- Channel analysis

### For Advanced Users
Explore **02_online_retail_ii_clv_rfm.ipynb** or **03_olist_brazil_rfm.ipynb**
- CLV prediction models
- Multi-table data integration
- Cohort and churn analysis
- Complex business scenarios

---

## 📊 Dataset Sources and Download Links

### 1. Online Retail (UCI)
**Download:**
- Kaggle: https://www.kaggle.com/datasets/jihyeseo/online-retail-data-set-from-uci-ml-repo
- UCI: https://archive.ics.uci.edu/ml/datasets/online+retail

**Details:**
- ~540K transactions
- ~4,300 customers
- Period: 2010-2011
- Format: Excel/CSV

### 2. Online Retail II
**Download:**
- UCI: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

**Details:**
- Extended version (2009-2011)
- Two sheets: Year 2009-2010 and Year 2010-2011
- Format: Excel

### 3. Olist (Brazil)
**Download:**
- Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

**Details:**
- 100K orders
- Multiple CSV files
- Period: 2016-2018
- Includes: orders, customers, products, reviews, payments, sellers

### 4. FLO Shoes
**Download:**
- Kaggle: https://www.kaggle.com/datasets/serhatckl/flo-rfm-analysis-dataset

**Details:**
- ~20K customers
- Omni-channel data (Online/Offline/Mobile)
- Pre-aggregated format

### 5. CDNOW
**Download:**
- GitHub: https://github.com/boboppie/CDNOW_RFM
- Often included in R packages (BTYD)

**Details:**
- ~70K transactions
- ~23K customers
- Period: 1997-1998
- Format: Text file or CSV

---

## 🛠️ Installation & Setup

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Optional (for enhanced visualizations)
```bash
pip install plotly
```

### Quick Setup
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
```

---

## 📖 Detailed Notebook Descriptions

### 1️⃣ Online Retail (UCI) - General Practice

**File:** `01_online_retail_uci_rfm.ipynb`

**What You'll Learn:**
- Standard RFM calculation methodology
- Quintile-based scoring (1-5)
- 11-segment customer classification
- Data cleaning best practices
- Basic to intermediate visualizations

**Key Sections:**
1. Data loading and exploration
2. Data preprocessing and cleaning
3. RFM metric calculation
4. Scoring and segmentation
5. Segment analysis and visualization
6. Business recommendations
7. Top customer identification

**Use Cases:**
- E-commerce customer analysis
- Marketing campaign targeting
- Customer retention strategies

---

### 2️⃣ Online Retail II - CLV & Advanced Analysis

**File:** `02_online_retail_ii_clv_rfm.ipynb`

**What You'll Learn:**
- Multiple CLV prediction approaches
- Cohort retention analysis
- Churn risk scoring
- Time-series customer behavior
- High-value at-risk identification

**Key Sections:**
1. Extended data loading (multiple years)
2. Standard RFM analysis
3. **4 CLV calculation methods:**
   - Simple extrapolation
   - Time-based prediction
   - Weighted CLV
   - Probabilistic CLV (BG/NBD inspired)
4. Cohort analysis with retention matrix
5. Churn risk modeling
6. Strategic prioritization

**Advanced Metrics:**
- Customer lifetime value predictions
- Probability of being "alive"
- Purchase rate estimation
- Expected purchases per year

**Use Cases:**
- Customer lifetime value optimization
- Churn prediction and prevention
- Long-term revenue forecasting
- Retention program ROI

---

### 3️⃣ Olist (Brazil) - Real-world Complexity

**File:** `03_olist_brazil_rfm.ipynb`

**What You'll Learn:**
- Multi-table data integration
- Geographic segmentation
- Review sentiment impact
- Product category analysis
- Payment method preferences

**Key Sections:**
1. Loading 8+ related tables
2. Data merging and integration
3. Enhanced RFM with additional features:
   - Geographic data (state, city)
   - Review scores
   - Product category diversity
   - Payment preferences
4. Multi-dimensional analysis
5. State-level insights
6. Cross-feature correlations

**Unique Features:**
- Real anonymized Brazilian e-commerce data
- Multiple data sources integration
- Geographic and behavioral insights
- Review quality analysis

**Use Cases:**
- Complex business scenarios
- Geographic expansion planning
- Product portfolio optimization
- Customer satisfaction analysis

---

### 4️⃣ FLO Shoes - Omni-channel Focus

**File:** `04_flo_shoes_omnichannel_rfm.ipynb`

**What You'll Learn:**
- Online vs offline behavior analysis
- Channel preference classification
- Cross-channel customer value
- Omni-channel uplift calculation
- Channel-specific RFM scores

**Key Sections:**
1. Channel data loading
2. Overall RFM calculation
3. **Channel-specific analysis:**
   - Online-only RFM
   - Offline-only RFM
   - Omni-channel RFM
4. Channel preference classification:
   - Online-Dominant
   - Offline-Dominant
   - Omni-channel
   - Single-Channel
5. Cross-channel behavior patterns
6. AOV comparison by channel

**Unique Insights:**
- Omni-channel customers typically have 30-50% higher value
- Channel migration opportunities
- Cross-channel promotion strategies

**Use Cases:**
- Retail with physical and online presence
- Channel strategy optimization
- Cross-channel marketing
- Store + web integration

---

### 5️⃣ CDNOW - Beginner Friendly

**File:** `05_cdnow_beginner_rfm.ipynb`

**What You'll Learn:**
- RFM fundamentals step-by-step
- Basic scoring methodology
- Simple 4-segment classification
- Quick business insights
- Reusable RFM function

**Key Sections:**
1. Simple data loading
2. **Step-by-step RFM calculation:**
   - Recency calculation explained
   - Frequency calculation explained
   - Monetary calculation explained
3. Scoring with quintiles
4. Simple segmentation (Best/High/Medium/Low)
5. Quick visualizations
6. Actionable insights
7. **Bonus:** Reusable RFM function

**Perfect For:**
- Learning RFM basics
- Quick prototyping
- Teaching material
- Small dataset testing

**Use Cases:**
- Educational purposes
- RFM proof of concept
- Quick customer analysis
- Team training

---

## 🎨 Visualization Examples

Each notebook includes:

### Standard Visualizations
- RFM distribution histograms
- Score distribution bar charts
- Segment distribution (bar + pie)
- 3D scatter plots (R, F, M)
- Correlation heatmaps

### Advanced Visualizations
- Cohort retention heatmaps
- Geographic distribution maps
- Channel comparison charts
- CLV prediction plots
- Churn risk matrices

---

## 💡 Common RFM Segments Explained

| Segment | RFM Characteristics | Description | Action |
|---------|-------------------|-------------|--------|
| **Champions** | High R, F, M (444+) | Best customers who buy recently, often, and spend most | Reward, early access, VIP treatment |
| **Loyal Customers** | High F, M (x4x+) | Regular customers with high value | Upsell, loyalty program |
| **Potential Loyalists** | Recent, moderate F (4-5, 2-3, x) | Recent customers who could become loyal | Engage, increase frequency |
| **New Customers** | Very recent, low F (5, 1-2, x) | Just started buying | Onboard, first repeat purchase |
| **At Risk** | Low R, high F, M (1-2, 4+, 4+) | Were great customers but haven't returned | Win-back campaigns |
| **Can't Lose Them** | Lowest R, high F, M (1, 4+, 4+) | Used to be best customers | Aggressive win-back |
| **Hibernating** | Low R, F (1-2, 1-2, x) | Long time inactive, low value | Re-activation or ignore |
| **Lost** | Lowest scores overall | Haven't bought in long time | Ignore or final attempt |

---

## 📈 Business Applications

### Marketing
- **Email Campaign Targeting:** Send different messages to different segments
- **Offer Optimization:** Discount depth based on segment
- **Channel Selection:** Use preferred channels by segment

### Customer Success
- **Churn Prevention:** Identify at-risk customers early
- **Upsell Opportunities:** Target promising customers
- **Customer Health Scoring:** Monitor segment transitions

### Finance
- **Revenue Forecasting:** Use CLV predictions
- **Budget Allocation:** Spend based on segment value
- **ROI Measurement:** Track campaign effectiveness by segment

### Product
- **Feature Prioritization:** Based on champion feedback
- **Product Recommendations:** Aligned with purchase history
- **Inventory Management:** Stock based on segment demand

---

## 🔧 Customization Tips

### Adjusting Scoring
```python
# Change number of bins
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=10, labels=range(10,0,-1))  # 10 bins instead of 5

# Manual thresholds instead of quantiles
rfm['R_Score'] = pd.cut(rfm['Recency'], 
                        bins=[0, 30, 60, 90, 180, np.inf],
                        labels=[5, 4, 3, 2, 1])
```

### Custom Segments
```python
def custom_segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    # Define your own logic
    if r == 5 and f == 5 and m == 5:
        return 'Platinum'
    # ... add more conditions
    return 'Standard'

rfm['Custom_Segment'] = rfm.apply(custom_segment, axis=1)
```

### Weighted RFM
```python
# Give more weight to certain metrics
rfm['Weighted_Score'] = (
    rfm['R_Score'] * 0.3 +  # 30% weight on recency
    rfm['F_Score'] * 0.3 +  # 30% weight on frequency
    rfm['M_Score'] * 0.4    # 40% weight on monetary
)
```

---

## 📊 Expected Outputs

Each notebook generates:

### CSV Files
- `*_rfm_results.csv` - Full RFM table with scores and segments
- `*_segment_summary.csv` - Aggregated metrics by segment
- `*_top_customers.csv` - High-value customer list

### Visualizations
- Distribution plots (PNG/interactive)
- Segment analysis charts
- Correlation matrices
- 3D visualizations

### Business Insights
- Segment breakdown with counts and revenue
- Top customer identification
- At-risk customer alerts
- Revenue opportunities

---

## 🎓 Learning Path

### Week 1: Fundamentals
1. **Day 1-2:** Study CDNOW notebook (`05_cdnow_beginner_rfm.ipynb`)
2. **Day 3-4:** Practice with your own small dataset
3. **Day 5:** Understand segment definitions thoroughly

### Week 2: Intermediate
1. **Day 1-3:** Work through Online Retail UCI (`01_online_retail_uci_rfm.ipynb`)
2. **Day 4-5:** Experiment with different scoring methods
3. **Weekend:** Create custom segments for specific business

### Week 3: Advanced
1. **Day 1-2:** Study CLV prediction (`02_online_retail_ii_clv_rfm.ipynb`)
2. **Day 3-4:** Learn omni-channel analysis (`04_flo_shoes_omnichannel_rfm.ipynb`)
3. **Day 5:** Complex integration (`03_olist_brazil_rfm.ipynb`)

### Week 4: Application
1. Apply to real business dataset
2. Present findings to stakeholders
3. Implement in production

---

## 🤝 Contributing

These notebooks are designed to be educational and practical. Suggestions for improvements:
- Additional datasets
- Alternative scoring methods
- New visualization techniques
- Business case studies
- Industry-specific adaptations

---

## 📚 Additional Resources

### Books
- "Customer Data Science" by Ryohei Fujimaki
- "Data Science for Business" by Foster Provost

### Papers
- "Counting Your Customers" (Schmittlein, Morrison, Colombo, 1987)
- "Predicting Customer Lifetime Value" (Fader, Hardie)

### Online Courses
- Coursera: Customer Analytics
- DataCamp: Marketing Analytics

### Tools
- Python libraries: pandas, numpy, matplotlib, seaborn
- BI tools: Tableau, Power BI for visualization
- CRM integration: Salesforce, HubSpot

---

## ⚠️ Common Pitfalls to Avoid

1. **Don't ignore data quality**
   - Remove cancelled orders
   - Handle negative values
   - Check for duplicates

2. **Don't use arbitrary cutoffs without business context**
   - Quintiles work but may not reflect business reality
   - Validate with domain experts

3. **Don't create too many segments**
   - 8-12 segments is usually optimal
   - More segments = harder to action

4. **Don't forget to update regularly**
   - RFM is a snapshot
   - Recalculate monthly or quarterly

5. **Don't ignore segment transitions**
   - Track how customers move between segments
   - This reveals behavior patterns

---

## 📞 Support

For issues, questions, or suggestions about these notebooks:
- Review the code comments
- Check the troubleshooting section in each notebook
- Refer to the pandas/numpy documentation
- Search for similar issues on Stack Overflow

---

## 📄 License

These notebooks are provided for educational purposes. Datasets have their own licenses:
- UCI datasets: Check UCI ML Repository terms
- Kaggle datasets: Check individual dataset licenses
- CDNOW: Public domain for research

---

## 🚀 Next Steps

After mastering RFM analysis:
1. **Implement in production** - Automate monthly RFM calculation
2. **A/B Testing** - Test segment-specific campaigns
3. **Predictive Models** - Build ML models on top of RFM
4. **Real-time Scoring** - Calculate RFM on streaming data
5. **Integration** - Connect to CRM, email platforms, BI tools

---

**Happy Analyzing! 🎉**

Start with the CDNOW notebook if you're new to RFM, or jump to the dataset that matches your business need. Each notebook is self-contained and fully documented.
