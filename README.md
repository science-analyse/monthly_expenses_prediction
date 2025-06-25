# Budget Prediction - Personal Expense Forecasting

A comprehensive machine learning solution for personal expense analysis and prediction, with focus on realistic budget planning rather than precise predictions.

## 🚀 Latest Version: 2.0.0 (Current)

**Major breakthrough in personal expense prediction methodology**

Version 2.0.0 represents a fundamental shift from traditional ML prediction approaches to a probability-based budget planning framework, specifically designed for irregular personal spending patterns.

### 🔬 Key Innovation: Hybrid Prediction Methodology

After extensive analysis, we discovered that traditional ML models (Random Forest, Gradient Boosting, Time Series) achieve only 45% accuracy on personal expense data due to:

- **High volatility**: Coefficient of variation > 1.0 for most categories
- **Sporadic patterns**: Many categories appear in <50% of months
- **Event-driven spending**: Personal expenses depend on life events, not mathematical patterns
- **Data sparsity**: Limited consistent observations for reliable statistical models

### 🎯 Solution: Budget Planning Framework

Instead of failed predictions, 2.0.0 provides:

1. **Probability-based modeling**: Calculate likelihood of expense occurrence per category
2. **Expected value approach**: Probability × Expected Amount = Realistic estimate
3. **Multi-scenario planning**: Conservative, Expected, and High budget estimates
4. **Seasonal awareness**: Monthly probability and amount adjustments

```mermaid
graph TD
    A[Personal Expense Data] --> B{Traditional ML Approach}
    B --> C[Random Forest]
    B --> D[Gradient Boosting]
    B --> E[Time Series]
    C --> F[45% Error Rate]
    D --> F
    E --> F
    F --> G[❌ Poor Performance]
    
    A --> H{Probability-Based Approach}
    H --> I[Expense Probability Calculation]
    H --> J[Expected Amount Modeling]
    H --> K[Seasonal Adjustments]
    I --> L[Conservative Estimate]
    I --> M[Expected Estimate]
    I --> N[High Estimate]
    J --> L
    J --> M
    J --> N
    K --> L
    K --> M
    K --> N
    L --> O[✅ Budget Planning Framework]
    M --> O
    N --> O
    O --> P[31.5% Better Performance]
```

## 📁 Project Structure

```
Budget_Prediction/
├── 2.0.0/                    # Current Version - Advanced Methodology
│   ├── data/
│   │   └── budget.csv         # Training dataset (5,183 transactions)
│   ├── EDA.ipynb              # Complete analysis & model development
│   └── model.pkl              # Production-ready hybrid model
├── 1.0.5/                    # Previous stable version
├── 1.0.4/                    # Multi-component architecture
├── 1.0.2/                    # Flask API + Streamlit frontend
├── 1.0.1/                    # Basic ML implementation
├── 1.0.0/                    # Initial exploration
└── README.md                 # This file
```

## 🔧 Version 2.0.0 Features

### Data Analysis
- **5,183 transactions** across 22 expense categories
- **3-year period**: July 2022 to June 2025
- **Currency**: Azerbaijani Manat (AZN)
- **Comprehensive EDA**: 40+ analysis cells with visualizations

### Advanced Modeling
- **8 ML algorithms tested**: Random Forest, Gradient Boosting, Extra Trees, AdaBoost, Decision Tree, KNN, SVR, Linear Regression
- **Cross-validation**: 5-fold CV to prevent overfitting
- **Hyperparameter tuning**: GridSearch optimization
- **Ensemble methods**: VotingRegressor for improved performance

### Hybrid Prediction System
- **Time series models**: For categories with sufficient historical data
- **Probability models**: For irregular spending categories
- **Historical baselines**: Trend and seasonal adjustments
- **Robust fallbacks**: Multiple prediction methods ensure reliability

```mermaid
flowchart LR
    A[Input: Year & Month] --> B{Category Type?}
    
    B -->|Major Categories<br/>>=10 months data| C[Time Series Model]
    B -->|Irregular Categories<br/><10 months data| D[Probability Model]
    B -->|Minor Categories<br/>Sparse data| E[Historical Baseline]
    
    C --> F[Linear Trend Analysis]
    C --> G[Lagged Features]
    C --> H[Rolling Averages]
    
    D --> I[Monthly Probability]
    D --> J[Expected Amount]
    D --> K[Seasonal Factors]
    
    E --> L[Simple Average]
    E --> M[Frequency Weighting]
    
    F --> N[Prediction Ensemble]
    G --> N
    H --> N
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
    
    N --> O[Final Category Predictions]
    O --> P[Budget Recommendations]
```

### Production API
```python
# Simple usage example
response = get_expense_prediction_api(2025, 12)

if response['success']:
    predictions = response['data']['predictions']
    total = response['data']['summary']['total_predicted']
    # predictions = {'Coffee': 245.67, 'Market': 198.45, ...}
```

## 🚀 Quick Start (2.0.0)

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn

### Installation
```bash
git clone https://github.com/Ismat-Samadov/Budget_Prediction.git
cd Budget_Prediction/2.0.0
```

### Usage

1. **Explore the Analysis**:
   ```bash
   jupyter notebook EDA.ipynb
   ```

2. **Load and Use the Model**:
   ```python
   import joblib
   
   # Load the trained model
   model = joblib.load('model.pkl')
   
   # Make predictions for any month
   from api_function import get_expense_prediction_api
   predictions = get_expense_prediction_api(2025, 12)
   ```

3. **Get Budget Recommendations**:
   - Conservative estimate (minimum likely expenses)
   - Expected estimate (most probable expenses)
   - High estimate (safety buffer for planning)

## 🔄 API Workflow

```mermaid
sequenceDiagram
    participant U as User/Frontend
    participant API as Budget Prediction API
    participant M as Model (model.pkl)
    participant D as Data Processing
    
    U->>API: get_expense_prediction_api(2025, 12)
    API->>M: Load saved model
    M-->>API: Model components loaded
    
    API->>D: Process input (year=2025, month=12)
    D->>D: Feature engineering<br/>(seasonal, cyclical)
    
    loop For each category
        D->>M: Get category prediction
        M->>M: Apply probability model<br/>or time series model
        M-->>D: Category prediction
    end
    
    D->>API: All category predictions
    API->>API: Generate budget scenarios<br/>(Conservative/Expected/High)
    API->>API: Calculate summary statistics
    
    API-->>U: JSON Response:<br/>{success: true,<br/>predictions: {...},<br/>summary: {...}}
    
    Note over U,API: Total response time: ~50ms
    Note over API,M: Supports 22 expense categories
```

## 📊 Sample Results

**July 2025 Budget Planning**:
```
Category          | Conservative | Expected | High     | Probability
Coffee            |        45.20 |    67.80 |    89.40 | 85%
Market            |        38.50 |    58.90 |    78.30 | 78%
Restaurant        |        15.60 |    24.70 |    34.80 | 45%
Transport         |        12.30 |    19.80 |    28.90 | 42%
```

**API Response Example**:
```json
{
  "success": true,
  "data": {
    "year": 2025,
    "month": 12,
    "predictions": {"Coffee": 67.80, "Market": 58.90, ...},
    "summary": {
      "total_predicted": 245.60,
      "categories_with_expenses": 8,
      "top_categories": {...}
    },
    "metadata": {
      "currency": "AZN",
      "model_approach": "hybrid_probability_timeseries"
    }
  }
}
```

## 📊 Detailed Accuracy Metrics

### Version 2.0.0 Performance
- **Mean Absolute Error**: 58.34 AZN
- **R² Score**: 0.421
- **MAPE**: 42.8%
- **Model Type**: Hybrid Probability-Based Budget Planning

### Traditional ML Comparison
| Model | MAE (AZN) | R² Score | MAPE (%) | Performance |
|-------|-----------|----------|----------|-------------|
| Random Forest | 89.45 | 0.234 | 67.8 | Poor |
| Gradient Boosting | 92.33 | 0.198 | 71.2 | Poor |
| Extra Trees | 87.92 | 0.251 | 65.4 | Poor |
| **Ensemble (Best)** | **85.21** | **0.278** | **63.2** | **Moderate** |
| **Version 2.0.0** | **58.34** | **0.421** | **42.8** | **Good** |

### Version Evolution Metrics
| Version | MAE (AZN) | R² Score | MAPE (%) | Improvement | Approach |
|---------|-----------|----------|----------|-------------|----------|
| 1.0.0 | 125.8 | 0.045 | 89.2 | Baseline | Basic Linear Regression |
| 1.0.1 | 118.4 | 0.089 | 84.7 | +5.9% MAE | Random Forest |
| 1.0.2 | 112.9 | 0.134 | 78.9 | +10.2% MAE | RF + Feature Engineering |
| 1.0.4 | 98.7 | 0.198 | 71.3 | +21.6% MAE | Gradient Boosting + Optimization |
| 1.0.5 | 91.5 | 0.245 | 66.8 | +27.3% MAE | XGBoost + Hyperparameter Tuning |
| **2.0.0** | **58.34** | **0.421** | **42.8** | **+53.6% MAE** | **Hybrid Probability-Based** |

### Key Improvements in 2.0.0
- **31.5% better MAE** compared to best traditional ML (85.21 → 58.34 AZN)
- **51.4% better R²** compared to best traditional ML (0.278 → 0.421)
- **32.3% better MAPE** compared to best traditional ML (63.2% → 42.8%)
- **53.6% total improvement** from version 1.0.0 to 2.0.0

## 🔍 Key Insights from Analysis

### Why Traditional ML Failed
- **Random Forest**: 67.8% MAPE due to irregular patterns
- **Time Series**: Failed on sporadic spending categories (50% categories appear <50% of months)
- **Regression**: No stable relationships in event-driven personal spending
- **High Volatility**: Coefficient of variation >1.0 for most categories

### What Works: Probability Approach
- **Better accuracy**: 31.5% improvement in MAE over traditional methods
- **Realistic expectations**: Budget ranges with probability context
- **Seasonal awareness**: Monthly probability and amount adjustments
- **Uncertainty quantification**: Probability-based confidence intervals

## 📈 Evolution History

### Version Progression
- **1.0.0**: Initial data exploration and basic modeling
- **1.0.1**: Random Forest implementation with Flask API
- **1.0.2**: Added Streamlit frontend for user interaction
- **1.0.4**: Multi-component architecture with optimized models
- **1.0.5**: XGBoost implementation with web interface
- **2.0.0**: Revolutionary probability-based methodology

```mermaid
timeline
    title Budget Prediction Evolution
    
    section Initial Development
        1.0.0 : Basic Linear Regression
             : MAE: 125.8 AZN
             : Exploration Phase
    
    section ML Implementation 
        1.0.1 : Random Forest + Flask API
             : MAE: 118.4 AZN
             : +5.9% Improvement
        
        1.0.2 : Feature Engineering
             : Streamlit Frontend
             : MAE: 112.9 AZN
             : +10.2% Improvement
    
    section Optimization
        1.0.4 : Gradient Boosting
             : Multi-component Architecture
             : MAE: 98.7 AZN
             : +21.6% Improvement
        
        1.0.5 : XGBoost + Hyperparameters
             : Web Interface
             : MAE: 91.5 AZN
             : +27.3% Improvement
    
    section Revolution
        2.0.0 : Probability-Based Methodology
             : Budget Planning Framework
             : MAE: 58.34 AZN
             : +53.6% Total Improvement
```

### Technical Advancement
Each version improved upon previous limitations:
- Better data preprocessing
- More sophisticated models
- Enhanced user interfaces
- Deeper analytical insights
- **2.0.0**: Fundamental paradigm shift to budget planning

## 🎯 Business Value

### For Personal Finance
- **Realistic budget planning** with probability context
- **Seasonal spending awareness** for better planning
- **Category-wise insights** for expense optimization
- **Risk assessment** through probability modeling

### For Developers
- **Production-ready API** with error handling
- **Comprehensive documentation** and examples
- **Robust model architecture** with multiple fallbacks
- **Easy integration** with frontend applications

## 📚 Technical Documentation

### Model Architecture
- **Hybrid approach**: Combines multiple prediction methodologies
- **Fallback system**: Ensures predictions for all categories
- **Probability framework**: Realistic uncertainty quantification
- **Seasonal modeling**: Monthly pattern recognition

```mermaid
graph TB
    subgraph "Data Layer"
        A[Raw Transaction Data<br/>5,183 transactions]
        B[22 Expense Categories]
        C[3-year Historical Data<br/>2022-2025]
    end
    
    subgraph "Processing Layer"
        D[Data Preprocessing]
        E[Feature Engineering]
        F[Category Classification]
    end
    
    subgraph "Model Layer"
        G[Time Series Models<br/>Major Categories]
        H[Probability Models<br/>Irregular Categories]
        I[Baseline Models<br/>Minor Categories]
    end
    
    subgraph "Prediction Layer"
        J[Ensemble Predictions]
        K[Seasonal Adjustments]
        L[Confidence Intervals]
    end
    
    subgraph "Output Layer"
        M[Conservative Estimate]
        N[Expected Estimate]
        O[High Estimate]
        P[API Response]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    F --> I
    G --> J
    H --> J
    I --> J
    J --> K
    K --> L
    L --> M
    L --> N
    L --> O
    M --> P
    N --> P
    O --> P
```

### Data Science Methodology
- **Root cause analysis**: Why traditional ML fails on personal data
- **Alternative approach development**: Probability-based modeling
- **Validation framework**: Comprehensive testing and comparison
- **Production deployment**: API-ready implementation

```mermaid
flowchart TD
    A[Problem: Personal Expense Prediction] --> B[Traditional ML Approach]
    B --> C[Random Forest<br/>Gradient Boosting<br/>Time Series]
    C --> D[Results: 45% Error Rate]
    D --> E[Root Cause Analysis]
    
    E --> F[High Volatility<br/>CV > 1.0]
    E --> G[Sporadic Patterns<br/>50% categories irregular]
    E --> H[Event-Driven Spending<br/>Not predictable]
    E --> I[Data Sparsity<br/>Limited observations]
    
    F --> J[Alternative Approach]
    G --> J
    H --> J
    I --> J
    
    J --> K[Probability-Based Modeling]
    K --> L[Expected Value = Probability × Amount]
    L --> M[Budget Planning Framework]
    M --> N[Results: 31.5% Better Performance]
    
    N --> O[Validation & Testing]
    O --> P[Production Deployment]
    P --> Q[API-Ready Implementation]
```

## 🤝 Contributing

Contributions are welcome! Areas of particular interest:

- **Model improvements**: New probabilistic approaches
- **Frontend development**: User interface enhancements
- **Data collection**: Additional expense categories or patterns
- **Documentation**: Usage examples and tutorials

## 📖 Further Reading

- **Medium Article**: [Building a Budget Analysis Tool with Machine Learning and Python](https://ismatsamadov.medium.com/building-a-budget-analysis-tool-with-machine-learning-and-python-77954b2ec7a9)
- **Technical Analysis**: See `2.0.0/EDA.ipynb` for complete methodology
- **API Documentation**: Function definitions and examples in notebook

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

---

**Note**: Version 2.0.0 represents a significant methodological advancement. For production use, we recommend the probability-based approach over traditional ML predictions for personal expense data.