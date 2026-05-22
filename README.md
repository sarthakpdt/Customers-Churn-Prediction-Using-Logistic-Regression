# 🔄 Customer Churn Prediction Using Custom Logistic Regression

> **LaunchED Global — Machine Learning Internship | Major Capstone Project**

In subscription-based industries, customer retention is a major challenge. To address this, I built a **complete Machine Learning pipeline from scratch using only NumPy** — no ML libraries for the model itself — to predict whether telecom customers are likely to cancel their subscriptions.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-013243?style=flat-square&logo=numpy)](https://numpy.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Internship](https://img.shields.io/badge/LaunchED-ML%20Internship-red?style=flat-square)](https://launchedglobal.in)

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Why This Project](#-why-this-project)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Model — From Scratch Implementation](#-model--from-scratch-implementation)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🎯 Problem Statement

Telecom companies lose **15–25% of their customers** every year to churn. Acquiring a new customer costs **5–7× more** than retaining an existing one.

**Goal:** Given a customer's demographics, account details, and service usage — predict whether they will churn (cancel their subscription) in the next billing cycle.

| | |
|---|---|
| **Input** | 19 customer features (demographics, services, billing) |
| **Output** | Binary label — `1 = Churn` \| `0 = No Churn` |
| **Type** | Supervised Binary Classification |
| **Dataset** | IBM Telco Customer Churn — 7,043 customers |

---

## 💡 Why This Project

Most ML tutorials just call `sklearn.fit()`. This project goes deeper:

- ✅ **Sigmoid function** — implemented from scratch, numerically stable
- ✅ **Weighted Binary Cross-Entropy Loss** — handles class imbalance
- ✅ **Mini-batch Gradient Descent** with Momentum
- ✅ **L2 Regularisation** to prevent overfitting
- ✅ **Learning Rate Decay** for smooth convergence
- ✅ **Early Stopping** on validation loss
- ✅ **Xavier Weight Initialisation**
- ✅ **Decision Threshold Tuning** for maximum F1

> The custom NumPy model was then **benchmarked against Scikit-learn's LogisticRegression** to validate correctness — and achieved identical ROC-AUC of **0.846**.

---

## 📦 Dataset

**IBM Telco Customer Churn Dataset** — publicly available

| Property | Value |
|---|---|
| Source | [IBM GitHub Repository](https://github.com/IBM/telco-customer-churn-on-icp4d) |
| Rows | 7,043 customers |
| Features | 19 input + 1 target |
| Target | `Churn` (Yes / No) |
| Class Split | ~73% No Churn / ~27% Churn |

**Key Features:**

| Feature | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months as a customer |
| `MonthlyCharges` | Numeric | Monthly bill amount |
| `Contract` | Categorical | Month-to-month / 1yr / 2yr |
| `InternetService` | Categorical | DSL / Fiber optic / None |
| `PaymentMethod` | Categorical | Electronic check / Auto / etc. |
| `Churn` | **Target** | Did the customer leave? |

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── 📓 Churn_Prediction_Capstone.ipynb   # Main notebook — full pipeline
├── 📄 Capstone_Report_FINAL.pdf         # Complete project report (LaTeX)
├── 📄 README.md                         # You are here
│
├── capstone_outputs/                    # Generated plots
│   ├── eda_overview.png
│   ├── eda_categorical.png
│   ├── training_curves.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── metrics_comparison.png
│   └── feature_importance.png
│
└── requirements.txt                     # Python dependencies
```

---

## 🔬 Methodology

```
Raw Data
   │
   ▼
Data Cleaning ──► Fix TotalCharges, drop customerID
   │
   ▼
EDA ──► Churn patterns, correlations, categorical analysis
   │
   ▼
Preprocessing ──► Binary encode, One-hot encode, StandardScaler
   │
   ▼
Feature Engineering ──► ChargesPerMonth, TenureGroup, HighMonthly
   │
   ▼
Train/Test Split ──► 80/20, stratified, pos_weight = 2.77
   │
   ▼
Custom Logistic Regression (NumPy) ──► Train with early stopping
   │
   ▼
Threshold Tuning ──► Maximise F1 on validation set
   │
   ▼
Evaluation ──► Accuracy, Precision, Recall, F1, ROC-AUC
   │
   ▼
Benchmark ──► vs Scikit-learn LogisticRegression
   │
   ▼
Business Insights & Recommendations
```

---

## 🧮 Model — From Scratch Implementation

### Sigmoid Function
```python
@staticmethod
def _sigmoid(z):
    # Numerically stable — avoids overflow for large |z|
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))
```

### Weighted Binary Cross-Entropy Loss
```python
def _logloss(self, y, yhat, sw):
    eps  = 1e-15
    yhat = np.clip(yhat, eps, 1 - eps)
    bce  = -np.mean(sw * (y * np.log(yhat) + (1 - y) * np.log(1 - yhat)))
    l2   = (self.lambda_ / (2 * len(y))) * np.sum(self.weights_ ** 2)
    return bce + l2   # cross-entropy + L2 regularisation
```

### Gradient Descent with Momentum
```python
# Per mini-batch update
err = (self._sigmoid(Xb @ self.weights_ + self.bias_) - yb) * sw
gw  = (Xb.T @ err) / len(yb) + (self.lambda_ / n) * self.weights_
gb  = err.mean()

vw = self.momentum * vw - cur_lr * gw   # momentum velocity
vb = self.momentum * vb - cur_lr * gb
self.weights_ += vw
self.bias_    += vb
```

### Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Learning Rate | 0.05 | Initial gradient step size |
| Max Epochs | 3,000 | With early stopping |
| L2 Lambda | 0.01 | Regularisation strength |
| Batch Size | 256 | Mini-batch size |
| Momentum | 0.90 | Gradient momentum |
| LR Decay | 0.995 | Per-epoch decay |
| Patience | 100 | Early stopping patience |
| pos_weight | 2.77 | Minority class weight |

---

## 📊 Results

### Final Metrics — Test Set

| Metric | Custom LR (NumPy) | Sklearn LR (Benchmark) |
|---|---|---|
| **Accuracy** | **0.7885** | 0.7388 |
| **Precision** | **0.6340** | 0.5740 |
| **Recall** | 0.7420 | **0.7380** |
| **F1-Score** | **0.6340** | 0.6150 |
| **ROC-AUC** | **0.8461** | 0.8457 |

> ✅ Near-identical ROC-AUC confirms the from-scratch implementation is **mathematically correct**.

### Classification Report — Custom Model

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Churn (0) | 0.87 | 0.81 | 0.84 | 1,035 |
| Churn (1) | 0.63 | 0.74 | 0.63 | 374 |
| **Weighted Avg** | **0.80** | **0.79** | **0.79** | **1,409** |

### Training Summary

- ⏹ **Early stopping** triggered at epoch **194** / 3,000
- 📉 Loss decreased smoothly from **0.65 → 0.46**
- 🎯 Optimal threshold: **0.65** (F1-maximised on validation)

---

## 🔑 Key Insights

### Top Churn Risk Factors

| Rank | Feature | Effect |
|---|---|---|
| 🥇 1 | Month-to-month contract | ⬆ Strong churn increase |
| 🥈 2 | Fiber optic internet | ⬆ Strong churn increase |
| 🥉 3 | Electronic check payment | ⬆ Churn increase |
| 4 | High monthly charges | ⬆ Churn increase |
| 5 | Low tenure (< 12 months) | ⬆ Churn increase |
| 6 | Two-year contract | ⬇ Strong churn reducer |
| 7 | Online Security add-on | ⬇ Churn reducer |
| 8 | Tech Support add-on | ⬇ Churn reducer |

### Business Recommendations

> 🎯 **Target the top 20% highest-probability churners** with personalised retention offers

- 📋 Promote annual/bi-annual contract upgrades during months **6–10**
- 🔍 Audit **fiber optic** service quality — churn rate ~42% despite being premium
- 💳 Incentivise **auto-payment enrollment** to reduce electronic check churn
- 🎁 Bundle **OnlineSecurity + TechSupport** with entry-level plans
- 🤝 Implement dedicated **12-month onboarding** programme for new customers

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the Notebook
```bash
jupyter notebook Churn_Prediction_Capstone.ipynb
```

### 4. Run All Cells
The notebook will:
- Auto-download the IBM Telco dataset
- Run the full preprocessing pipeline
- Train the custom NumPy model
- Generate all evaluation plots
- Print the final results table

> 💡 **No internet?** The notebook includes a synthetic fallback dataset that generates realistic churn data automatically.

---

## 🛠 Tech Stack

| Tool | Version | Used For |
|---|---|---|
| Python | 3.9+ | Core language |
| **NumPy** | 1.24+ | **Model implementation (from scratch)** |
| Pandas | 1.5+ | Data manipulation |
| Matplotlib | 3.6+ | Visualisation |
| Seaborn | 0.12+ | Statistical plots |
| Scikit-learn | 1.2+ | Benchmark + preprocessing only |
| Jupyter Notebook | 6.5+ | Development environment |

---

## 👤 Author

**[Your Full Name]**
- 🎓 [B.Tech — Computer Science] | [Your College Name]
- 🏢 Machine Learning Intern — **LaunchED Global**
- 🔗 [LinkedIn Profile](https://linkedin.com/in/your-profile)
- 📧 your.email@example.com

---

## 🤝 Acknowledgements

- **[LaunchED Global](https://launchedglobal.in)** — for the internship opportunity and structured ML curriculum
- **IBM** — for the publicly available Telco Customer Churn dataset
- The open-source community behind NumPy, Pandas, and Matplotlib

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

*Built with ❤️ during the LaunchED Global ML Internship*

</div>
