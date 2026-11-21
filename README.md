# car-insurance-prediction
# Ta'meenak · تأمينك

**Two neural networks for car insurance: one predicts claim risk, the other predicts the yearly premium.**

Deep Learning Project · Team: Tarfah Alsunaidi, Razan, Aleen, Layan

---

## The problem

Every car insurer has to answer two questions about each driver:

1. **Will this driver file a claim?** (Should we cover them, and how risky are they?)
2. **How much should they pay?** (A fair yearly premium for this driver and vehicle.)

Charging everyone the same price either loses money on risky drivers or overcharges careful ones.

## Our solution

We built two Artificial Neural Networks (ANNs) that work together to give one assessment per driver:

| | Model 1: Claim Risk | Model 2: Premium |
|---|---|---|
| Task | Binary classification | Regression |
| Output | Claim probability, then Low / Medium / High risk | Yearly premium (EUR) |
| Dataset | Kaggle Car Insurance Data (10,000 rows) | Mendeley Motor Vehicle Insurance Data (105,555 rows) |
| Notebook | `ClaimRisk_EDA_Preprocessing_and_ANN_Model.ipynb` | `Deep_Learning_Project.ipynb` |

---

## Datasets

| Model | Source | Size | Target |
|---|---|---|---|
| Claim Risk | [Kaggle: Car Insurance Data](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data) | 10,000 × 19 | `OUTCOME` (1 = claim, 0 = no claim) |
| Premium | [Mendeley: Dataset of an actual motor vehicle insurance portfolio](https://data.mendeley.com/datasets/5cxyb5fp4f/2) | 105,555 × 30 | `Premium` (yearly premium of the policy) |

The premium dataset comes from a Spanish insurer's portfolio (Nov 2015 to Dec 2018), so prices are in euros and reflect that market.

---

## Model 1: Claim Risk

### Data and EDA
- 31.3% of drivers filed a claim, so the classes are imbalanced. We track AUC and Recall, not only Accuracy.
- Driving experience is the strongest signal: the claim rate falls from 63% (0–9 years) to 2% (30+ years).
- A higher credit score and owning the vehicle are linked to fewer claims.

### Preprocessing
- Dropped `ID` (identifier) and `RACE` (fairness). 16 features remain.
- Filled missing `CREDIT_SCORE` (982) and `ANNUAL_MILEAGE` (957) with the median.
- One-hot encoded the categorical features and scaled the numeric ones (`StandardScaler`), giving 33 inputs.
- Stratified 70 / 15 / 15 split, so each part keeps the same claim rate.

### Architecture (final, selected by Keras Tuner)
```
Input (33) → Dense(16, ReLU) → Dropout(0.2) → Dense(16, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)
```
Loss: binary cross-entropy · Optimizer: Adam (lr = 0.001) · Callbacks: EarlyStopping, ReduceLROnPlateau

We used an ANN rather than an autoencoder because the task is supervised (we have a labeled target), the data is tabular, and claims are not rare.

### Experiments (validation AUC)
| Experiment | Val AUC |
|---|---|
| Baseline (32 → 16) | 0.9137 |
| L2 (0.001) | 0.9143 |
| Dropout (0.3) | 0.9167 |
| Batch Normalization (dropped) | 0.9019 |
| **Keras Tuner: 16 → 16, Dropout 0.2 (final)** | **0.9173** |

### Test results
| Metric | Value |
|---|---|
| AUC | 0.9155 |
| Accuracy | 0.853 |
| Precision (claim class) | 0.77 |
| Recall (claim class) | 0.76 |

Confusion matrix on the test set (1,500 drivers): 356 claims caught, 114 missed, 106 false alarms, 924 correct non-claims.

**Risk levels:** probability < 0.3 = Low, 0.3–0.6 = Medium, > 0.6 = High.

---

## Model 2: Premium

### Data cleaning
- **Missing values:** `Length` (9.8%) filled with the median of the same vehicle type, plus a `Length_missing` flag. `Type_fuel` (1.7%) set to `Unknown`.
- **Invalid values:** `Power = 0` (1,764 rows) treated as missing and filled with the median power of the same vehicle type, plus a `Power_missing` flag. `N_doors = 0` kept, because those rows are all motorbikes or agricultural vehicles.
- **Feature engineering:** `Age`, `Driving_experience`, `Licence_age`, `Contract_years` and `Vehicle_age` computed from the date columns.
- **Outliers:** IQR method on `Premium`, `Value_vehicle`, `Age`, `Driving_experience` and `Length`. This removed 12,154 rows (11.5%), leaving 93,401.

### EDA highlights
- Median premium ≈ €284. Most premiums fall between €200 and €400.
- 82% of policy-years have no claim.
- Cars and vans pay ≈ €290, motorbikes ≈ €130 and agricultural vehicles ≈ €85.
- Drivers aged 18–25 pay the most (≈ €364), and premiums fall with age.

### Preprocessing
- 25 features (19 numeric + 6 categorical), one-hot encoded into 34 inputs and scaled.
- The target is `log(1 + Premium)` to reduce skewness. Predictions are converted back to euros.
- **Split by customer ID (70 / 15 / 15):** each customer has one row per year, so a customer never appears in both training and test (this prevents data leakage).
- This year's claims (`N_claims_year`, `Cost_claims_year`) are excluded because they are not known at pricing time.

### Architecture (final)
```
Input (34) → Dense(128, ReLU, L2) → Dropout(0.1) → Dense(64, ReLU, L2) → Dropout(0.1) → Dense(1, linear)
```
Loss: MSE · Optimizer: SGD + Momentum 0.9 (lr = 0.01) · EarlyStopping (patience 10, stopped at epoch 98)

### Experiments (validation MAE on log premium)
| Experiment | Val MAE |
|---|---|
| Optimizers (10 epochs): SGD / SGD + Momentum / RMSProp / Adam | 0.204 / **0.177** / 0.207 / 0.184 |
| Learning rate (SGD + Momentum): 0.1 / 0.01 / 0.001 / 0.0001 | 0.264 / **0.178** / 0.193 / 0.290 |
| Baseline (128 → 64) | 0.1759 |
| L2 (0.001) | 0.1712 |
| Dropout (0.2) | 0.1755 |
| Early Stopping | 0.1753 |
| Keras Tuner (best: 3 × 128, no dropout) | 0.1740 |
| **L2 + Dropout (0.1) + Early Stopping (final)** | **0.1697** |

### Test results
| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Our ANN** | **€48.47** | **€65.52** | **0.456** |
| Linear Regression | €51.66 | €69.92 | 0.380 |

Train MAE (€47.84) is close to test MAE (€48.47), so the model does not overfit.

**Examples:** a 22-year-old with 2 years of experience and a powerful car gets €607.41/year. A 50-year-old with 30 years of experience and a small 12-year-old car gets €231.99/year.
