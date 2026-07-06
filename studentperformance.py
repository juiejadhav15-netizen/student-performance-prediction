# STEP 1 — Load & First Look at Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# Load data
df = pd.read_csv('C:/Users/juiej/OneDrive/Desktop/ML code/StudentsPerformance.csv')

print("=" * 50)
print("STEP 1: FIRST LOOK AT THE DATA")
print("=" * 50)

print(f"\nShape: {df.shape[0]} students, {df.shape[1]} columns")
print(f"\nColumn Names:\n{list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")

# STEP 2 — Descriptive Statistics
print("=" * 50)
print("STEP 2: DESCRIPTIVE STATISTICS")
print("=" * 50)

scores = df[['math score', 'reading score', 'writing score']]

print("\n--- CENTRAL TENDENCY ---")
print(f"{'Subject':<18} {'Mean':>8} {'Median':>8} {'Mode':>8}")
print("-" * 45)
for col in scores.columns:
    print(f"{col:<18} {scores[col].mean():>8.1f} {scores[col].median():>8.1f} {scores[col].mode()[0]:>8}")

print("\n--- SPREAD / DISPERSION ---")
print(f"{'Subject':<18} {'Std Dev':>8} {'Min':>6} {'Max':>6} {'Range':>8}")
print("-" * 50)
for col in scores.columns:
    print(f"{col:<18} {scores[col].std():>8.1f} {scores[col].min():>6} "
          f"{scores[col].max():>6} {scores[col].max()-scores[col].min():>8}")

print("\n--- FULL SUMMARY ---")
print(scores.describe().round(2))

# STEP 3 — Visualize Data
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Student Performance — Complete Data Analysis", fontsize=15, fontweight='bold')

# Row 1: Distribution
for i, col in enumerate(['math score', 'reading score', 'writing score']):
    axes[0][i].hist(df[col], bins=20, color=['steelblue','salmon','green'][i],
                    edgecolor='white', alpha=0.8)
    axes[0][i].axvline(df[col].mean(), color='red', linestyle='--', lw=2,
                       label=f'Mean={df[col].mean():.1f}')
    axes[0][i].axvline(df[col].median(), color='orange', linestyle='--', lw=2,
                       label=f'Median={df[col].median():.1f}')
    axes[0][i].set_title(f'{col} Distribution')
    axes[0][i].set_xlabel('Score')
    axes[0][i].set_ylabel('Number of Students')
    axes[0][i].legend(fontsize=8)

# Row 2: Box plots (fixed for Matplotlib 3.9+ and Seaborn)
axes[1][0].boxplot([df['math score'], df['reading score'], df['writing score']],
                   tick_labels=['Math', 'Reading', 'Writing'],  # FIXED: was 'labels'
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
axes[1][0].set_title('Score Spread — Box Plot')
axes[1][0].set_ylabel('Score')

# Math score by lunch type (FIXED: use color, not palette)
sns.boxplot(data=df, x='lunch', y='math score', ax=axes[1][1],
            color='lightblue', legend=False)
axes[1][1].set_title('Math Score by Lunch Type')
axes[1][1].set_xlabel('Lunch Type')
axes[1][1].set_ylabel('Math Score')

# Math score by test prep (FIXED: use color, not palette)
sns.boxplot(data=df, x='test preparation course', y='math score', ax=axes[1][2],
            color='lightblue', legend=False)
axes[1][2].set_title('Math Score by Test Prep')
axes[1][2].set_xlabel('Test Preparation')
axes[1][2].set_ylabel('Math Score')

plt.tight_layout()
plt.show()

# STEP 4 — Check & Clean Data
print("=" * 50)
print("STEP 4: DATA CLEANING")
print("=" * 50)

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing cells: {missing.sum()}")

print("\n--- OUTLIER DETECTION (3-Sigma Rule) ---")
for col in ['math score', 'reading score', 'writing score']:
    mean = df[col].mean()
    std  = df[col].std()
    lower = mean - 3*std
    upper = mean + 3*std
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers found "
          f"(normal range: {lower:.0f} to {upper:.0f})")

print(f"\nDuplicate rows: {df.duplicated().sum()}")
print("\n✅ Data looks clean — no major issues found!")

# STEP 5 — Prepare Data for ML
df_ml = df.copy()

# Use separate LabelEncoders for each column
encoders = {}
text_cols = ['gender', 'race/ethnicity', 'parental level of education',
             'lunch', 'test preparation course']

print("Converting text → numbers:")
for col in text_cols:
    if col not in df_ml.columns:
        # Try common alternative name
        alt = col.replace('parental education', 'parental level of education')
        if alt in df_ml.columns:
            col = alt
            text_cols[text_cols.index(col)] = alt
        else:
            raise ValueError(f"Column '{col}' not found in dataset")
    
    le = LabelEncoder()
    encoders[col] = le
    original = df_ml[col].unique()[:3]
    df_ml[col] = le.fit_transform(df_ml[col])
    encoded  = df_ml[col].unique()[:3]
    print(f"  {col}: {original} → {encoded}")

X_all = df_ml[['reading score', 'writing score',
               'gender', 'lunch', 'test preparation course']]
Y     = df_ml['math score']

print(f"\nX shape (inputs) : {X_all.shape}")
print(f"Y shape (target) : {Y.shape}")
print(f"\nTarget: Predict math score")
print(f"Inputs: reading, writing, gender, lunch, test_prep")

X_train, X_test, Y_train, Y_test = train_test_split(
    X_all, Y, test_size=0.2, random_state=42)

print(f"\nTraining data : {X_train.shape[0]} students (80%)")
print(f"Testing data  : {X_test.shape[0]}  students (20%)")
print("→ Machine learns on 160 students, predicts for 40 students it has never seen")

# STEP 6 — SLR
print("=" * 50)
print("STEP 6: SIMPLE LINEAR REGRESSION")
print("Using ONLY reading score to predict math score")
print("=" * 50)

X_slr_train = X_train[['reading score']]
X_slr_test  = X_test[['reading score']]

slr = LinearRegression()
slr.fit(X_slr_train, Y_train)

slr_pred  = slr.predict(X_slr_test)
slr_r2    = r2_score(Y_test, slr_pred)
slr_mae   = mean_absolute_error(Y_test, slr_pred)

print(f"\nFormula: math_score = {slr.coef_[0]:.2f} × reading_score + {slr.intercept_:.2f}")
print(f"\nR² Score : {slr_r2:.2f}")
print(f"MAE      : {slr_mae:.2f} marks off on average")

print(f"\nIn simple words:")
print(f"→ When predicting math score, model is wrong by {slr_mae:.1f} marks on average")

print(f"\n{'Actual':>8} {'Predicted':>10} {'Difference':>12}")
print("-" * 35)
for actual, pred in zip(list(Y_test[:8]), slr_pred[:8]):
    diff = abs(actual - pred)
    flag = "✅" if diff < 10 else "⚠️"
    print(f"{actual:>8} {pred:>10.1f} {diff:>10.1f}  {flag}")

# STEP 7 — MLR
print("\n" + "=" * 50)
print("STEP 7: MULTIPLE LINEAR REGRESSION")
print("Using ALL 5 features to predict math score")
print("=" * 50)

mlr = LinearRegression()
mlr.fit(X_train, Y_train)

mlr_pred = mlr.predict(X_test)
mlr_r2   = r2_score(Y_test, mlr_pred)
mlr_mae  = mean_absolute_error(Y_test, mlr_pred)

print(f"\nR² Score: {mlr_r2:.2f}")
print(f"MAE     : {mlr_mae:.2f} marks off on average")

print(f"\n--- FEATURE IMPORTANCE ---")
print(f"{'Feature':<30} {'Coefficient':>12}")
print("-" * 45)
for feat, coef in sorted(zip(X_all.columns, mlr.coef_),
                          key=lambda x: abs(x[1]), reverse=True):
    bar = "█" * int(abs(coef) * 2)
    print(f"{feat:<30} {coef:>10.3f}  {bar}")

print(f"\nImprovement over SLR:")
print(f"SLR R² = {slr_r2:.2f}  →  MLR R² = {mlr_r2:.2f}  "
      f"(+{(mlr_r2 - slr_r2)*100:.1f}% better)")

# STEP 8 — PLR
print("\n" + "=" * 50)
print("STEP 8: POLYNOMIAL REGRESSION (degree 2)")
print("Does a CURVE fit better than a straight line?")
print("=" * 50)

X_plr_train = X_train[['reading score']]
X_plr_test  = X_test[['reading score']]

poly      = PolynomialFeatures(degree=2)
X_p_train = poly.fit_transform(X_plr_train)
X_p_test  = poly.transform(X_plr_test)

plr = LinearRegression()
plr.fit(X_p_train, Y_train)

plr_pred = plr.predict(X_p_test)
plr_r2   = r2_score(Y_test, plr_pred)
plr_mae  = mean_absolute_error(Y_test, plr_pred)

print(f"\nR² Score: {plr_r2:.2f}")
print(f"MAE     : {plr_mae:.2f} marks off on average")

if plr_r2 > slr_r2:
    print(f"\nPLR is better than SLR — curve fits reading→math better")
else:
    print(f"\nPLR did NOT help here — data is already linear enough")
    print(f"This is normal! Not all data needs a curve.")

# STEP 9 — Compare ALL Models
print("\n" + "=" * 55)
print("       STEP 9: FINAL MODEL COMPARISON")
print("=" * 55)

models = {
    'SLR (reading only)': {'r2': slr_r2, 'mae': slr_mae, 'pred': slr_pred, 'col': '#3B82F6'},
    'MLR (5 features)':    {'r2': mlr_r2, 'mae': mlr_mae, 'pred': mlr_pred, 'col': '#22C55E'},
    'PLR (curve deg 2)':   {'r2': plr_r2, 'mae': plr_mae, 'pred': plr_pred, 'col': '#A855F7'},
}

best_r2 = max(v['r2'] for v in models.values())

print(f"\n{'Model':<22} {'R² Score':>10} {'MAE':>8}   {'Verdict'}")
print("-" * 60)
for name, m in models.items():
    verdict = "✅ BEST" if m['r2'] == best_r2 else ""
    print(f"{name:<22} {m['r2']:>10.2f} {m['mae']:>8.2f}   {verdict}")

best_name = max(models, key=lambda k: models[k]['r2'])
print(f"\n🏆 Best Model: {best_name}  (R² = {best_r2:.2f})")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Comparison — Actual vs Predicted Math Scores",
             fontsize=14, fontweight='bold')

for ax, (name, m) in zip(axes, models.items()):
    ax.scatter(Y_test, m['pred'], alpha=0.7,
               color=m['col'], s=60, edgecolors='white')
    ax.plot([Y_test.min(), Y_test.max()],
            [Y_test.min(), Y_test.max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_title(f"{name}\nR²={m['r2']:.2f}  MAE={m['mae']:.1f}")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# STEP 10 — Predict New Students
print("\n" + "=" * 55)
print("   STEP 10: PREDICT NEW STUDENT MATH SCORES")
print("=" * 55)

new_students = pd.DataFrame({
    'reading score':             [85, 55, 70],
    'writing score':             [88, 50, 72],
    'gender':                    [0,  1,  0],
    'lunch':                     [1,  0,  1],
    'test preparation course':   [1,  0,  1],
})

profiles = [
    "Good student     (high scores, test prep done)",
    "Struggling       (low scores, no test prep)   ",
    "Average student  (mid scores, test prep done) ",
]

predictions = mlr.predict(new_students)

print(f"\n{'Profile':<45} {'Predicted Math':>14}  {'Grade'}")
print("-" * 68)
for profile, pred in zip(profiles, predictions):
    grade = "A" if pred >= 80 else "B" if pred >= 70 else "C" if pred >= 60 else "D"
    print(f"{profile:<45} {pred:>14.1f}  Grade {grade}")

print(f"\n✅ Complete ML Pipeline Done!")
print(f"   Statistics → EDA → Cleaning → SLR → MLR → PLR → Compare → Predict")