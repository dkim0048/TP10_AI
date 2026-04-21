"""Ridge Regression Epic 3"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report
import wordfreq
import joblib

FILENAME = 'AoA_refined.xlsx'

# 1. Show Dataset Distribution
df_raw = pd.read_excel(FILENAME)
print(f'Total rows: {len(df_raw):,}')
print(f'Columns: {list(df_raw.columns)}')
print(df_raw.head())

df_view = df_raw[df_raw['AoA_Kup'].notna()].copy()
print(f'AoA valid samples: {len(df_view):,}')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
# Plot AoA_Kup values
axes[0].hist(df_view['AoA_Kup'], bins=40, color='#2A63E8', alpha=0.8, edgecolor='white')
axes[0].set_title('AoA_Kup Distribution')
axes[0].set_xlabel('AoA (years)')
axes[0].set_ylabel('Count')
# Lines for our targeting group age boundaries
axes[0].axvline(6, color='red', linestyle='--', label='6 yrs')
axes[0].axvline(8, color='orange', linestyle='--', label='8 yrs')
axes[0].legend()

def bin3(v):
    if v < 6:
        return '<6'
    if v < 8:
        return '6-8'
    return '8+'

df_view['bin'] = df_view['AoA_Kup'].apply(bin3)
counts = df_view['bin'].value_counts().sort_index()

# Plot bar chart for 3 groups distribution
axes[1].bar(counts.index, counts.values, color=['#2BA36B', '#E8A93B', '#D94B3B'])
axes[1].set_title('Age group Distribution')
for i, (k, v) in enumerate(counts.items()):
    axes[1].text(i, v + 100, f'{v:,}\n({v/len(df_view)*100:.1f}%)', ha='center')

plt.tight_layout()
plt.show()

# 2. Data Cleaning
df = df_raw.copy()
df['Word'] = df['Word'].astype(str)

# Keep only rows with valid AoA and frequency values
before = len(df)
df = df[df['AoA_Kup'].notna() & df['Freq_pm'].notna()].copy()
print(f'After AoA+Freq filter: {len(df):,} (removed {before - len(df):,})')

# Filter out words known by less than 80% of responser
before = len(df)
df = df[df['Perc_known'] >= 0.8].copy()
print(f'After Perc_known ≥ 0.8: {len(df):,} (removed {before - len(df):,})')

# Remove too low frequency words using threshold
before = len(df)
df = df[df['Freq_pm'] >= 3].copy()
print(f'After Freq_pm ≥ 3: {len(df):,} (removed {before - len(df):,})')

# Remove non-standard lexical items
if 'Dom_PoS_SUBTLEX' in df.columns:
    before = len(df)
    df = df[~df['Dom_PoS_SUBTLEX'].isin(['Name', 'Abbreviation', 'Letter', '#N/A'])].copy()
    print(f'After POS filter: {len(df):,} (removed {before - len(df):,})')

df = df.reset_index(drop=True)
print(f'\nCleaned dataset: {len(df):,} words')

# Visualize the distribution after filtering
plt.figure(figsize=(7, 4))
plt.hist(df['AoA_Kup'].dropna(), bins=40, alpha=0.8, color='#2A63E8', edgecolor='white')
plt.title('Distribution After Filtering')
plt.xlabel('AoA')
plt.ylabel('Count')
plt.axvline(6, color='red', linestyle='--', label='6 yrs')
plt.axvline(8, color='orange', linestyle='--', label='8 yrs')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Feature Engineering based on paper
def get_wf_features(word):
    # Get external corpus-based frequency features
    w = str(word).lower().strip()
    return {
        'log_freq_wf': np.log1p(wordfreq.word_frequency(w, 'en') * 1e6),
        'zipf_score': wordfreq.zipf_frequency(w, 'en'),
    }
# Extract wordfreq-based features for all words
wf = df['Word'].apply(get_wf_features).apply(pd.Series)
# Merge extracted features into the main dataframe
df = pd.concat([df, wf], axis=1)
# Create log-transformed frequency feature from dataset frequency
df['log_freq_kup'] = np.log1p(df['Freq_pm'])
# Define final feature set
FEATURES = [
    'Nletters',
    'Nsyll',
    'Nphon',
    'log_freq_kup',
    'log_freq_wf',
    'zipf_score'
]
# Remove rows with missing feature values or missing target values
df = df.dropna(subset=FEATURES + ['AoA_Kup']).reset_index(drop=True)
# Print final dataset size
print(f'Final dataset: {len(df):,} words with {len(FEATURES)} features')
# Preview final features and target
print(df[FEATURES + ['AoA_Kup']].head())
    
# 4. Train / Validation / Test split
# Build the feature matrix and target vector
X = df[FEATURES].values.astype(float)
y = df['AoA_Kup'].values.astype(float)
# Split the full dataset into train / validation and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Split the train / validation set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)
# Print the size of each split
print(f'Train: {len(X_train):,}')
print(f'Validation: {len(X_val):,}')
print(f'Test: {len(X_test):,}')

# 5. Standardization
# Define a simple custom standard scaler class
class MyStandardScaler:
    # Initialize placeholders for mean and scale values
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    # Learn the mean and standard deviation from the training data
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # Prevent division by zero for constant features
        self.scale_[self.scale_ == 0] = 1.0
        return self

    # Apply standardization using the stored mean and scale
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    # Fit the scaler and immediately transform the same data
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# 6. Ridge Regression
# Ridge Regression model with L2 regularization
class MyRidgeRegressor:
    # Initialize the regularization strength and model parameters
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    # Fit the Ridge model using the closed-form solution
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        # Get the number of samples and features
        n_samples, n_features = X.shape
        # Compute the mean of X and y
        X_mean = X.mean(axis=0)
        y_mean = y.mean()

        # Center X and y so the intercept can be handled separately
        X_centered = X - X_mean
        y_centered = y - y_mean
        # Create an identity matrix for L2 regularization
        I = np.eye(n_features)

        # Solve the Ridge normal equation to get coefficients
        self.coef_ = np.linalg.solve(
            X_centered.T @ X_centered + self.alpha * I,
            X_centered.T @ y_centered
        )
        # Recover the intercept after fitting centered data
        self.intercept_ = y_mean - X_mean @ self.coef_
        return self

    # Predict target values for new input data
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

# 7. Scale data
scaler = MyStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 8. Retrain
# Combine the training and validation sets for final training
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

# Refit the scaler on the combined data
scaler = MyStandardScaler()
# Standardize the combined training data
X_trainval_scaled = scaler.fit_transform(X_trainval)
X_test_scaled = scaler.transform(X_test)
ridge = MyRidgeRegressor(alpha=0.01)

# Train the final model on the combined dataset
ridge.fit(X_trainval_scaled, y_trainval)
# Predict AoA values for the test set
y_pred = ridge.predict(X_test_scaled)
# Print final regression performance on the test set
print('Test performance:')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.3f} years')
print(f'R² : {r2_score(y_test, y_pred):.3f}')
print('\nLearned coefficients:')
for f, c in sorted(zip(FEATURES, ridge.coef_), key=lambda x: -abs(x[1])):
    print(f'  {f:<22} {c:+.4f}')
# Print the learned intercept term
print(f'  intercept              {ridge.intercept_:+.4f}')

# 9. Evaluation
# Function to convert continuous AoA values to 3 categories
def to_bin(v):
    if v < 6:
        return 0
    if v < 8:
        return 1
    return 2  # 8+ class
labels = ['<6', '6-8', '8+']

# Evaluate classification
def eval_3bin(y_true, y_pred, title=''):
    # True labels and predicted labels are converted to 3 classes
    y_t = np.array([to_bin(v) for v in y_true])
    y_p = np.array([to_bin(v) for v in y_pred])

    # Calculate the accuracy of classification
    acc = (y_t == y_p).mean()
    # Calculate confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2])
    print(f'{title}')
    print(f'Overall Accuracy: {acc*100:.1f}%')
    print(classification_report(y_t, y_p, target_names=labels, zero_division=0))
    return cm  # Return confusion matrix

# Call the evaluation function and display confusion matrix
cm = eval_3bin(y_test, y_pred, f'Ridge Regressor')
# Plot confusion matrix as heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels, yticklabels=labels, cbar=False
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# 10. Feature importance
# Print all feature coefficients
print('All feature coefficients:')
for f, c in zip(FEATURES, ridge.coef_):
    print(f'  {f:<15} {c:+.4f}')

# Print all feature importance values using absolute coefficients
imp = pd.Series(np.abs(ridge.coef_), index=FEATURES).sort_values(ascending=False)
print('\nAll feature importance (absolute coefficients):')
for f, v in imp.items():
    print(f'  {f:<15} {v:.4f}')

# Plot feature importance as a horizontal bar chart
plt.figure(figsize=(9, 5))
imp.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Coefficient')
plt.tight_layout()
plt.show()