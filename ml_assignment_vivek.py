# ============================================================
# FSP Assignment - ML with Python
# Name: Vivek Kumar Shaw | Roll No: 13005324027
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# ─────────────────────────────────────────────
# Q1. Handling Missing Values
# ─────────────────────────────────────────────
print("=" * 50)
print("Q1. Handling Missing Values")
print("=" * 50)

data = {
    'Name':   ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age':    [25, np.nan, 30, np.nan, 22],
    'Salary': [50000, 60000, np.nan, 45000, np.nan]
}
df = pd.DataFrame(data)
print("\nOriginal DataFrame:")
print(df)

# Detect missing values
print("\nMissing Values Detected:")
print(df.isnull())
print("\nMissing Value Count per Column:")
print(df.isnull().sum())

# Replace missing values with mean
df_mean = df.copy()
df_mean['Age'] = df_mean['Age'].fillna(df_mean['Age'].mean())
df_mean['Salary'] = df_mean['Salary'].fillna(df_mean['Salary'].mean())
print("\nAfter Replacing Missing Values with Mean:")
print(df_mean)

# Replace missing values with median
df_median = df.copy()
df_median['Age'] = df_median['Age'].fillna(df_median['Age'].median())
df_median['Salary'] = df_median['Salary'].fillna(df_median['Salary'].median())
print("\nAfter Replacing Missing Values with Median:")
print(df_median)

# Drop rows with missing values
df_dropped = df.dropna()
print("\nAfter Dropping Rows with Missing Values:")
print(df_dropped)


# ─────────────────────────────────────────────
# Q2. Handling Duplicate Rows
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Q2. Handling Duplicate Rows")
print("=" * 50)

data2 = {
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'Age':  [25, 30, 25, 35, 30],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai']
}
df2 = pd.DataFrame(data2)
print("\nOriginal DataFrame:")
print(df2)

print("\nDuplicate Rows:")
print(df2[df2.duplicated()])
print("\nDuplicate Flags (True = duplicate):")
print(df2.duplicated())

df2_clean = df2.drop_duplicates()
print("\nAfter Removing Duplicates:")
print(df2_clean)


# ─────────────────────────────────────────────
# Q3. Min-Max Normalization
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Q3. Min-Max Normalization")
print("=" * 50)

values = [12, 25, 40, 55, 70]
x_min = min(values)
x_max = max(values)
normalized = [(x - x_min) / (x_max - x_min) for x in values]

print(f"\nOriginal Values  : {values}")
print(f"Min = {x_min}, Max = {x_max}")
print(f"Normalized Values: {[round(v, 4) for v in normalized]}")


# ─────────────────────────────────────────────
# Q4. Z-Score Standardization
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Q4. Z-Score Standardization")
print("=" * 50)

dataset = [10, 20, 30, 40, 50, 60, 70]
mean = np.mean(dataset)
std  = np.std(dataset)
z_scores = [(x - mean) / std for x in dataset]

print(f"\nDataset       : {dataset}")
print(f"Mean          : {mean}")
print(f"Std Deviation : {round(std, 4)}")
print(f"Z-Scores      : {[round(float(z), 4) for z in z_scores]}")


# ─────────────────────────────────────────────
# Q5. Outlier Detection using Z-Score
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Q5. Outlier Detection using Z-Score")
print("=" * 50)

data5 = [10, 12, 11, 14, 13, 100, 12, 11, 10, 200]
mean5 = np.mean(data5)
std5  = np.std(data5)

print(f"\nDataset : {data5}")
print(f"Mean    : {round(mean5, 2)}, Std Dev: {round(std5, 2)}")
print("\nZ-Scores and Outlier Detection (threshold = 2):")

outliers = []
for x in data5:
    z = (x - mean5) / std5
    flag = "<- OUTLIER" if abs(z) > 2 else ""
    print(f"  Value: {x:>4}  |  Z-Score: {round(float(z), 4):>8}  {flag}")
    if abs(z) > 2:
        outliers.append(x)

print(f"\nOutliers Found: {outliers}")


# ─────────────────────────────────────────────
# Q6. Label Encoding
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Q6. Label Encoding")
print("=" * 50)

colors = ["Red", "Blue", "Green", "Blue"]
print(f"\nOriginal Categorical Data : {colors}")

le = LabelEncoder()
encoded = le.fit_transform(colors)
classes = list(le.classes_)

print(f"Label Encoded Values      : {list(encoded)}")
print(f"Classes (mapping)         : {classes} -> {list(range(len(classes)))}")


# ─────────────────────────────────────────────
# Q7. One-Hot Encoding
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Q7. One-Hot Encoding")
print("=" * 50)

df7 = pd.DataFrame({'City': ['Kolkata', 'Delhi', 'Mumbai', 'Chennai']})
print("\nOriginal DataFrame:")
print(df7)

df7_encoded = pd.get_dummies(df7, columns=['City'])
print("\nAfter One-Hot Encoding:")
print(df7_encoded)


# ─────────────────────────────────────────────
# Q8. Feature Scaling using StandardScaler
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Q8. Feature Scaling using StandardScaler")
print("=" * 50)

data8 = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [50, 60, 70, 80, 90],
    'Age':    [20, 25, 30, 35, 40]
}
df8 = pd.DataFrame(data8)
print("\nOriginal Dataset:")
print(df8)

scaler = StandardScaler()
df8_scaled = pd.DataFrame(scaler.fit_transform(df8), columns=df8.columns)
print("\nAfter StandardScaler Scaling:")
print(df8_scaled.round(4))


# ─────────────────────────────────────────────
# Q9. Min-Max Scaling vs Standardization
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Q9. Min-Max Scaling vs Standardization")
print("=" * 50)

data9 = {'Feature': [10, 20, 30, 40, 50]}
df9 = pd.DataFrame(data9)
print("\nOriginal Dataset:")
print(df9)

minmax = MinMaxScaler()
df9_minmax = pd.DataFrame(minmax.fit_transform(df9), columns=['MinMax_Scaled'])

standard = StandardScaler()
df9_standard = pd.DataFrame(standard.fit_transform(df9), columns=['Z_Score_Standardized'])

result9 = pd.concat([df9, df9_minmax, df9_standard], axis=1)
print("\nComparison of Scaling Methods:")
print(result9.round(4))


# ─────────────────────────────────────────────
# Q10. Converting Numeric Strings to Integer
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Q10. Converting Numeric Strings to Integer")
print("=" * 50)

df10 = pd.DataFrame({'Score': ['85', '90', '78', '92', '88']})
print("\nOriginal DataFrame:")
print(df10)
print(f"Data type before conversion: {df10['Score'].dtype}")

df10['Score'] = df10['Score'].astype(int)
print("\nAfter Converting to Integer:")
print(df10)
print(f"Data type after conversion : {df10['Score'].dtype}")
