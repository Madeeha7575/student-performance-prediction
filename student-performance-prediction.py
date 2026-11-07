import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("StudentsPerformance.csv")

# Create Average Score Feature
df["avg_score"] = (
    df["math score"]
    + df["reading score"]
    + df["writing score"]
) / 3

# Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Correlation Analysis
correlation = df[
    [
        "math score",
        "reading score",
        "writing score",
        "avg_score"
    ]
].corr()

print("\nCorrelation Matrix:")
print(correlation)

# Heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# Features and Target
X = df[["math score", "reading score"]]

y = df["writing score"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Training
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(predictions[:10])

# Model Evaluation
score = r2_score(y_test, predictions)

print("\nR² Score:", score)
