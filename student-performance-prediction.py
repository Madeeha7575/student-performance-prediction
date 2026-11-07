import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("StudentsPerformance.csv")

df["avg_score"] = (
    df["math score"]
    + df["reading score"]
    + df["writing score"]
) / 3

print("\nMissing Values:")
print(df.isnull().sum())


subject_avg = df[
    [
        "math score",
        "reading score",
        "writing score"
    ]
].mean()

print("\nAverage Scores:")
print(subject_avg)

plt.figure(figsize=(8, 5))

plt.bar(
    ["Math", "Reading", "Writing"],
    subject_avg,
    color=["skyblue", "lightgreen", "lightcoral"]
)

plt.title("Average Scores by Subject")
plt.ylabel("Average Score")

plt.savefig(
    "average_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

course_impact = df.groupby(
    "test preparation course"
)[
    [
        "math score",
        "reading score",
        "writing score"
    ]
].mean()

print("\nTest Preparation Impact:")
print(course_impact)

course_impact.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Impact of Test Preparation Course")
plt.ylabel("Average Score")

plt.savefig(
    "test_preparation_impact.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# GENDER ANALYSIS

gender_scores = df.groupby(
    "gender"
)[
    [
        "math score",
        "reading score",
        "writing score"
    ]
].mean()

print("\nGender Performance:")
print(gender_scores)

gender_scores.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Average Scores by Gender")
plt.ylabel("Average Score")

plt.savefig(
    "gender_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# OVERALL AVERAGE BY GENDER
overall = df.groupby("gender")["avg_score"].mean()

print("\nOverall Average Score:")
print(overall)

# CORRELATION ANALYSIS

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

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# MACHINE LEARNING MODEL
X = df[
    [
        "math score",
        "reading score"
    ]
]

y = df["writing score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(
    X_train,
    y_train
)
predictions = model.predict(X_test)

# PREDICTIONS
print("\nFirst 10 Predictions:")
print(predictions[:10])

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nActual vs Predicted:")
print(comparison.head(10))


# ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

plt.xlabel("Actual Writing Scores")
plt.ylabel("Predicted Writing Scores")
plt.title("Actual vs Predicted Writing Scores")

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#Model evaluation 
score = r2_score(
    y_test,
    predictions
)

print("\nR² Score:", score)
