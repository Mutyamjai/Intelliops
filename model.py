import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.head())
print(train.info())
print(train.describe())

print(train.isnull().sum())

print(train["Transported"].value_counts())
sns.countplot(x="Transported", data=train)
plt.show()
sns.countplot(x="HomePlanet", hue="Transported", data=train)
plt.show()

sns.countplot(x="CryoSleep", hue="Transported", data=train)
plt.show()

sns.boxplot(x="Transported", y="Age", data=train)
plt.show()

train.drop(["PassengerId", "Name"], axis=1, inplace=True)
test_ids = test["PassengerId"]
test.drop(["PassengerId", "Name"], axis=1, inplace=True)

train['Age'] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(train["Age"].median())

Q1 = train["Age"].quantile(0.25)
Q3 = train["Age"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(lower, upper)

train["Age"] = train["Age"].clip(lower = 0, upper = upper)

categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
for col in categorical_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])

expense_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in expense_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

train["Deck"] = train["Cabin"].str[0]
test["Deck"] = test["Cabin"].str[0]

train = train.drop("Cabin", axis=1)
test = test.drop("Cabin", axis=1)

train["Deck"] = train["Deck"].fillna(train["Deck"].mode()[0])
test["Deck"] = test["Deck"].fillna(train["Deck"].mode()[0])

train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

X = train.drop("Transported", axis=1)
y = train["Transported"]

y = y.astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

test_predictions = model.predict(test)

result = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": test_predictions.astype(bool)
})

print(result)