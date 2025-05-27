import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb 
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Load penguins dataset from seaborn
import seaborn as sns
penguins = sns.load_dataset("penguins")

# Drop rows with missing values
penguins = penguins.dropna()

# Select features and target
X = penguins.drop(columns=["species"])
y = penguins["species"]

# Convert categorical features to numeric (like 'sex' and 'island')
X = pd.get_dummies(X, drop_first=True)

# Encode target labels (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Cre

from xgboost import XGBClassifier
model = XGBClassifier()  # No need for use_label_encoder=False anymore

model.fit(X_train,y_train)


r=  model.score(X_test,y_test)

print("model sentiment",r)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)


plt.figure(figsize=(8,4))
sns.heatmap(cm,annot=True)

plt.show()






