import seaborn as sns
import pandas as pd

# Load Penguins dataset
penguins = sns.load_dataset("penguins")
print(penguins.head())

from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
df = df.dropna()

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
