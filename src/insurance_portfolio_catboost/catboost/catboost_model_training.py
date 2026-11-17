import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from pathlib import Path
from data import DATA_FOLDER_PATH


PATH_CURRENT_FILE = Path(__file__).resolve().parent

df = pd.read_csv(Path(DATA_FOLDER_PATH) / "data_cleaned.csv", delimiter=";")
df = df.drop(["contrat_connue", "Date_last_renewal", "ID", "connu"], axis=1)


X = df.drop("Premium", axis=1)
y = df["Premium"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cat_features = [
    "Distribution_channel",
    "Payment",
    "Type_risk",
    "Area",
    "N_doors",
    "Second_driver",
    "diesel",
    "plomb",
    "nombre d'année ",
]
model = CatBoostRegressor(
    loss_function="RMSE",
    cat_features=cat_features,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_state=42,
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=100,
    verbose=100,
)

model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

model_path = Path(PATH_CURRENT_FILE) / "catboost_premium.cbm"
model.save_model(str(model_path))
