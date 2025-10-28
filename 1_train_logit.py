# train_logit.py
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

ARTIFACT_DIR = Path("model_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1) Wczytaj dane
    data = pd.read_csv("generated_dataset.csv")

    # (opcjonalnie) utrzymujemy te same kroki przygotowania co wcześniej
    data["sector"] = data["sector"].astype("string")
    data["key_target"] = data.groupby("id")["credit_limit"].transform("mean")

    # 2) Definicja cech/targetu – tak, by zgadzało się z tym, co podasz później do modelu
    drop_cols = ["def_ind_1m", "def_ind_2m", "def_ind_3m", "date", "sector", "id"]
    X_train = data.drop(columns=drop_cols, axis=1)
    X_train = X_train.fillna(0)  # logit nie lubi NaN-ów
    # X_train.isna().sum().sum()  # sprawdzenie liczby NaN-ów

    y_train = data["def_ind_1m"].astype(int)  # logit wymaga b
    # 3) Prosty logit (bez strojenia) – jako „szkielet”
    # class_weight="balanced" zmniejsza wrażliwość na ewentualny niezbalansowany target
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",  # stabilny dla binarnego problemu i małych/średnich zbiorów
    )
    model.fit(X_train, y_train)

    # 4) Zapis modelu i schematu cech
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    feature_list = list(X_train.columns)
    (ARTIFACT_DIR / "feature_list.json").write_text(json.dumps(feature_list, indent=2))

    print("Saved:", ARTIFACT_DIR.resolve())


if __name__ == "__main__":
    main()
