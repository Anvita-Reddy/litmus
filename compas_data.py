import pandas as pd

URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"


def load_compas():
    df = pd.read_csv(URL)

    # standard ProPublica filtering
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ]
    df = df[df["race"].isin(["African-American", "Caucasian"])]

    X = pd.DataFrame({
        "age": df["age"],
        "sex": (df["sex"] == "Male").astype(int),
        "race": (df["race"] == "African-American").astype(int),
        "priors_count": df["priors_count"],
        "juv_fel_count": df["juv_fel_count"],
        "juv_misd_count": df["juv_misd_count"],
        "charge_degree_felony": (df["c_charge_degree"] == "F").astype(int),
    }).reset_index(drop=True)
    y = df["two_year_recid"].astype(int).reset_index(drop=True)
    return X, y