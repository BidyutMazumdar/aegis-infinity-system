import pandas as pd
import numpy as np

def compute_qev(df):
    # Select same features
    X = df[['MC1', 'MC2', 'EP1', 'EP3', 'CC1']].values

    # Compute change (gradient approximation)
    qev = np.gradient(X, axis=0)

    # Convert to score
    qev_score = np.mean(qev, axis=1)

    return qev_score


if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    df["QEV"] = compute_qev(df)
    print(df[["Country", "QEV"]])
