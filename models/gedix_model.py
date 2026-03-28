import pandas as pd
import numpy as np

def compute_gedix(df):
    # Example weights
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Select some columns (adjust as needed)
    X = df[['MC1', 'MC2', 'EP1', 'EP3', 'CC1']].values

    # Normalize
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # GEDI-X calculation
    gedix = np.dot(X, weights)

    return gedix


if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    df["GEDI_X"] = compute_gedix(df)
    print(df[["Country", "GEDI_X"]])
