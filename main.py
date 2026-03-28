import pandas as pd

from models.gedix_model import compute_gedix
from models.qev_model import compute_qev
from models.sentinel_model import sentinel_decision


def run_system():
    df = pd.read_csv("data/dataset.csv")

    df["GEDI_X"] = compute_gedix(df)
    df["QEV"] = compute_qev(df)
    df["ACTION"] = sentinel_decision(df["GEDI_X"], df["QEV"])

    return df


if __name__ == "__main__":
    result = run_system()

    print("\n=== A.E.G.I.S.-∞ SYSTEM OUTPUT ===\n")
    print(result[["Country", "GEDI_X", "QEV", "ACTION"]])
