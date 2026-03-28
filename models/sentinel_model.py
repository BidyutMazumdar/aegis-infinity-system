import pandas as pd

def sentinel_decision(gedix, qev):
    decisions = []

    for g, q in zip(gedix, qev):
        if g > 0.7 and q > 0:
            decisions.append("ESCALATE")
        elif g > 0.5:
            decisions.append("MONITOR")
        else:
            decisions.append("STABLE")

    return decisions


if __name__ == "__main__":
    from models.gedix_model import compute_gedix
    from models.qev_model import compute_qev

    df = pd.read_csv("data/dataset.csv")

    df["GEDI_X"] = compute_gedix(df)
    df["QEV"] = compute_qev(df)

    df["ACTION"] = sentinel_decision(df["GEDI_X"], df["QEV"])

    print(df[["Country", "GEDI_X", "QEV", "ACTION"]])
