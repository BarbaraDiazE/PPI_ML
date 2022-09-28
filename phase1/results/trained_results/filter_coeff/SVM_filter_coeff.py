import pandas as pd
from phase1.support_functions.vars import local_root

local_root_ = local_root["phase1"]
root_ = f"{local_root_}/results/trained_results/models_reports/coeff_reports/"

filename = "SVMF2L6P3K1A_coeff.csv"

"""
SVM17
-0.095000 primer cuartile
0.103000 tercer cuartile
"""


def get_cuts(data):
    s_cut = data["Coeff"].max() - data["Coeff"].std()
    l_cut = data["Coeff"].min() + data["Coeff"].std()
    return s_cut, l_cut


if __name__ == "__main__":
    df = pd.read_csv(f"{root_}{filename}", index_col="Unnamed: 0")
    # print(df.head(2))
    print(df["Coeff"].describe())
    # positives = df[df["Coeff"] >= 0.103000]
    s, l = get_cuts(df)
    print(s, l)
    positives = df[df["Coeff"] >= s]
    print(positives.shape[0])

    negatives = df[df["Coeff"] <= l]
    # print(negatives)
    # print(negatives.shape[0])
    final_data = pd.concat([positives, negatives], ignore_index=True)
    print(final_data.head())
    output_filename = "SVM17_+_-std.csv"
    final_data.to_csv(f"{root_}{output_filename}", index=False)
