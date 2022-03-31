import os
import pandas as pd
from phase1.support_functions.vars import local_root


def get_numerical_descriptors(data_filename):
    data = pd.read_csv(os.path.join(local_root["data"], data_filename), index_col="Unnamed: 0")
    ids = ["ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"]
    numerical_data = data.drop(ids, axis=1)
    descriptors_list = numerical_data.columns.to_list()
    return descriptors_list
