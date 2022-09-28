import os
import pandas as pd
import numpy as np

from phase1.support_functions.json_functions import read_id_json


def get_key(val, dict_name):
    for key, value in dict_name.items():
        if val == value:
            return key
    return "key doesn't exist"


class SummaryValidation:
    def __init__(self, algorithm: str, local_root: str):
        self.algorithm = algorithm
        self.json_root = f"{local_root}/results/trained_results/id_models"
        self.validation_data_root = f"{local_root}/results/validation_results"

    def get_numerical_id(self, model_names):
        data_dict = read_id_json(self.json_root, f"{self.algorithm.upper()}_models_id.json")
        numerical_id = [get_key(name, data_dict) for name in model_names]
        return numerical_id

    def get_filenames(self):
        filenames = os.listdir(self.validation_data_root)
        filenames = [file for file in filenames if self.algorithm.upper() in file]
        return filenames

    def get_value_from_df(self, filenames) -> list:
        accuracy_mean_values = list()
        accuracy_std_values = list()
        for filename in filenames:
            df = pd.read_csv(
                os.path.join(self.validation_data_root, filename),
                sep=",",
                index_col="Unnamed: 0",
            )
            accuracy_mean_values.append(df.loc["accuracy mean"][0])
            accuracy_std_values.append(df.loc["accuracy std"][0])
        return accuracy_mean_values, accuracy_std_values

    def build_df(self):
        from natsort import index_natsorted

        filenames = self.get_filenames()
        accuracy_mean_values, accuracy_std_values = self.get_value_from_df(filenames)
        filenames_ = [filename.replace("k20_validation_", "") for filename in filenames]
        model_names = [filename.replace(".csv", "") for filename in filenames_]
        numerical_id = self.get_numerical_id(model_names)
        data = {
            "model": numerical_id,
            "accuracy mean": accuracy_mean_values,
            "accuracy std": accuracy_std_values,
        }
        df = pd.DataFrame.from_dict(data=data).round(3)
        df = df.sort_values("model", key=lambda x: np.argsort(index_natsorted(df["model"])))
        print(df)
        df_filename = f"summary_{self.algorithm}_validation.csv"
        df.to_csv(os.path.join(self.validation_data_root, df_filename))


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    local_root_ = local_root["phase1"]
    algorithms = ["svm", "lrg", "rf"]
    for algorithm_ in algorithms:
        p = SummaryValidation(algorithm_, local_root_)
        p.build_df()
