import os
import pandas as pd
import numpy as np

from phase1.support_functions.json_functions import read_id_json


class Metrics:
    def __init__(self, algorithm: str, local_root: str, quartile: str):
        self.algorithm = algorithm
        self.local_root = local_root
        self.metrics = [
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "F1",
            "Recall",
            "Confusion matrix",
        ]
        self.quartile = quartile

    def get_filtered_models(self, quartile):
        df_root = f"{self.local_root}/results/metrics_results/filtered_results"
        filename = f"{self.algorithm.lower()}_{quartile}_best_models.csv"
        df_models = pd.read_csv(os.path.join(df_root, filename), index_col="Unnamed: 0")
        return df_models["PK model"].to_list()

    def get_reports_filenames(self):
        json_filename = f"{self.algorithm.upper()}_models_id.json"
        json_root = f"{self.local_root}/results/trained_results/id_models"
        data_dict = read_id_json(json_root, json_filename)
        numerical_id = [k for k in data_dict]
        pk_models = [data_dict[k] for k in data_dict]
        files_list = [pk + ".csv" for pk in pk_models]
        return numerical_id, pk_models, files_list

    def get_report_value(self, filename: str) -> list:
        data_root = f"{self.local_root}/results/trained_results/models_reports"
        df = pd.read_csv(os.path.join(data_root, filename), sep=",", index_col="Unnamed: 0")
        model_metrics_values = [df.loc[metric].value for metric in self.metrics]
        return model_metrics_values

    def build_df(self):
        numerical_id, pk_models, files_list = self.get_reports_filenames()
        values_list = [self.get_report_value(i) for i in files_list]
        data = np.array(values_list)
        df = pd.DataFrame(data=data, columns=self.metrics, index=numerical_id)
        print(df)
        o_root = f"{self.local_root}/results/metrics_results/metrics_reports"
        filename = f"metrics_{self.algorithm}_summary.csv"
        df.to_csv(os.path.join(o_root, filename))


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    algorithms = ["svm", "lrg", "rf", "dt"]
    local_root = local_root["phase1"]
    for algorithm_ in algorithms:
        p = Metrics(algorithm_, local_root, "Q2")
        p.build_df()
