import os
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

from phase1.support_functions.json_functions import read_id_json


class ConfusionMatrix:
    def __init__(self, algorithm, root):
        self.metric = "Confusion matrix"
        self.algorithm = algorithm
        self.root = root

    def get_reports_filenames(self):
        json_filename = f"{self.algorithm.upper()}_models_id.json"
        json_root = f"{self.root}/results/trained_results/id_models"
        data_dict = read_id_json(json_root, json_filename)
        numerical_id = [k for k in data_dict]
        pk_models = [data_dict[k] for k in data_dict]
        files_list = [pk + ".csv" for pk in pk_models]
        return numerical_id, pk_models, files_list

    def get_report_value(self, filename: str) -> float:
        data_root = f"{self.root}/results/trained_results/models_reports"
        df = pd.read_csv(os.path.join(data_root, filename), sep=",", index_col="Unnamed: 0")
        return df.loc[self.metric].value

    def get_values(self) -> DataFrame:
        numerical_id, pk_models, files_list = self.get_reports_filenames()
        metric_list = [self.get_report_value(i) for i in files_list]
        data = np.transpose(np.array([numerical_id, pk_models, metric_list]))
        df = pd.DataFrame(data, columns=["Numerical ID", "Model name", "Value"])
        df_filename = f"{self.metric.lower()}_{self.algorithm}_summary.csv"
        output_root = f"{self.root}/results/metrics_results/metrics_reports"
        df.to_csv(os.path.join(output_root, df_filename))
        print(df)
        return df


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    local_root = local_root["phase1"]
    algorithms = ["svm", "lrg", "rf"]
    for i in algorithms:
        ConfusionMatrix(algorithm=i, root=local_root).get_values()
