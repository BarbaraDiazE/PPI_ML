"""Get models with all metrics values higher than Q2"""
import os
import pandas as pd
import numpy as np

from phase1.support_functions.json_functions import read_id_json
from phase1.support_functions.metrics import metrics


class GetReportValues:
    def __init__(self, local_root):
        self.local_root = local_root

    def get_reports_filenames(self, algorithm: str):
        json_filename = f"{algorithm.upper()}_models_id.json"
        json_root = f"{self.local_root}/results/trained_results/id_models"
        data_dict = read_id_json(json_root, json_filename)
        numerical_id = [k for k in data_dict]
        pk_models = [data_dict[k] for k in data_dict]
        files_list = [pk + ".csv" for pk in pk_models]
        return numerical_id, pk_models, files_list

    def get_report_value(self, metric: str, filename: str) -> float:
        data_root = f"{self.local_root}/results/trained_results/models_reports"
        df = pd.read_csv(os.path.join(data_root, filename), sep=",", index_col="Unnamed: 0")
        value = float(df.loc[metric].value)
        return value

    @classmethod
    def get_numerical_id(cls, filename, algorithm) -> str:
        """get equivalence filename and numerical_id"""
        json_root = f"{local_root}/results/trained_results/id_models"
        json_filename = f"{algorithm.upper()}_models_id.json"
        data_dict = read_id_json(json_root, json_filename)
        for numerical_id, filename_ in data_dict.items():
            if filename_ == filename.replace(".csv", ""):
                return numerical_id
            else:
                continue


class ModelFiltering(GetReportValues):
    def __init__(self, algorithm: str, quartile):
        self.algorithm = algorithm
        self.quartile = quartile
        super().__init__(local_root)

    def get_q3(self, metric):
        df_filename = f"{metric.lower()}_{self.algorithm}_summary.csv"
        metrics_root = f"{local_root}/results/metrics_results/metrics_reports"
        df = pd.read_csv(os.path.join(metrics_root, df_filename), index_col="Unnamed: 0")
        df[metric] = [float(i) for i in list(df["Value"])]
        stat_df = df.describe()
        q3 = stat_df.loc["75%", metric]
        return q3

    def get_q2(self, metric):
        df_filename = f"{metric.lower()}_{self.algorithm}_summary.csv"
        metrics_root = f"{local_root}/results/metrics_results/metrics_reports"
        df = pd.read_csv(os.path.join(metrics_root, df_filename), index_col="Unnamed: 0")
        df[metric] = [float(i) for i in list(df["Value"])]
        stat_df = df.describe()
        q2 = stat_df.loc["50%", metric]
        return q2

    def get_best_metrics_models(self, metric):
        """Those with metrics higher than q3 for a single metric"""
        numerical_id, pk_models, files_list = self.get_reports_filenames(self.algorithm)
        q = str()
        if self.quartile == "Q2":
            q = self.get_q2(metric)
        elif self.quartile == "Q3":
            q = self.get_q3(metric)
        best_models = dict()
        for _ in files_list:
            _val = self.get_report_value(metric, _)
            if _val >= q:
                numerical_id_ = self.get_numerical_id(_, self.algorithm)
                best_models[numerical_id_] = _val
            else:
                continue
        return best_models, numerical_id

    def get_best_models(self, metrics):
        """those present in all metrics filtered list"""
        numerical_id = list()
        precision_bm = balanced_accuracy_bm = accuracy_bm = f1_bm = recall_bm = list()
        for metric in metrics:
            if metric.lower() == "precision":
                precision_bm, numerical_id = self.get_best_metrics_models(metric)
            if metric.lower() == "balanced accuracy":
                balanced_accuracy_bm, numerical_id = self.get_best_metrics_models(metric)
            if metric.lower() == "accuracy":
                accuracy_bm, numerical_id = self.get_best_metrics_models(metric)
            if metric.lower() == "f1":
                f1_bm, numerical_id = self.get_best_metrics_models(metric)
            if metric.lower() == "recall":
                recall_bm, numerical_id = self.get_best_metrics_models(metric)
        best_models_names = [
            i
            for i in numerical_id
            if i in precision_bm
            and i in balanced_accuracy_bm
            and i in accuracy_bm
            and i in f1_bm
            and i in recall_bm
        ]
        print(
            "\n",
            f"total models      : {len(numerical_id)}",
            "\n",
            f"best models       : {len(best_models_names)}",
            "\n",
            f"precision         : {len(precision_bm)}",
            "\n",
            f"balanced accuracy : {len(balanced_accuracy_bm)}",
            "\n",
            f"accuracy          : {len(accuracy_bm)}",
            "\n",
            f"f1                : {len(f1_bm)}",
            "\n",
            f"recall            : {len(recall_bm)}",
            "\n",
        )
        best_models_ = np.array([best_models_names])
        best_models_df = pd.DataFrame(
            data=best_models_.transpose(),
            columns=[
                "FK model",
            ],
        )
        output_filename = f"{self.algorithm}_{self.quartile}_best_models.csv"
        print(f"output_filename: {output_filename}")
        output_root = f"{self.local_root}/results/metrics_results/filtered_results"
        best_models_df.to_csv(os.path.join(output_root, output_filename))
        print("best models: ", "\n", best_models_df)


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    local_root = local_root["phase1"]
    algorithms = ["svm", "lrg", "rf", "dt"]
    for algorithm_ in algorithms:
        ModelFiltering(algorithm_, "Q2").get_best_models(metrics)
        ModelFiltering(algorithm_, "Q3").get_best_models(metrics)
