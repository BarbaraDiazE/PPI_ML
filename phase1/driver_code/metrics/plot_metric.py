"""Plot metrics"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

from phase1.support_functions.json_functions import read_id_json
from phase1.support_functions.vars import local_root

colors = {
    "precision": "teal",
    "recall": "deeppink",
    "balanced accuracy": "navy",
    "accuracy": "yellowgreen",
    "f1": "blueviolet",
}


class PlotMetric:
    def __init__(self, metric: str, y_lim: list, algorithm: str, local_root: str):
        self.metric = metric
        self.y_lim = y_lim
        self.algorithm = algorithm
        self.local_root = local_root

    def get_reports_filenames(self):
        json_filename = f"{self.algorithm.upper()}_models_id.json"
        json_root = f"{self.local_root}/results/trained_results/id_models"
        data_dict = read_id_json(json_root, json_filename)
        numerical_id = [k for k in data_dict]
        pk_models = [data_dict[k] for k in data_dict]  # id correnponding to json type
        files_list = [pk + ".csv" for pk in pk_models]
        return numerical_id, pk_models, files_list

    def get_report_value(self, filename: str) -> float:
        data_root = f"{self.local_root}/results/trained_results/models_reports"
        df = pd.read_csv(os.path.join(data_root, filename), sep=",", index_col="Unnamed: 0")
        value = float(df.loc[self.metric].value)
        return value

    def get_values_metric_by_algorithm_type(self) -> DataFrame:
        numerical_id, pk_models, files_list = self.get_reports_filenames()
        metric_list = [self.get_report_value(i) for i in files_list]
        data = np.transpose(np.array([numerical_id, pk_models, metric_list]))
        df = pd.DataFrame(data, columns=["Numerical ID", "Model name", "Value"])
        df_filename = f"{self.metric.lower()}_{self.algorithm}_summary.csv"
        output_root = f"{self.local_root}/results/metrics_results/metrics_reports"
        df.to_csv(os.path.join(output_root, df_filename))
        return df

    def statistical_values(self):
        df = self.get_values_metric_by_algorithm_type()
        df[self.metric] = [float(i) for i in list(df["Value"])]
        stat_df = df.describe()
        output_filename = f"{self.metric.lower()}_{self.algorithm}_stats.csv"
        output_root = f"{local_root}/results/metrics_results/metrics_stats"
        stat_df.to_csv(os.path.join(output_root, output_filename))
        return stat_df

    def plot(self):
        df = self.get_values_metric_by_algorithm_type()
        x = [i for i in range(df.shape[0])]
        x_label = df["Numerical ID"].to_list()
        metric = self.metric.lower()
        y = list(df["Value"])
        y = [float(i) for i in y]
        plt.figure(figsize=[10, 4.8], dpi=200)
        plt.plot(x, y, "ro", color=colors[metric], alpha=0.7)
        plt.grid(color="lightgray", axis="y", linestyle="dotted", linewidth=2)
        plt.xticks(x, x_label, rotation="vertical")
        plt.ylabel(self.metric)
        plt.ylim(self.y_lim)
        plt.subplots_adjust()
        output_filename = f"{metric}_{self.algorithm}_summary.tiff"
        output_root = f"{self.local_root}/results/metrics_results/metrics_plots"
        plt.savefig(os.path.join(output_root, output_filename), dpi=200)
        # plt.show()
        print(f"{self.metric} results for {self.algorithm} models have been processed")


if __name__ == "__main__":
    local_root = local_root["phase1"]
    algorithms = ["svm", "lrg", "rf", "dt"]
    metrics = {
        "Precision": [0.5, 1.0],
        "Balanced Accuracy": [0.5, 1.0],
        "Accuracy": [0.5, 1.0],
        "F1": [0.5, 1.0],
        "Recall": [0.5, 1.0],
    }
    for algorithm in algorithms:
        for k in metrics:
            p = PlotMetric(k, metrics[k], algorithm, local_root)
            p.statistical_values()
            p.plot()
