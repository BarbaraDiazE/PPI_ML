"""heatmap plot of best models"""

import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from phase1.support_functions.json_functions import read_id_json


class PlotHeatmap:
    def __init__(self, algorithm: str, local_root: str, quartile: str):
        self.algorithm = algorithm
        self.local_root = local_root
        self.metrics = ["Accuracy", "Balanced Accuracy", "Precision", "F1", "Recall"]
        self.quartile = quartile

    def get_filtered_models(self, quartile):
        df_root = f"{local_root}/results/metrics_results/filtered_results"
        filename = f"{self.algorithm.lower()}_{quartile}_best_models.csv"
        df_models = pd.read_csv(os.path.join(df_root, filename), index_col="Unnamed: 0")
        return df_models["FK model"].to_list()

    def get_reports_filenames(self, filtered_models):
        json_filename = f"{self.algorithm.upper()}_models_id.json"
        json_root = f"{self.local_root}/results/trained_results/id_models"
        data_dict = read_id_json(json_root, json_filename)
        filtered_data_dict = {k: v for (k, v) in data_dict.items() if k in filtered_models}
        print(filtered_data_dict)
        numerical_id = [k for k in filtered_data_dict]
        numerical_id = [k + " " if len(numerical_id[:0]) != len(k) else k for k in numerical_id]
        pk_models = [filtered_data_dict[k] for k in filtered_data_dict]
        files_list = [pk + ".csv" for pk in pk_models]
        return numerical_id, pk_models, files_list

    def get_report_value(self, filename: str) -> list:
        data_root = f"{self.local_root}/results/trained_results/models_reports"
        df = pd.read_csv(os.path.join(data_root, filename), sep=",", index_col="Unnamed: 0")
        model_metrics_values = [float(df.loc[metric].value) for metric in self.metrics]
        return model_metrics_values

    def build_df(self, quartile):
        filtered_models = self.get_filtered_models(quartile)
        numerical_id, pk_models, files_list = self.get_reports_filenames(filtered_models)
        values_list = [self.get_report_value(i) for i in files_list]
        data = np.array(values_list)
        df = pd.DataFrame(data=data, columns=self.metrics, index=numerical_id)
        print(df)
        return df, numerical_id

    def plot(self):
        df, numerical_id = self.build_df(self.quartile)
        plt.figure(figsize=[12, 16])
        plt.subplot()
        cmap = "YlGnBu"
        sns.set_context("paper")
        sns.set_style("darkgrid")
        ax = sns.heatmap(
            data=df,
            cmap=cmap,
            linewidth=1,
            annot=True,
            linecolor="ivory",
            vmin=0.90,
            vmax=1.0,
            cbar_kws={"shrink": 0.8},
            yticklabels=True,
        )
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=12)
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0.1)
        plt.xlabel("Metrics", fontsize=14)
        plt.ylabel("Trained models", fontsize=14)
        plt.yticks(rotation=0)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)
        plt.tight_layout(h_pad=0.9)
        output_root = f"{self.local_root}/results/metrics_results/metrics_plots"
        output_figure = f"heatmap_{self.algorithm}_{self.quartile}.tiff"
        # output_figure = f"heatmap_{self.algorithm}_{self.quartile}.png"
        plt.savefig(
            os.path.join(output_root, output_figure),
            # dpi=200,
            transparent=False,
        )
        plt.show()
        return f"{self.algorithm} heatmap is ready"


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    algorithms = ["svm", "lrg", "rf"]
    local_root = local_root["phase1"]
    # Q2
    for algorithm_ in algorithms:
        p = PlotHeatmap(algorithm_, local_root, "Q2")
        p.plot()
    # Q3
    # for algorithm in algorithms:
    #     p = PlotHeatmap(algorithm, local_root, "Q3")
    #     p.plot()
