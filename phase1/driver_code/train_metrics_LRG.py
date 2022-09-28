import os
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from phase1.support_functions.json_functions import read_id_json
from phase1.support_functions.lrg.functions_LRG import lrg_report_train_metrics
import joblib

"""Compute training metrics (from training data)"""


class TrainMetrics:
    def __init__(self, algorithm, local_root, input_file, target, metric):
        self.algorithm = algorithm
        self.local_root = local_root
        input_file = input_file
        self.target = target
        data = pd.read_csv(
            os.path.join(local_root.replace("phase1", "data"), input_file),
            index_col="Unnamed: 0",
        )
        ids = [
            "ipp_id",
            "chembl_id",
            "SMILES",
            "library",
            "PPI family",
            "PPI",
        ]
        self.numerical_data = data.drop(ids, axis=1)
        self.data = data
        self.metric = metric

    def get_reports_filenames(self):
        json_filename = f"{self.algorithm.upper()}_models_id.json"
        json_root = f"{self.local_root}/results/trained_results/id_models"
        data_dict = read_id_json(json_root, json_filename)
        print(data_dict)
        numerical_id = [k for k in data_dict]
        pk_models = [data_dict[k] for k in data_dict]
        files_list = [pk + ".csv" for pk in pk_models]
        return numerical_id, pk_models, files_list

    def get_population(self, fraction):
        y = np.array(self.data[self.target])
        y = label_binarize(y, classes=["No", "Yes"])
        y = np.reshape(y, int(y.shape[0]))
        # numerical_data = np.array(self.data[self.descriptors])
        x_train, x_test, y_train, y_test = train_test_split(
            self.numerical_data, y, test_size=fraction, random_state=1992
        )
        return x_train, x_test, y_train, y_test

    def compute_metrics(self):
        # model_names = self.get_reports_filenames()
        model_names = {"LRG1": "LRGF1L6P3S1A"}
        # TODO revisar los datos de LRG1
        x_train, x_test, y_train, y_test = self.get_population(0.2)
        for key, value in model_names.items():
            model = joblib.load(
                os.path.join(
                    self.local_root,
                    "results",
                    "trained_results",
                    "trained_models",
                    f"{value}.pkl",
                )
            )
            test_predictions = np.array(model.predict(x_test))
            train_predictions = np.array(model.predict(x_train))
            lrg_report_train_metrics(
                output_reference=key,
                y_train=y_train,
                y_test=y_test,
                train_predictions=train_predictions,
                test_predictions=test_predictions,
                local_root=self.local_root,
            )

    def get_report_value(self, filename: str) -> float:
        data_root = f"{self.local_root}/results/trained_results/models_reports"
        df = pd.read_csv(os.path.join(data_root, filename), sep=",", index_col="Unnamed: 0")
        print(df.head())
        return df.loc[self.metric].value

    def get_values(self) -> DataFrame:
        numerical_id, pk_models, files_list = self.get_reports_filenames()
        metric_list = [self.get_report_value(i) for i in files_list]
        data = np.transpose(np.array([numerical_id, pk_models, metric_list]))
        df = pd.DataFrame(data, columns=["Numerical ID", "Model name", "Value"])
        df_filename = f"{self.metric.lower()}_{self.algorithm}_summary.csv"
        output_root = f"{self.local_root}/results/metrics_results/metrics_reports"
        df.to_csv(os.path.join(output_root, df_filename))
        print(df)
        return df


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    local_root = local_root["phase1"]
    # algorithms = ["svm", "lrg", "rf"]
    algorithms = ["lrg"]
    fp = "ECFP6"
    filename = f'{"dataset_"}{fp.lower()}.csv'
    for i in algorithms:
        TM = TrainMetrics(
            algorithm=i,
            local_root=local_root,
            input_file=filename,
            target="PPI",
            metric="racall",
        )
        TM.compute_metrics()
