import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, cross_val_score

from phase1.support_functions.validation.utils import models_to_validate


class Validation:
    def __init__(self, root_data: str, data_filename: str, target: str, fraction: float):
        self.data = pd.read_csv(
            os.path.join(root_data, data_filename.lower()), index_col="Unnamed: 0"
        )
        self.root_data = root_data
        ids = [
            "ipp_id",
            "chembl_id",
            "SMILES",
            "library",
            "PPI family",
            "PPI",
        ]
        self.numerical_data = np.array(self.data.drop(ids, axis=1))
        self.target = target
        self.fraction = fraction

    def evaluate_model(self, root: str, model_filename: str, cv: int):
        """compute accuracy with kfold"""
        y = np.array(self.data[self.target])
        y = label_binarize(y, classes=["No", "Yes"])
        y = np.reshape(y, int(y.shape[0]))
        x_train, x_test, y_train, y_test = train_test_split(
            self.numerical_data, y, test_size=self.fraction, random_state=1992
        )
        model = joblib.load(
            os.path.join(root, "results", "trained_results", "trained_models", model_filename)
        )
        accuracies = cross_val_score(estimator=model, X=x_train, y=y_train, cv=cv)
        output_data = {
            "accuracy mean": float(accuracies.mean()),
            "accuracy std": float(accuracies.std()),
        }
        output_data = pd.DataFrame.from_dict(output_data, orient="index")
        output_filename = f'k{cv}_validation_{model_filename.replace(".pkl", ".csv")}'
        output_data.to_csv(os.path.join(root, "results", "validation_results", output_filename))
        print(f'{model_filename.replace(".pkl", "")} has been validated')


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    models_to_validate = models_to_validate(
        root=local_root["phase1"], algorithm="rf", quartile="Q2"
    )
    for i in range(len(models_to_validate)):
        input_filename = f"dataset_{models_to_validate[i][0]}.csv"
        model_name = models_to_validate[i][1]
        A = Validation(
            root_data=local_root["data"],
            data_filename=input_filename,
            target="PPI",
            fraction=0.2,
        )
        A.evaluate_model(root=local_root["phase1"], model_filename=model_name, cv=20)
