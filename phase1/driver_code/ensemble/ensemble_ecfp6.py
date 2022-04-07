import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from phase1.support_functions.ensemble.functions_ensemble import ensemble_report
from phase1.support_functions.ensemble.functions_ensemble import save_ensemble


class Ensemble:
    def __init__(
        self,
        data_root: str,
        root: str,
        input_file: str,
        target: str,
        descriptors: list,
        fraction: float,
    ):
        data = pd.read_csv(os.path.join(data_root, input_file), index_col="Unnamed: 0")
        self.root_data = data_root
        ids = [
            "ipp_id",
            "chembl_id",
            "SMILES",
            "library",
            "PPI family",
            "PPI",
        ]
        print(
            "\n",
            "PPI types: ",
            data.PPI.unique(),
            "\n",
            "Libraries types: ",
            data.library.unique(),
            "\n",
            "Total compounds number: ",
            data.shape[0],
        )
        self.root = root
        self.target = target
        self.descriptors = descriptors
        self.fraction = fraction
        self.numerical_data = data.drop(ids, axis=1)
        self.data = data

    def train_model(self):
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        models = [
            (
                "rf27",
                RandomForestClassifier(
                    n_estimators=500,
                    criterion="entropy",
                    class_weight="balanced",
                    random_state=1992,
                ),
            ),
            (
                "lr22",
                LogisticRegression(
                    solver="newton-cg", fit_intercept=True, class_weight=None, random_state=1992
                ),
            ),
            (
                "lr24",
                LogisticRegression(
                    solver="lbfgs", fit_intercept=True, class_weight=None, random_state=1992
                ),
            ),
            (
                "lr27",
                LogisticRegression(
                    solver="saga", fit_intercept=True, class_weight=None, random_state=1992
                ),
            ),
            (
                "svm22",
                SVC(
                    kernel="rbf",
                    class_weight=None,
                    random_state=1992,
                    probability=True,
                ),
            ),
        ]
        ensemble = VotingClassifier(estimators=models, voting="soft")
        y = np.array(self.data[self.target])
        print(y)
        y = label_binarize(y, classes=["No", "Yes"])
        y = np.reshape(y, int(y.shape[0]))
        numerical_data = np.array(self.data[self.descriptors])
        x_train, x_test, y_train, y_test = train_test_split(
            numerical_data, y, test_size=self.fraction, random_state=1992
        )
        return ensemble.fit(x_train, y_train), x_test, y_test

    @classmethod
    def get_predictions(cls, model, x_test, y_test):
        prediction_data = {
            "predictions": model.predict(x_test),
            "x_text": x_test,
            "y_test": y_test,
        }
        return prediction_data

    def report(self, output_reference: str):
        ensemble, x_test, y_test = self.train_model()
        prediction_data = self.get_predictions(ensemble, x_test, y_test)
        ensemble_report(
            output_reference=output_reference,
            data=self.data,
            y_test=prediction_data["y_test"],
            predictions=prediction_data["predictions"],
            descriptors=self.descriptors,
            local_root=self.root,
        )
        save_ensemble(ensemble, output_reference, self.root)


if __name__ == "__main__":
    from phase1.support_functions.support_descriptors import get_numerical_descriptors
    from phase1.support_functions.vars import local_root

    input_filename = "dataset_ecfp6.csv"
    descriptor_list = get_numerical_descriptors(input_filename)
    E = Ensemble(
        data_root=local_root["data"],
        root=local_root["phase1"],
        input_file=input_filename,
        target="PPI",
        descriptors=descriptor_list,
        fraction=0.2,
    )
    E.report(output_reference="ensemble_ecfp6")
