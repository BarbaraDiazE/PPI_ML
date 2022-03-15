import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


from phase1.support_functions.lrg.functions_LRG import lrg_report, plot_roc, save_model


class LRG:
    def __init__(
        self,
        root_data: str,
        local_root: str,
        input_file: str,
        target: str,
        descriptors: list,
        fraction: float,
    ):
        data = pd.read_csv(os.path.join(root_data, input_file), index_col="Unnamed: 0")
        self.root_data = root_data
        ids = [
            "ipp_id",
            "chembl_id",
            "SMILES",
            "library",
            "PPI family",
            "PPI",
        ]
        print(
            "PPI types: ",
            data.PPI.unique(),
            "\n",
            "Libraries types: ",
            data.library.unique(),
            "\n",
            "Total compounds number: ",
            data.shape[0],
        )
        self.root_data = root_data
        self.local_root = local_root
        self.target = target
        self.descriptors = descriptors
        self.fraction = fraction
        self.numerical_data = data.drop(ids, axis=1)
        self.data = data

    def train_model(self, solver: str, class_weight):
        """
        class_weight: dict or ‘balanced’, optional (default=None)
        solver: str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional
        """
        y = np.array(self.data[self.target])
        y = label_binarize(y, classes=["No", "Yes"])
        y = np.reshape(y, int(y.shape[0]))
        numerical_data = np.array(self.data[self.descriptors])
        x_train, x_test, y_train, y_test = train_test_split(
            numerical_data, y, test_size=self.fraction, random_state=1992
        )
        model = LogisticRegression(
            solver=solver,
            fit_intercept=True,
            class_weight=class_weight,
            random_state=1992,
            n_jobs=2,
        )
        return model.fit(x_train, y_train), x_test, y_test

    @classmethod
    def get_attributes(cls, model):
        attributes = {
            "classes": model.classes_,
            "Coeff": list(model.coef_[0]),
            "inter": model.intercept_,
            "iter": model.n_iter_,
        }
        return attributes

    @classmethod
    def get_params(cls, solver, class_weight, fraction):
        parameters = {
            "Method": "Linear Regression",
            "class weight": class_weight,
            "solver": solver,
            "fraction": fraction * 100,
        }
        return parameters

    @classmethod
    def get_predictions(cls, model, x_test, y_test):
        prediction_data = {
            "predictions": model.predict(x_test),
            "y_score": model.decision_function(x_test),
            "x_text": x_test,
            "y_test": y_test,
        }
        return prediction_data

    def report(self, solver: str, class_weight, output_reference: str):
        model, x_test, y_test = self.train_model(solver, class_weight)
        prediction_data = self.get_predictions(model, x_test, y_test)
        roc_auc = plot_roc(
            output_reference,
            prediction_data["y_test"],
            prediction_data["y_score"],
            self.local_root,
        )
        lrg_report(
            output_reference=output_reference,
            data=self.data,
            parameters=self.get_params(solver, class_weight, self.fraction),
            y_test=prediction_data["y_test"],
            predictions=prediction_data["predictions"],
            descriptors=self.descriptors,
            attributes=self.get_attributes(model),
            roc_auc=roc_auc,
            local_root=self.local_root,
        )
        save_model(model, output_reference, self.local_root)
