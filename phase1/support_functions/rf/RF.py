import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from phase1.support_functions.rf.functions_RF import rf_report, plot_roc, save_model


class RF:
    def __init__(self, data_root: str, local_root, input_file, target, descriptors, fraction):
        data = pd.read_csv(os.path.join(data_root, input_file), index_col="Unnamed: 0")
        self.data_root = data_root
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
        self.data_root = data_root
        self.local_root = local_root
        self.target = target
        self.descriptors = descriptors
        self.fraction = fraction
        self.numerical_data = data.drop(ids, axis=1)
        self.data = data

    def train_model(self, n_estimators, criterion, max_depth, class_weight):
        """
        n_estimators: int, {100,500,1000}.
        criterion: str {"gini", "entropy”}
        max_depth, int, default=None
        class_weight: str, {‘balanced’ or None}
        """
        y = np.array(self.data[self.target])
        y = label_binarize(y, classes=["No", "Yes"])
        numerical_data = np.array(self.data[self.descriptors])
        x_train, x_test, y_train, y_test = train_test_split(
            numerical_data, y, test_size=self.fraction, random_state=1992
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            # min_samples_split=2,
            # min_samples_leaf=1,
            # min_weight_fraction_leaf=0.0,
            # max_features="auto",
            # max_leaf_nodes=None,
            random_state=2020,
            class_weight=class_weight,
            # ccp_alpha=0.0,
        )
        return model.fit(x_train, y_train), x_test, y_test

    @classmethod
    def get_attributes(cls, model):
        attributes = {
            "classes": model.classes_,
            "base_estimator": model.base_estimator_,
            "estimators": model.estimators_,
            "feature_importances": model.feature_importances_,
        }
        return attributes

    @classmethod
    def get_params(cls, class_weight, n_estimators, criterion, max_depth, fraction):
        parameters = {
            "Method": "Random Forest",
            "class weight": class_weight,
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "fraction": fraction * 100,
        }
        return parameters

    @classmethod
    def get_predictions(cls, model, x_test, y_test):
        prediction_data = {
            "predictions": model.predict(x_test),
            "x_text": x_test,
            "y_test": y_test,
        }
        return prediction_data

    def report(self, n_estimators, criterion, max_depth, class_weight, output_reference):
        model, x_test, y_test = self.train_model(n_estimators, criterion, max_depth, class_weight)
        prediction_data = self.get_predictions(model, x_test, y_test)
        plot_roc(output_reference, model, x_test, y_test, self.local_root)
        rf_report(
            output_reference=output_reference,
            data=self.data,
            parameters=self.get_params(
                class_weight, n_estimators, criterion, max_depth, self.fraction
            ),
            y_test=prediction_data["y_test"],
            predictions=prediction_data["predictions"],
            descriptors=self.descriptors,
            attributes=self.get_attributes(model),
            local_root=self.local_root,
        )
        save_model(model, output_reference, self.local_root)
