import os
import numpy as np
import json
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def get_ecfp6(smiles):
    ms = [Chem.MolFromSmiles(element) for element in smiles]
    fp = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in ms]
    return fp


class Prediction:
    def __init__(
        self,
        data_root: str,
        root: str,
        input_file: str,
        descriptors: list,
        test_compound: str,
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
        self.root = root
        self.data = data
        self.descriptors = descriptors
        self.numerical_data = data.drop(ids, axis=1)
        self.test_compound = test_compound

    def explicit_descriptor(self):
        fp = get_ecfp6([self.test_compound])
        output = []
        for f in fp:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(f, arr)
            output.append(arr)
        return np.asarray(output)

    def predict(self, model_info: tuple) -> str:
        test_data = self.explicit_descriptor()
        model = joblib.load(
            os.path.join(
                self.root,
                "results",
                "trained_results",
                "trained_models",
                (model_info[1]),
            )
        )
        x = model.predict(test_data)
        prediction = str()
        if x[0] == 0:
            prediction = "Inactive"
        elif x[0] == 1:
            prediction = "Active"
        return prediction


model_filenames = [
    ("RF27", "RFF2L6P3EN2A.pkl"),
    ("LRG22", "LRGF2L6P3S1B.pkl"),
    ("LRG24", "LRGF2L6P3S2B.pkl"),
    ("LRG27", "LRGF2L6P3S4A.pkl"),
    ("SVM22", "SVMF2L6P3K3B.pkl"),
    ("Ensemble1", "ensemble_ecfp6_1.pkl"),
    ("Ensemble2", "ensemble_ecfp6_2.pkl"),
]


if __name__ == "__main__":
    from phase1.support_functions.support_descriptors import get_numerical_descriptors
    from phase1.support_functions.vars import local_root
    import pandas as pd

    input_filename = "dataset_ecfp6.csv"
    descriptor_list = get_numerical_descriptors(input_filename)
    json_path = os.getcwd()
    f = open(os.path.join(json_path, "test_compounds.json"))
    test_compounds = json.load(f)
    results = dict()
    for key, value in test_compounds.items():
        test_smiles_ = value
        P = Prediction(
            data_root=local_root["data"],
            root=local_root["phase1"],
            input_file=input_filename,
            descriptors=descriptor_list,
            test_compound=test_smiles_,
        )
        _result = dict()
        for _model in model_filenames:
            _result[_model[0]] = P.predict(_model)
        results[key] = _result
    df = pd.DataFrame.from_dict(results).transpose()
    print(df)
    df.to_csv(os.path.join(f'{local_root["phase1"]}/results/predictions', "predictions_ecfp6.csv"))
