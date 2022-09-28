import os
import pandas as pd


class Data:
    def __init__(
        self,
        root_data: str,
        local_root: str,
        input_file: str,
        target: str,
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
        # print(
        #     "PPI types: ",
        #     data.PPI.unique(),
        #     "\n",
        #     "Libraries types: ",
        #     data.library.unique(),
        #     "\n",
        #     "Total compounds number: ",
        #     data.shape[0],
        # )
        self.root_data = root_data
        self.local_root = local_root
        self.target = target
        # self.descriptors = descriptors
        # self.fraction = fraction
        self.numerical_data = data.drop(ids, axis=1)
        self.data = data

    def get_smiles(self):
        print("numero de compuestos: ", self.data.shape[0])
        print("numero de compuestos unicos: ", len(self.data.SMILES.unique()))

    def get_smiles(self):
        print("numero de compuestos: ", self.data.shape[0])
        print("numero de compuestos unicos: ", len(self.data.SMILES.unique()))

    def identify_repeated(self):
        unique_smiles = list(self.data.SMILES.unique())
        print(type(unique_smiles))
        for i in unique_smiles:
            df_ = self.data[self.data["SMILES"] == i]
            if df_.shape[0] > 1:
                print(df_.iloc[0]["ipp_id"])

    def count_features(self):
        # print("numero de descriptores")
        return self.numerical_data.shape[1]
        # print(self.numerical_data.head(3))


if __name__ == "__main__":
    from phase1.support_functions.vars import fp_list, local_root

    solver_list = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    balanced_list = ["balanced", None]
    # fp_list = ["ECFP6"]
    for i in range(len(fp_list)):
        filename = f'{"dataset_"}{fp_list[i].lower()}{".csv"}'
        D = Data(
            root_data=local_root["data"],
            local_root=local_root["phase1"],
            input_file=filename,
            target="PPI",
            # fp=fp_list[i],
            # fraction=proportion_list[p],
            # solver=solver_list[solver],
            # balanced=balanced_list[b],
        )
        # D.get_smiles()
        # D.identify_repeated()
        n_feature = D.count_features()
        print(fp_list[i], ":", n_feature)
