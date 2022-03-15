import pandas as pd
import os

from phase1.support_functions.json_functions import read_json


def get_filtered_models(root: str, algorithm: str, quartile: str) -> list:
    json_root = f"{root}/results/metrics_results/filtered_results/"
    # json_filename = f"{id_validation}_{algorithm}_{quartile}_best_models.csv"
    filename = f"{algorithm}_{quartile}_best_models.csv"
    data = pd.read_csv(os.path.join(json_root, filename))
    model_name_to_validate = data["FK model"].to_list()
    return model_name_to_validate


def models_to_validate(root: str, algorithm: str, quartile: str):
    id_models = get_filtered_models(root, algorithm, quartile)
    json_root = f"{root}/results/trained_results/information_models"
    filename = f"{algorithm.lower()}_information_models.json"
    data = read_json(json_root, filename)
    if algorithm == "svm":
        representations = [data.get(model).get("descriptor") for model in id_models]
    else:
        representations = [data.get(model).get("descriptors") for model in id_models]
    filenames = [f'{data.get(model).get("model_name")}.pkl' for model in id_models]
    merged_list = [(representations[i], filenames[i]) for i in range(0, len(representations))]
    return merged_list
