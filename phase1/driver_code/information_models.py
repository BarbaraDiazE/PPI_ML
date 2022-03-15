"""Generate dict with all model, information"""
import os
import json
from phase1.support_functions.get_information_models import (
    get_dt_model_information,
    get_rf_model_information,
    get_lrg_model_information,
    get_svm_model_information,
)
from phase1.support_functions.json_functions import read_id_json


def dt_json(algorithm: str, local_root: str):
    json_filename = f"{algorithm.upper()}_models_id.json"
    json_root = f"{local_root}/results/trained_results/id_models"
    data_dict = read_id_json(json_root, json_filename)
    dt_dict = {k: get_dt_model_information(data_dict[k]) for k in data_dict}
    output_root = f"{local_root}/results/trained_results/information_models"
    output_filename = open(os.path.join(output_root, f"{algorithm}_information_models.json"), "w")
    json.dump(dt_dict, output_filename, indent=4)


def rf_json(algorithm: str, local_root: str):
    json_filename = f"{algorithm.upper()}_models_id.json"
    json_root = f"{local_root}/results/trained_results/id_models"
    data_dict = read_id_json(json_root, json_filename)
    dt_dict = {k: get_rf_model_information(data_dict[k]) for k in data_dict}
    output_root = f"{local_root}/results/trained_results/information_models"
    output_filename = open(os.path.join(output_root, f"{algorithm}_information_models.json"), "w")
    json.dump(dt_dict, output_filename, indent=4)


def lrg_json(algorithm: str, local_root: str):
    json_filename = f"{algorithm.upper()}_models_id.json"
    json_root = f"{local_root}/results/trained_results/id_models"
    data_dict = read_id_json(json_root, json_filename)
    dt_dict = {k: get_lrg_model_information(data_dict[k]) for k in data_dict}
    output_root = f"{local_root}/results/trained_results/information_models"
    output_filename = open(os.path.join(output_root, f"{algorithm}_information_models.json"), "w")
    json.dump(dt_dict, output_filename, indent=4)


def svm_json(algorithm: str, local_root: str):
    json_filename = f"{algorithm.upper()}_models_id.json"
    json_root = f"{local_root}/results/trained_results/id_models"
    data_dict = read_id_json(json_root, json_filename)
    dt_dict = {k: get_svm_model_information(data_dict[k]) for k in data_dict}
    output_root = f"{local_root}/results/trained_results/information_models"
    output_filename = open(os.path.join(output_root, f"{algorithm}_information_models.json"), "w")
    json.dump(dt_dict, output_filename, indent=4)


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    local_root = local_root["phase1"]
    dt_json("dt", local_root)
    rf_json("rf", local_root)
    lrg_json("lrg", local_root)
    svm_json("svm", local_root)
