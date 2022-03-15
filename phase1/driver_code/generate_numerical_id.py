"""Generate json with numerical_id(FK)  and model_name (PK)"""

import os
import json


def generate_numerical_id(algorithm_type: str, local_root: str):
    reports_root = f"{local_root}/results/trained_results/models_reports"
    report_names = os.listdir(reports_root)
    model_names = [i for i in report_names if algorithm_type in i and ".csv" in i]
    model_names = [i.replace(".csv", "") for i in model_names]
    model_names.sort()
    data_dict = {f"{algorithm_type}{i + 1}": model_names[i] for i in range(len(model_names))}
    output_root = f"{local_root}/results/trained_results/id_models"
    json_filename = open(os.path.join(output_root, f"{algorithm_type}_models_id.json"), "w")
    json.dump(data_dict, json_filename, indent=4)


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    algorithms = ["SVM", "LRG", "RF", "DT"]
    for algorithm in algorithms:
        generate_numerical_id(algorithm, local_root["phase1"])
