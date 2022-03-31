import os


def read_id_json(json_root, json_filename):
    import json

    f = open(os.path.join(json_root, json_filename))
    data_dict = json.load(f)
    return data_dict


def read_json(json_root, json_filename):
    import json

    f = open(os.path.join(json_root, json_filename))
    data_dict = json.load(f)
    return data_dict
