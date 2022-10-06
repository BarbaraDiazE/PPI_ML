import pandas as pd


def get_data(filename):
    r = "/home/babs/Documents/DIFACQUIM/PPI_ML/data/"
    data = pd.read_csv(f"{r}{filename}", index_col="Unnamed: 0")
    return data


def get_coefficients(filename):
    r = "/home/babs/Documents/DIFACQUIM/PPI_ML/phase1/results/trained_results/models_reports/coeff_reports/"
    data = pd.read_csv(f"{r}{filename}")
    return data


def common_bits(a: list, b: list):
    result = [i for i in a if i in b]
    return result


def get_random_smiles(data, index):
    # get smiles bi index
    molecule = data.iloc[index]
    return molecule.SMILES


def get_positive_records(molecule_data: pd.Series):
    positives = list()
    for i in range(2048):
        if molecule_data[str(i)] == 1:
            positives.append(i)
    return positives


def filter_positive_bits(bits_list: list, data: pd.DataFrame):
    # filter considering two bits
    if len(bits_list) == 3:
        d = data[
            (data[str(bits_list[0])] == 1)
            & (data[str(bits_list[1])] == 1)
            & (data[str(bits_list[2])] == 1)
        ]
        return d
    if len(bits_list) == 2:
        d = data[(data[str(bits_list[0])] == 1) & (data[str(bits_list[1])] == 1)]
        return d
    if len(bits_list) == 1:
        d = data[(data[str(bits_list[0])] == 1)]
        return d


def filter_negative_bits(bits_list: list, data: pd.DataFrame):
    # filter considering two bits
    if len(bits_list) == 3:
        d = data[
            (data[str(bits_list[0])] == 0)
            & (data[str(bits_list[1])] == 0)
            & (data[str(bits_list[2])] == 0)
        ]
        return d
    if len(bits_list) == 2:
        d = data[(data[str(bits_list[0])] == 0) & (data[str(bits_list[1])] == 0)]
        return d
    if len(bits_list) == 1:
        d = data[(data[str(bits_list[0])] == 0)]
        return d
