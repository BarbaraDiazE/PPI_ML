# set local root
local_root = {
    "data": "/home/babs/Documents/DIFACQUIM/PPIClassifier/data",
    "phase1": "/home/babs/Documents/DIFACQUIM/PPIClassifier/phase1",
}

fp_list = ["ECFP4", "ECFP6", "MACCSKEYS", "AtomPairs"]
proportion_list = [0.2, 0.3]
svm_keys = {
    "kernel_keys": {"linear": "K1", "poly": "K2", "rbf": "K3", "sigmoid": "K4"},
    "descriptor_keys": {
        "ECFP4": "F1",
        "ECFP6": "F2",
        "MACCSKEYS": "F3",
        "AtomPairs": "F4",
    },
    "proportion_keys": {0.2: "P3", 0.3: "P5"},
    "libraries": {"PPI and FDA": "L6"},
    "class weight": {"balanced": "A", None: "B"},
}

rf_keys = {
    "estimators_keys": {100: "N1", 500: "N2", 1000: "N3"},
    "descriptor_keys": {
        "ECFP4": "F1",
        "ECFP6": "F2",
        "MACCSKEYS": "F3",
        "AtomPairs": "F4",
    },
    "proportion_keys": {0.2: "P3", 0.3: "P5"},
    "criterion_keys": {"gini": "G", "entropy": "E"},
    "libraries": {"PPI and FDA": "L6"},
    "class_weight": {"balanced": "A", None: "B"},
}

dt_keys = {
    "descriptor_keys": {
        "ECFP4": "F1",
        "ECFP6": "F2",
        "MACCSKEYS": "F3",
        "AtomPairs": "F4",
        "descriptors": "D1",
    },
    "proportion_keys": {0.2: "P3", 0.3: "P5"},
    "criterion_keys": {"gini": "G", "entropy": "E"},
    "libraries": {"PPI and FDA": "L6"},
    "class_weight": {"balanced": "A", None: "B"},
}

lrg_keys = {
    "solver_keys": {
        "newton-cg": "S1",
        "lbfgs": "S2",
        "liblinear": "S3",
        "sag": "S4",
        "saga": "S5",
    },
    "descriptor_keys": {
        "ECFP4": "F1",
        "ECFP6": "F2",
        "MACCSKEYS": "F3",
        "AtomPairs": "F4",
    },
    "proportion_keys": {0.2: "P3", 0.3: "P5"},
    "libraries": {"PPI and FDA": "L6"},
    "class weight": {"balanced": "A", None: "B"},
}
