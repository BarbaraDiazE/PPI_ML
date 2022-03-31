""" Generate ref_output which correspond to PK"""
from phase1.support_functions.vars import dt_keys, rf_keys, lrg_keys, svm_keys


def get_dt_output(
    d_type: str, l_type: str, p_type: float, criterion_type: str, balance_type: str
) -> str:
    al_t = "DT"
    d_t = dt_keys["descriptor_keys"][d_type]
    l_t = l_type
    p_t = dt_keys["proportion_keys"][p_type]
    c_t = dt_keys["criterion_keys"][criterion_type]
    b_t = dt_keys["class_weight"][balance_type]
    ref_output = f"{al_t}{d_t}{l_t}{p_t}{c_t}{b_t}"
    return ref_output


def get_rf_output(
    d_type: str,
    l_type: str,
    p_type: float,
    estimators_type: int,
    criterion_type: str,
    balance_type: str,
) -> str:
    al_t = "RF"
    d_t = rf_keys["descriptor_keys"][d_type]
    l_t = l_type
    p_t = rf_keys["proportion_keys"][p_type]
    c_t = rf_keys["criterion_keys"][criterion_type]
    rf_t = rf_keys["estimators_keys"][estimators_type]
    b_t = rf_keys["class_weight"][balance_type]
    ref_output = f"{al_t}{d_t}{l_t}{p_t}{c_t}{rf_t}{b_t}"
    return ref_output


def get_lrg_output(
    d_type: str, l_type: str, p_type: float, solver_type, balance_type: bool
) -> str:
    al_t = "LRG"
    d_t = lrg_keys["descriptor_keys"][d_type]
    l_t = l_type
    p_t = lrg_keys["proportion_keys"][p_type]
    s_t = lrg_keys["solver_keys"][solver_type]
    b_t = lrg_keys["class weight"][balance_type]
    output_reference = f"{al_t}{d_t}{l_t}{p_t}{s_t}{b_t}"
    return output_reference


def get_svm_output(
    d_type: str, l_type: str, p_type: float, kernel_type, balance_type: bool
) -> str:
    al_t = "SVM"
    d_t = svm_keys["descriptor_keys"][d_type]
    l_t = l_type
    p_t = svm_keys["proportion_keys"][p_type]
    k_t = svm_keys["kernel_keys"][kernel_type]
    b_t = svm_keys["class weight"][balance_type]
    output_reference = f"{al_t}{d_t}{l_t}{p_t}{k_t}{b_t}"
    return output_reference


def get_pca_output(l_type: str) -> str:
    al_t = "PCA"
    l_t = l_type
    output_reference = f"{al_t}{l_t}"
    return output_reference
