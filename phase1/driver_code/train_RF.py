"""Instructions to train  Random Forest from different representation"""

from phase1.support_functions.rf.RF import RF
from phase1.support_functions.support_descriptors import get_numerical_descriptors
from phase1.support_functions.generate_output_model_name import get_rf_output


def execute(
    root_data: str,
    local_root: str,
    input_file: str,
    target: str,
    fp: str,
    fraction: float,
    n_estimators: int,
    criterion: str,
    max_depth,
    class_weight: str,
):
    descriptor_list = get_numerical_descriptors(input_file)
    a = RF(root_data, local_root, input_file, target, descriptor_list, fraction)
    output_reference = get_rf_output(fp, "L6", fraction, n_estimators, criterion, class_weight)
    a.report(n_estimators, criterion, max_depth, class_weight, output_reference)


"""set params to train"""
if __name__ == "__main__":
    from phase1.support_functions.vars import local_root, fp_list, proportion_list

    estimators_list = [100, 500, 1000]
    criterion_list = ["gini", "entropy"]
    balanced_list = ["balanced", None]
    max_depth_list = [None]
    for i in range(len(fp_list)):
        filename = f'{"dataset_"}{fp_list[i].lower()}{".csv"}'
        for p in range(len(proportion_list)):
            for criterion in range(len(criterion_list)):
                for estimator in range(len(estimators_list)):
                    for b in range(len(balanced_list)):
                        execute(
                            root_data=local_root["data"],
                            local_root=local_root["phase1"],
                            input_file=filename,
                            target="PPI",
                            fp=fp_list[i],
                            fraction=proportion_list[p],
                            n_estimators=estimators_list[estimator],
                            criterion=criterion_list[criterion],
                            max_depth=None,
                            class_weight=balanced_list[b],
                        )
