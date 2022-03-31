"""Instructions to train LRG from different representations"""
from phase1.support_functions.lrg.LRG import LRG
from phase1.support_functions.support_descriptors import get_numerical_descriptors
from phase1.support_functions.generate_output_model_name import get_lrg_output


def execute(
    root_data: str,
    local_root: str,
    input_file: str,
    target: str,
    fp: str,
    fraction: float,
    solver: str,
    balanced: str,
):
    descriptors_list = get_numerical_descriptors(input_file)
    a = LRG(root_data, local_root, input_file, target, descriptors_list, fraction)
    output_reference = get_lrg_output(fp, "L6", fraction, solver, balanced)
    a.report(solver, balanced, output_reference)


"""set params to train"""
if __name__ == "__main__":
    from phase1.support_functions.vars import local_root, fp_list, proportion_list

    solver_list = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    balanced_list = ["balanced", None]

    for i in range(len(fp_list)):
        filename = f'{"dataset_"}{fp_list[i].lower()}{".csv"}'
        for p in range(len(proportion_list)):
            for solver in range(len(solver_list)):
                for b in range(len(balanced_list)):
                    execute(
                        root_data=local_root["data"],
                        local_root=local_root["phase1"],
                        input_file=filename,
                        target="PPI",
                        fp=fp_list[i],
                        fraction=proportion_list[p],
                        solver=solver_list[solver],
                        balanced=balanced_list[b],
                    )
