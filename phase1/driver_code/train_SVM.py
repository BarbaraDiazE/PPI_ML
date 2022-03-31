from phase1.support_functions.svm.SVM_FP import SvmFp as SVM
from phase1.support_functions.support_descriptors import get_numerical_descriptors
from phase1.support_functions.generate_output_model_name import get_svm_output


def execute(root_data, local_root, input_file, target, fp, fraction, kernel, balanced):
    descriptors_list = get_numerical_descriptors(input_file)
    a = SVM(root_data, local_root, input_file, target, descriptors_list, fraction)
    output_reference = get_svm_output(fp, "L6", fraction, kernel, balanced)
    a.report(kernel, balanced, output_reference)


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root, fp_list, proportion_list

    """set params to train"""
    kernel_list = ["linear", "poly", "rbf", "sigmoid"]
    balanced_list = ["balanced", None]

    for i in range(len(fp_list)):
        filename = f'{"dataset_"}{fp_list[i].lower()}{".csv"}'
        for j in range(len(proportion_list)):
            for kernel in range(len(kernel_list)):
                for b in range(len(balanced_list)):
                    execute(
                        root_data=local_root["data"],
                        local_root=local_root["phase1"],
                        input_file=filename,
                        target="PPI",
                        fp=fp_list[i],
                        fraction=proportion_list[j],
                        kernel=kernel_list[kernel],
                        balanced=balanced_list[b],
                    )
