class DescribeDT:
    # def __init__(self):
    #     pass

    @classmethod
    def get_criterion(cls, filename):
        if "G" in filename:
            return "gini"
        elif "E" in filename:
            return "entropy"

    @classmethod
    def get_libraries(cls, filename):
        if "L6" in filename:
            return "PPI and FDA"

    @classmethod
    def get_descriptors(cls, filename):
        if "F1" in filename:
            return "ECFP4"
        elif "F2" in filename:
            return "ECFP6"
        elif "F3" in filename:
            return "MACCSKEYS"
        elif "F4" in filename:
            return "AtomPairs"
        else:
            return "Physicochemical descriptors"

    @classmethod
    def get_proportion(cls, filename):
        if "P3" in filename:
            return 0.2
        elif "P5" in filename:
            return 0.3

    @classmethod
    def get_class_weight(cls, filename):
        if "A" in filename:
            return "balanced"
        elif "B" in filename:
            return "None"


class DescribeRF:
    def __init__(self):
        pass

    @classmethod
    def get_estimator(cls, filename):
        if "N1" in filename:
            return "100"
        elif "N2" in filename:
            return "500"
        elif "N3" in filename:
            return "1000"

    @classmethod
    def get_libraries(cls, filename):
        if "L6" in filename:
            return "PPI and FDA"

    @classmethod
    def get_descriptors(cls, filename):
        if "F1" in filename:
            return "ECFP4"
        elif "F2" in filename:
            return "ECFP6"
        elif "F3" in filename:
            return "MACCSKEYS"
        elif "F4" in filename:
            return "AtomPairs"

    @classmethod
    def get_proportion(cls, filename):
        if "P3" in filename:
            return 0.2
        elif "P5" in filename:
            return 0.3

    @classmethod
    def get_criterion(cls, filename):
        if "G" in filename:
            return "gini"
        elif "E" in filename:
            return "entropy"

    @classmethod
    def get_class_weight(cls, filename):
        if "A" in filename:
            return "balanced"
        elif "B" in filename:
            return "None"

    def get_model_information(self, filename):
        model_information = {
            "model_name": filename,
            "descriptors": self.get_descriptors(filename),
            "proportion": self.get_proportion(filename),
            "criterion": self.get_criterion(filename),
            "libraries": self.get_libraries(filename),
            "class weight": self.get_class_weight(filename),
        }
        return model_information


class DescribeLRG:
    def __init__(self):
        pass

    @classmethod
    def get_solver(cls, filename):
        if "S1" in filename:
            return "newton-cg"
        elif "S2" in filename:
            return "lbfgs"
        elif "S3" in filename:
            return "liblinear"
        elif "S4" in filename:
            return "sag"
        elif "S5" in filename:
            return "saga"

    @classmethod
    def get_descriptors(cls, filename):
        if "F1" in filename:
            return "ECFP4"
        elif "F2" in filename:
            return "ECFP6"
        elif "F3" in filename:
            return "MACCSKEYS"
        elif "F4" in filename:
            return "AtomPairs"

    @classmethod
    def get_proportion(cls, filename):
        if "P3" in filename:
            return 0.2
        elif "P5" in filename:
            return 0.3

    @classmethod
    def get_libraries(cls, filename):
        if "L6" in filename:
            return "PPI and FDA"

    @classmethod
    def get_class_weight(cls, filename):
        if "A" in filename:
            return "balanced"
        elif "B" in filename:
            return "None"


class DescribeSVM:
    def __init__(self):
        pass

    @classmethod
    def get_kernel(cls, filename):
        if "K1" in filename:
            return "linear"
        elif "K2" in filename:
            return "poly"
        elif "K3" in filename:
            return "rbf"
        elif "K4" in filename:
            return "sigmoid"

    @classmethod
    def get_descriptors(cls, filename):
        if "F1" in filename:
            return "ECFP4"
        elif "F2" in filename:
            return "ECFP6"
        elif "F3" in filename:
            return "MACCSKEYS"
        elif "F4" in filename:
            return "AtomPairs"

    @classmethod
    def get_proportion(cls, filename):
        if "P3" in filename:
            return 0.2
        elif "P5" in filename:
            return 0.3

    @classmethod
    def get_libraries(cls, filename):
        if "L6" in filename:
            return "PPI and FDA"

    @classmethod
    def get_class_weight(cls, filename):
        if "A" in filename:
            return "balanced"
        elif "B" in filename:
            return "None"


def get_dt_model_information(filename):
    dt = DescribeDT()
    model_information = {
        "model_name": filename,
        "descriptors": dt.get_descriptors(filename),
        "proportion": dt.get_proportion(filename),
        "criterion": dt.get_criterion(filename),
        "libraries": dt.get_libraries(filename),
        "class weight": dt.get_class_weight(filename),
    }
    return model_information


def get_rf_model_information(filename):
    rf = DescribeRF()
    model_information = {
        "model_name": filename,
        "descriptors": rf.get_descriptors(filename),
        "proportion": rf.get_proportion(filename),
        "estimators": rf.get_estimator(filename),
        "criterion": rf.get_criterion(filename),
        "libraries": rf.get_libraries(filename),
        "class weight": rf.get_class_weight(filename),
    }
    return model_information


def get_lrg_model_information(filename):
    rf = DescribeLRG()
    model_information = {
        "model_name": filename,
        "descriptors": rf.get_descriptors(filename),
        "proportion": rf.get_proportion(filename),
        "solver_keys": rf.get_solver(filename),
        "libraries": rf.get_libraries(filename),
        "class weight": rf.get_class_weight(filename),
    }
    return model_information


def get_svm_model_information(filename):
    svm = DescribeSVM()
    model_information = {
        "model_name": filename,
        "descriptor": svm.get_descriptors(filename),
        "proportion": svm.get_proportion(filename),
        "kernel": svm.get_kernel(filename),
        "libraries": svm.get_libraries(filename),
        "class weight": svm.get_class_weight(filename),
    }
    return model_information
