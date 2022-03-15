import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    plot_roc_curve,
    confusion_matrix,
    recall_score,
)


def plot_roc(output_reference, model, x, y, local_root):
    plot_roc_curve(
        model,
        x,
        y,
        color="red",
    )
    lw = 2
    roc_root = f"{local_root}/results/trained_results/roc_plot"
    output_filename = f"{output_reference}.png"
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(roc_root, output_filename))
    # plt.show()


def rf_report(
    output_reference: str,
    data,
    parameters,
    y_test,
    predictions,
    descriptors,
    attributes,
    local_root,
):
    report_data = {
        "Method": parameters["Method"],
        "Class weight": parameters["class weight"],
        "Max depth": parameters["max_depth"],
        "Libraries": " ".join(map(str, list(data.library.unique()))),
        "Test fraction": parameters["fraction"],
        "Descriptors": " ".join(map(str, descriptors)),
        "classes": " ".join(map(str, list(attributes["classes"]))),
        "Accuracy": round(accuracy_score(y_test, predictions), 2),
        "Balanced Accuracy": round(balanced_accuracy_score(y_test, predictions), 2),
        "Precision": round(precision_score(y_test, predictions), 2),
        "F1": round(f1_score(y_test, predictions), 2),
        "ROC AUC score": round(roc_auc_score(y_test, predictions), 2),
        "Confusion matrix": confusion_matrix(y_test, predictions),
        "Recall": round(recall_score(y_test, predictions), 2),
    }
    report = pd.DataFrame.from_dict(report_data, orient="index", columns=["value"])
    output_report_name = f"{output_reference}.csv"
    output_root = f"{local_root}/results/trained_results/models_reports"
    report.to_csv(os.path.join(output_root, output_report_name), sep=",")
    print(f"report {output_reference} is ready")


def save_model(
    model,
    output_reference: str,
    local_root: str,
):
    trained_model_root = f"{local_root}/results/trained_results/trained_models"
    output_model_name = f"{output_reference}.pkl"
    joblib.dump(model, os.path.join(trained_model_root, output_model_name))
    print(f"model {output_reference} saved")
