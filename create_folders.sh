#!/bin/bash

PHASES=(
    phase1
)
for PHASE in "${PHASES[@]}"
do
mkdir -p \
    data\
    ${PHASE}/driver_code/{metrics,ensemble,validations}\
    ${PHASE}/support_functions/{dt,lrg,rf,svm,validation}\
    ${PHASE}/results/metrics_results/{metrics_plots,metrics_reports,filtered_results,metrics_stats}\
    ${PHASE}/results/validation_results\
    ${PHASE}/results/predictions\
    ${PHASE}/results/trained_results/{id_models,information_models,trained_models}\
    ${PHASE}/results/trained_results/models_reports/coeff_reports\
    ${PHASE}/results/trained_results/roc_plot/roc_data\
    ${PHASE}/results/trained_results/tree_plot

done
