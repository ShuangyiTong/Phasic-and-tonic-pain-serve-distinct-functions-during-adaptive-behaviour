# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import json
import dill

cpp_res = 'alse_hand_fitted.json'
py_res = 'temp/testTrueFalseExpt4.dill'

ACCURACY_EPSILON = 1e-4

with open(cpp_res, 'r') as f:
    cpp_fitted_params = json.load(f)

with open(py_res, 'rb') as f:
    py_fitted_res = dill.load(f)

py_fitted_params = {}
for subject, stimulation_results in zip(py_fitted_res[0], py_fitted_res[1]):
    py_fitted_params[subject] = list(stimulation_results[1][0]) + [stimulation_results[1][1]]

for subject in cpp_fitted_params.keys():
    py_params = py_fitted_params[subject]
    cpp_params = cpp_fitted_params[subject]

    for i in range(9):
        if abs(py_params[i] - cpp_params[i]) > ACCURACY_EPSILON:
            print('Error in ' + subject)
            print('Py: ', py_params)
            print('Cpp: ', cpp_params)

print('Validation complete')