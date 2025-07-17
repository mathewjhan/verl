# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def extract_solution(solution_str):
    solution = re.search(r"####\s*([-+]?\d*\.?\d+)", solution_str)
    if not solution or solution is None:
        return None
    final_solution = solution.group(1)
    return final_solution


def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    extracted = extract_solution(solution_str)
    try:
        sol_val = float(extracted)
    except Exception as e:
        print(f"Failed to extract {extracted}, exception {e}")
        return 0.0

    gt_val = float(ground_truth)

    if sol_val is None:
        return 0.0

    if abs(gt_val - sol_val) < 1e-7:
        return 1.0
    return 0.0
