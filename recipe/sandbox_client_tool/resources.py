import re
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.tools.schemas import OpenAIFunctionToolSchema

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    sol_val = float(extract_solution(solution_str))
    gt_val = float(ground_truth)

    if(abs(gt_val - sol_val) < 1e-7):
        return 1.0
    return 0.0
