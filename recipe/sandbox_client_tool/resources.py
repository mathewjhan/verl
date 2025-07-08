import re
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.tools.schemas import OpenAIFunctionToolSchema

def extract_solution(solution_str):
    solution = re.search(r'####\s*([-+]?\d*\.?\d+)', solution_str)
    if(not solution or solution is None):
        return None
    final_solution = solution.group(1)
    return final_solution


def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    print("SOL_STR", solution_str)
    print("GT", ground_truth)
    extracted = extract_solution(solution_str)
    try:
        sol_val = float(extracted)
    except:
        print(f"Failed to extract {extracted}")
        return 0.0

    gt_val = float(ground_truth)

    if(sol_val is None):
        return 0.0

    if(abs(gt_val - sol_val) < 1e-7):
        return 1.0
    return 0.0
