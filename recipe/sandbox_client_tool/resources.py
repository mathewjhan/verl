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


def SandboxClientTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "A tool to multiply two floats",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_val": {
                            "type": "number",
                            "description": "first value",
                        },
                        "second_val": {
                            "type": "number",
                            "description": "second value",
                        },
                    },
                    "required": ["first_val", "second_val"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)

        with open(self.config.get("python_file_path"), 'r') as f:
            self.python_code = f.read()

        self.func_entrypoint = self.config.get("python_func_entrypoint")
        self.func_arg_names = self.tool_schema.function.parameters.properties.keys()

    def args_to_str(self, args: dict) -> str:
        arg_strings = []
        for k, v in args.items():
            arg_strings.append(f"{k}={repr(v)}")
        return ",".join(arg_strings)

    def format_code(self, func_args):
        str_args = self.args_to_str(func_args)
        return f"{self.python_code}\nprint({self.func_entrypoint}({func_args}))"

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        func_args = zip(self.func_arg_names, map(lambda a : parameters[a], self.func_arg_names))
        code = self.format_code(func_args)
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)

        if language != "python":
            raise ValueError("Only Python supported with SandboxFusionPythonFunctionCallTool")

        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)

        return result, 0.0, {}

