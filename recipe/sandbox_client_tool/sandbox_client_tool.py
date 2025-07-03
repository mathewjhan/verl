from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.tools.schemas import OpenAIFunctionToolSchema

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

