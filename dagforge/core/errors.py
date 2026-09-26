"""分层错误码 E100~E500，所有异常继承自 DagForgeError。"""


class DagForgeError(Exception):
    code = "E000"

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class ConfigError(DagForgeError):
    code = "E100"


class DataError(DagForgeError):
    code = "E200"


class DiscoveryError(DagForgeError):
    code = "E300"


class EvalError(DagForgeError):
    code = "E400"


class PipelineError(DagForgeError):
    code = "E500"
