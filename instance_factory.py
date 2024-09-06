import builtins
from typing import Dict, Any
from pydantic import BaseModel, validator


class InstanceFactory(BaseModel):
    module: str
    parameters: Dict[str, Any] = {}

    @validator("parameters", pre=True)
    def _ensure_dict(cls, v):
        if v is None:
            return {}
        return v

    def create(self):
        splits = self.module.split(".")
        if len(splits) == 0:
            raise Exception("Invalid module name: {}".format(self.module))
        if len(splits) == 1:
            g = globals()
            if self.module in g:
                class_type = g[self.module]
            else:
                class_type = getattr(builtins, self.module)
            return class_type(**self.parameters)
        else:
            path = ".".join(splits[:-1])
            module = __import__(path, fromlist=[splits[-1]])
            return getattr(module, splits[-1])(**self.parameters)
