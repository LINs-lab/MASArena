from typing import List


class ComponentRegistry:
    def __init__(self):
        self.component_dict = {}

    def register(self, cls_name: str, cls):
        if cls_name in self.component_dict:
            raise ValueError(f"Component `{cls_name}` is already registered!")
        self.component_dict[cls_name] = cls

    def get_component(self, cls_name: str):
        if cls_name not in self.component_dict:
            raise KeyError(f"Component `{cls_name}` not found!")
        return self.component_dict[cls_name]

    def has_component(self, cls_name: str) -> bool:
        return cls_name in self.component_dict


COMPONENT_REGISTRY = ComponentRegistry()

def register_component(cls_name: str, cls):
    COMPONENT_REGISTRY.register(cls_name, cls)


class LLMRegistry:
    def __init__(self):
        self.models = {}
        self.model_configs = {}

    def register(self, key: str, model_cls, config_cls):
        if key in self.models:
            raise ValueError(f"LLM name '{key}' is already registered!")
        self.models[key] = model_cls
        self.model_configs[key] = config_cls

    def key_error_message(self, key: str):
        error_message = f"""`{key}` is not a registered model name. Currently available model names: {self.get_model_names()}. If `{key}` is a customized model, you should use @register_llm({key}) to register the model."""
        return error_message

    def get_model(self, key: str):
        model = self.models.get(key, None)
        if model is None:
            raise KeyError(self.key_error_message(key))
        return model

    def get_model_config(self, key: str):
        config = self.model_configs.get(key, None)
        if config is None:
            raise KeyError(self.key_error_message(key))
        return config

    def get_model_names(self):
        return list(self.models.keys())


LLM_REGISTRY = LLMRegistry()


def register_model(config_cls, alias: List[str] = None):
    def decorator(cls):
        class_name = cls.__name__
        LLM_REGISTRY.register(class_name, cls, config_cls)
        if alias is not None:
            for alia in alias:
                LLM_REGISTRY.register(alia, cls, config_cls)
        return cls

    return decorator
