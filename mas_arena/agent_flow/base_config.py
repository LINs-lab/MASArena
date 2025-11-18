from typing import Optional, List

from mas_arena.core_serializer.component import SerializableComponent


class BaseConfig(SerializableComponent):
    """
    Base configuration class for the MAS Arena framework.

    This class serves as a base for all configuration classes in the MAS Arena framework.
    It inherits from SerializableComponent to ensure that configurations can be serialized
    and deserialized as needed.
    """

    def save(self, path: str, **kwargs):
        return super().save_component(path, **kwargs)

    def get_set_params(self, ignore: List[str] = []) -> dict:
        """Get a dictionary of explicitly set parameters.

        Args:
            ignore: List of parameter names to ignore

        Returns:
            dict: Dictionary of explicitly set parameters, excluding 'class_name' and ignored parameters
        """
        explicitly_set_fields = {field: getattr(self, field) for field in self.model_fields_set}
        if self.kwargs:
            explicitly_set_fields.update(self.kwargs)
        for field in ignore:
            explicitly_set_fields.pop(field, None)
        explicitly_set_fields.pop("class_name", None)
        return explicitly_set_fields

    def get_config_params(self) -> List[str]:
        """Get a list of configuration parameters.

        Returns:
            List[str]: List of configuration parameter names, excluding 'class_name'
        """
        config_params = list(type(self).model_fields.keys())
        config_params.remove("class_name")
        return config_params