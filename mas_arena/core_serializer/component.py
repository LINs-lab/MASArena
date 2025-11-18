from pydantic._internal._model_construction import ModelMetaclass
from pydantic import BaseModel, ValidationError

from .callbacks import callback_manager
from .registry import COMPONENT_REGISTRY, register_component
from mas_arena.utils.serialization_utils import custom_serializer, get_base_module_init_error_message
import json
from typing import List, Dict, Any


class ComponentRegistry(ModelMetaclass):

    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Custom metaclass to register components in a global registry.

        This metaclass automatically registers any class that inherits from
        SerializableComponent into the global component registry.
        """
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        register_component(name, cls)
        return cls


class SerializableComponent(BaseModel, metaclass=ComponentRegistry):
    class_name: str = None
    model_config = {"arbitrary_types_allowed": True, "extra": "allow", "protected_namespaces": (),
                    "validate_assignment": False}

    def __init_subclass__(cls, **kwargs):
        """
        Subclass initialization method that automatically sets the class_name attribute.

        Args:
            cls (Type): The subclass being initialized
            **kwargs (Any): Additional keyword arguments
        """
        super().__init_subclass__(**kwargs)
        cls.class_name = cls.__name__

    def __init__(self, **kwargs):
        """
        Initializes a BaseModule instance.

        Args:
            **kwargs (Any): Keyword arguments used to initialize the instance

        Raises:
            ValidationError: When parameter validation fails
            Exception: When other errors occur during initialization
        """

        try:
            for field_name, _ in type(self).model_fields.items():
                field_value = kwargs.get(field_name, None)
                if field_value:
                    kwargs[field_name] = self._process_data(field_value)
                # if field_value and isinstance(field_value, dict) and "class_name" in field_value:
                #     class_name = field_value.get("class_name")
                #     sub_cls = MODULE_REGISTRY.get_module(cls_name=class_name)
                #     kwargs[field_name] = sub_cls._create_instance(field_value)
            super().__init__(**kwargs)
            self.init_module()
        except (ValidationError, Exception) as e:
            exception_handler = callback_manager.get_callback("exception_buffer")
            if exception_handler is None:
                error_message = get_base_module_init_error_message(
                    cls=self.__class__,
                    data=kwargs,
                    errors=e
                )
                print(error_message)
                raise
            else:
                exception_handler.add(e)

    def init_module(self):
        """
        Module initialization method that subclasses can override to provide additional initialization logic.
        """
        pass

    def save_component(self, ):

        def to_json(self, use_indent: bool = False, ignore: List[str] = [], **kwargs) -> str:
            """
            Convert the BaseModule to a JSON string.

            Args:
                use_indent: Whether to use indentation
                ignore: List of field names to ignore
                **kwargs (Any): Additional keyword arguments

            Returns:
                str: The JSON string
            """
            if use_indent:
                kwargs["indent"] = kwargs.get("indent", 4)
            else:
                kwargs.pop("indent", None)
            if kwargs.get("default", None) is None:
                kwargs["default"] = custom_serializer
            data = self.to_dict(exclude_none=True)
            for ignore_field in ignore:
                data.pop(ignore_field, None)
            return json.dumps(data, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs):
        """
        Create an instance of the class from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary to create the instance from.
            **kwargs: Additional keyword arguments.

        Returns:
            SerializableComponent: An instance of the class.
        """
        use_logger = kwargs.get("log", True)
        try:
            class_name = data.get("class_name", None)
            if class_name:
                cls = COMPONENT_REGISTRY.get_component(class_name)
            component = cls._create_instance(data)
        finally:
            pass

        return component

    def to_dict(self, exclude_none: bool = True, ignore: List[str] = [], **kwargs) -> dict:
        """
        Convert the BaseModule to a dictionary.

        Args:
            exclude_none: Whether to exclude fields with None values
            ignore: List of field names to ignore
            **kwargs (Any): Additional keyword arguments

        Returns:
            dict: Dictionary containing the object data
        """
        data = {}
        for field_name, _ in type(self).model_fields.items():
            if field_name in ignore:
                continue
            field_value = getattr(self, field_name, None)
            if exclude_none and field_value is None:
                continue
            if isinstance(field_value, SerializableComponent):
                data[field_name] = field_value.to_dict(exclude_none=exclude_none, ignore=ignore)
            elif isinstance(field_value, list):
                data[field_name] = [
                    item.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(item,
                                                                                         SerializableComponent) else item
                    for item in field_value
                ]
            elif isinstance(field_value, dict):
                data[field_name] = {
                    key: value.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(value,
                                                                                               SerializableComponent) else value
                    for key, value in field_value.items()
                }
            else:
                data[field_name] = field_value

        return data

    @classmethod
    def _process_data(cls, data: Any) -> Any:
        """
        Recursive method for processing data, with special handling for dictionaries containing class_name.

        Args:
            data: Data to be processed

        Returns:
            Processed data
        """
        if isinstance(data, dict):
            if "class_name" in data:
                sub_class = COMPONENT_REGISTRY.get_module(data.get("class_name"))
                return sub_class._create_instance(data)
            else:
                return {k: cls._process_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [cls._process_data(x) for x in data]
        else:
            return data

    @property
    def kwargs(self) -> dict:
        """
        Returns the extra fields of the model.

        Returns:
            dict: Dictionary containing all extra keyword arguments
        """
        return self.model_extra

    @classmethod
    def _create_instance(cls, data: Dict[str, Any]) -> "BaseModule":
        """
        Internal method for creating an instance from a dictionary.

        Args:
            data: Dictionary containing instance data

        Returns:
            BaseModule: The created instance
        """
        processed_data = {k: cls._process_data(v) for k, v in data.items()}
        # print(processed_data)
        return cls.model_validate(processed_data)