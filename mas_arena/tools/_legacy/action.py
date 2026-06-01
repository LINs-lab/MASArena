from enum import Enum, EnumMeta



class DynamicEnumMeta(EnumMeta):
    def __new__(metacls, cls, bases, classdict):
        for name, value in classdict.items():
            if isinstance(value, tuple) and len(value) == 2:
                classdict[name] = (value[0], value[1])
        return super().__new__(metacls, cls, bases, classdict)


class ToolAction(Enum, metaclass=DynamicEnumMeta):
    @classmethod
    def get_value_by_name(cls, name: str) -> ToolActionInfo | None:
        members = cls.members()
        name = name.upper()
        if name in members:
            if hasattr(members[name], 'value'):
                return members[name].value
            else:
                return members[name]
        return None

    @classmethod
    def members(cls):
        return dict(filter(lambda item: not item[0].startswith("_"), cls.__dict__.items()))
