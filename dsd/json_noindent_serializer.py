import json
from abc import ABC, abstractmethod
from typing import Union, Dict, Any


class NoIndent:
    # Value wrapper. Placing a value in this will stop it from being indented when converting to JSON
    # using SuppressableIndentEncoder

    def __init__(self, value: Any) -> None:
        self.value = value


class JSONSerializable(ABC):
    @abstractmethod
    def to_json_serializable(self, suppress_indent: bool = True) -> Union[NoIndent, Dict[str, Any]]:
        raise NotImplementedError()


def json_encode(obj: JSONSerializable, suppress_indent: bool = True) -> str:
    encoder = SuppressableIndentEncoder if suppress_indent else json.JSONEncoder
    # from dsd.stopwatch import Stopwatch
    # sw = Stopwatch()
    serializable = obj.to_json_serializable(suppress_indent=suppress_indent)
    # sw.log(f'{obj.__class__.__name__}.to_json_serializable', units='s')
    json_str = json.dumps(serializable, cls=encoder, indent=2)
    # sw.log(f'json.dumps                 ', units='s')
    return json_str


class SuppressableIndentEncoder(json.JSONEncoder):
    def __init__(self, *args: list, **kwargs: dict) -> None:
        self.unique_id = 0
        super().__init__(*args, **kwargs)  # type: ignore
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map: dict = {}

    def default(self, obj: Any) -> Any:
        if isinstance(obj, NoIndent):
            # key = uuid.uuid1().hex # this caused problems with Brython.
            key = self.unique_id
            self.unique_id += 1
            value_str: str = json.dumps(obj.value, **self.kwargs)  # type: ignore
            self._replacement_map[key] = value_str
            return f"@@{key}@@"
        else:
            return super().default(obj)

    def encode(self, obj: Any) -> Any:
        result = super().encode(obj)
        for k, v in self._replacement_map.items():
            result = result.replace(f'"@@{k}@@"', v)
        return result
