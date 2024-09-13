import ray
from dataclasses import dataclass
from typing import Dict, Union


Handle = ray.actor.ActorHandle
Future = ray._raylet.ObjectRef


@dataclass
class ActorInfo:
    handle: Handle
    future: Future
    host_name: str


class ActorInfoCollection:
    _by_handle: Dict[Handle, ActorInfo] = {}
    _by_future: Dict[Future, ActorInfo] = {}
    _by_host_name: Dict[str, ActorInfo] = {}

    @classmethod
    def register(cls, data: ActorInfo) -> None:
        """Registers an ActorInstance."""
        cls._by_handle[data.handle] = data
        cls._by_future[data.future] = data
        cls._by_host_name[data.host_name] = data

    @classmethod
    def get(cls, key: Union[Handle, Future, str]) -> ActorInfo:
        """Retrieves a registered ActorInfo.

        Raises:
            ValueError, if the key was never registered.
        """
        if isinstance(key, str):
            if key in cls._by_host_name:
                return cls._by_host_name[key]
            else:
                raise ValueError(f"Host name {key} was not recognized.")
        elif isinstance(key, Future):
            if key in cls._by_future:
                return cls._by_future[key]
            else:
                raise ValueError(f"Future {key} was not recognized.")
        elif isinstance(key, Handle):
            if key in cls._by_handle:
                return cls._by_handle[key]
            else:
                raise ValueError(f"Handle {key} was not recognized.")
        else:
            raise ValueError(f"Provided key {key} was of an unrecognized dtype ({type(key)}).")

    @classmethod
    def delete(cls, key: Union[Handle, Future, str]):
        """Deletes an ActorInfo by key.

        Raises:
            ValueError, if the key was never registered.
        """
        info = cls.get(key)

        del cls._by_host_name[info.host_name]
        del cls._by_future[info.future]
        del cls._by_handle[info.handle]

    @classmethod
    def clear(cls):
        """Deletes all registered info."""
        cls._by_host_name.clear()
        cls._by_future.clear()
        cls._by_handle.clear()
