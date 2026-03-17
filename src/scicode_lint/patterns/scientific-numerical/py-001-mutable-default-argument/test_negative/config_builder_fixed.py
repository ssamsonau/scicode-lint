from dataclasses import dataclass, field


@dataclass
class Config:
    """Dataclass with proper default factory for mutable fields."""

    name: str
    tags: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def apply_transforms(data, transforms: tuple = ()):
    """Use immutable tuple as default - safe pattern."""
    for transform in transforms:
        data = transform(data)
    return data


def process_with_kwargs(item, **kwargs):
    """Use **kwargs instead of mutable default dict."""
    result = {"item": item}
    result.update(kwargs)
    return result


class ImmutableDefaults:
    """Class using immutable defaults only."""

    def __init__(self, name: str, count: int = 0, label: str | None = None):
        self.name = name
        self.count = count
        self.label = label or "default"


config = Config(name="experiment")
config.tags.append("test")
