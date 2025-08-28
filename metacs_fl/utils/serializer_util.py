from pathlib import Path
from pickle import dump, load


def serialize(data: any,
              serialized_file: Path) -> None:
    with open(file=serialized_file, mode="wb") as sf:
        dump(obj=data, file=sf)


def deserialize(serialized_file: Path):
    with open(file=serialized_file, mode="rb") as sf:
        return load(sf)
