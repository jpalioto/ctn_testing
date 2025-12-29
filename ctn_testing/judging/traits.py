"""Trait definitions for blind judging."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TraitDimension:
    """A single trait dimension for scoring."""
    name: str
    description: str
    scale: str  # e.g., "0-100"
    anchors: dict[int, str]  # score -> description

    def format_anchors(self) -> str:
        """Format anchors for display in judge prompt."""
        lines = []
        for score in sorted(self.anchors.keys()):
            lines.append(f"  {score}: {self.anchors[score]}")
        return "\n".join(lines)


@dataclass
class TraitDefinitions:
    """Collection of trait dimensions."""
    version: str
    dimensions: list[TraitDimension] = field(default_factory=list)

    def dimension_names(self) -> list[str]:
        """Get list of dimension names."""
        return [d.name for d in self.dimensions]

    def get_dimension(self, name: str) -> TraitDimension | None:
        """Get dimension by name."""
        for d in self.dimensions:
            if d.name == name:
                return d
        return None


def load_traits(path: Path) -> TraitDefinitions:
    """Load trait definitions from YAML file.

    Args:
        path: Path to traits.yaml file

    Returns:
        TraitDefinitions with all dimensions loaded

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Traits file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid traits file format: expected dict, got {type(data)}")

    version = data.get("version", "unknown")
    raw_dimensions = data.get("dimensions", [])

    if not isinstance(raw_dimensions, list):
        raise ValueError(f"Invalid dimensions format: expected list, got {type(raw_dimensions)}")

    dimensions = []
    for raw in raw_dimensions:
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid dimension format: expected dict, got {type(raw)}")

        name = raw.get("name")
        if not name:
            raise ValueError("Dimension missing required 'name' field")

        # Convert anchors keys to int (YAML may load them as strings)
        raw_anchors = raw.get("anchors", {})
        anchors = {int(k): v for k, v in raw_anchors.items()}

        dimensions.append(TraitDimension(
            name=name,
            description=raw.get("description", ""),
            scale=raw.get("scale", "0-100"),
            anchors=anchors,
        ))

    return TraitDefinitions(version=version, dimensions=dimensions)
