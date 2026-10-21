"""Resolve archived evidence paths only within explicitly supplied study roots."""

from pathlib import Path, PurePosixPath, PureWindowsPath


class EvidencePaths:
    """Relocate historical Windows paths without guessing by a file basename.

    Relative references may cross between the supplied sibling studies. Absolute
    historical references must contain experiments/<an explicitly named study>.
    Callers still check the original content hashes before consuming the files.
    """

    def __init__(self, *roots: Path):
        self.roots = tuple(root.resolve() for root in roots)
        self.by_name = {root.name: root for root in self.roots}
        if len(self.by_name) != len(self.roots):
            raise ValueError("Study directory names must be unique")

    def resolve(self, recorded: str, base: Path) -> Path:
        normalized = recorded.replace("\\", "/")
        windows = PureWindowsPath(recorded)
        absolute = windows.is_absolute() or PurePosixPath(normalized).is_absolute()
        if not absolute:
            candidate = (base / normalized).resolve()
        else:
            candidate = Path(normalized).resolve()
            if not any(candidate.is_relative_to(root) for root in self.roots):
                parts = PurePosixPath(normalized).parts
                anchors = [
                    i
                    for i, part in enumerate(parts[:-1])
                    if part == "experiments" and parts[i + 1] in self.by_name
                ]
                if len(anchors) != 1:
                    raise ValueError(f"Unrecognized archived study path: {recorded}")
                index = anchors[0]
                tail = parts[index + 2 :]
                if ".." in tail:
                    raise ValueError("Archived absolute paths cannot traverse parents")
                candidate = self.by_name[parts[index + 1]].joinpath(*tail).resolve()
        if not any(candidate.is_relative_to(root) for root in self.roots):
            raise ValueError(f"Evidence path escapes the declared studies: {recorded}")
        return candidate
