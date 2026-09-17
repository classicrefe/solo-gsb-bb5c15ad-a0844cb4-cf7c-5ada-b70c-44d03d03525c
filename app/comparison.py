"""Version impact analysis: compare two published configurations.

The analysis answers two questions for a pair of versions:

1. **Direct changes** — which flags were *added*, *removed* or *modified*
   between the two versions. Flag definitions are compared on their
   normalized (canonical JSON) form, so insignificant serialization
   differences (key order, whitespace) are ignored.

2. **Indirect impact** — a change to a flag can invalidate every flag that
   transitively depends on it (its *dependers*). The prerequisite edges of
   both versions are unioned into a single graph; for every directly
   changed flag we walk the *reverse* dependency edges and return each
   reachable flag together with one shortest path (from the changed flag
   to the affected flag). Ties of equal length are broken by the
   lexicographically smallest path, and every affected flag is reported
   at most once per changed flag.

Like the evaluator and validator, this module is framework-free (plain
dicts in, plain dicts out) so it can be unit-tested with the standard
library alone.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Optional


def canonical_flag(flag: dict[str, Any]) -> str:
    """Return the canonical JSON representation of one flag definition.

    Definitions whose canonical strings are equal are considered the same
    regardless of input key order or whitespace.
    """
    return json.dumps(flag, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def reverse_dependency_graph(
    flags_a: dict[str, dict[str, Any]],
    flags_b: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, set[str]]:
    """Build the union of reverse-prerequisite edges for one or two configs.

    A prerequisite edge ``P -> Q`` means *flag P requires flag Q*; the
    reverse graph therefore maps ``Q -> {P, ...}`` from each prerequisite
    to the flags that depend on it. Edges from both versions are merged so
    that a depender removed in the new version or a prerequisite removed
    from the old version still participates in the impact propagation.
    """
    reverse: dict[str, set[str]] = {}

    def add_config(flags: dict[str, dict[str, Any]]) -> None:
        for name, spec in flags.items():
            reverse.setdefault(name, set())
            for prereq in spec.get("prerequisites", []) or []:
                dep = prereq.get("flag")
                if dep is None:
                    continue
                reverse.setdefault(dep, set()).add(name)

    add_config(flags_a)
    if flags_b is not None:
        add_config(flags_b)
    return reverse


def _shortest_paths_from(source: str, reverse: dict[str, set[str]]) -> dict[str, list[str]]:
    """BFS from ``source`` over reverse edges.

    Returns a mapping ``reachable_flag -> path`` where ``path`` starts at
    ``source`` and ends at the reachable flag. Expanding neighbours in
    sorted order makes the BFS discover each equal-length path in
    lexicographic order, so the recorded path is the shortest and, among
    equal-length paths, the lexicographically smallest.
    """
    paths: dict[str, list[str]] = {source: [source]}
    queue: deque[str] = deque([source])
    while queue:
        current = queue.popleft()
        for neighbour in sorted(reverse.get(current, ())):
            if neighbour not in paths:
                paths[neighbour] = paths[current] + [neighbour]
                queue.append(neighbour)
    return paths


class ConfigComparison:
    """Result of comparing two configurations (see :func:`compare_configs`)."""

    def __init__(
        self,
        *,
        added: list[str],
        removed: list[str],
        modified: list[str],
        impacts: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.added = added
        self.removed = removed
        self.modified = modified
        self.impacts = impacts

    def as_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "removed": self.removed,
            "modified": self.modified,
            "impacts": self.impacts,
        }


def compare_configs(
    config_from: dict[str, Any], config_to: dict[str, Any]
) -> ConfigComparison:
    """Compare two published config documents flag-by-flag.

    ``added``/``removed``/``modified`` hold flag names sorted
    lexicographically. ``impacts`` maps every directly changed flag to the
    flags indirectly affected by the change (sorted by name, each carrying
    the shortest lexicographic dependency path away from the changed flag).
    """
    flags_from: dict[str, dict[str, Any]] = config_from.get("flags", {}) or {}
    flags_to: dict[str, dict[str, Any]] = config_to.get("flags", {}) or {}

    names_from = set(flags_from)
    names_to = set(flags_to)

    added = sorted(names_to - names_from)
    removed = sorted(names_from - names_to)
    modified = sorted(
        name
        for name in names_from & names_to
        if canonical_flag(flags_from[name]) != canonical_flag(flags_to[name])
    )

    reverse = reverse_dependency_graph(flags_from, flags_to)

    impacts: dict[str, list[dict[str, Any]]] = {}
    for changed in sorted(added + removed + modified):
        paths = _shortest_paths_from(changed, reverse)
        affected = [
            {"flag": target, "path": path}
            for target, path in paths.items()
            if target != changed
        ]
        affected.sort(key=lambda item: item["flag"])
        impacts[changed] = affected

    return ConfigComparison(
        added=added, removed=removed, modified=modified, impacts=impacts
    )
