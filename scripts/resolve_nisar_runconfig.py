#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["ruamel.yaml"]
# ///
"""Show what the NISAR InSAR workflow *actually ran*, defaults filled in.

Why this exists
---------------
A delivered ``*.rc.yaml`` next to a GUNW is the **user** runconfig: it only
lists what the PGE overrode. Every other knob -- and there are hundreds -- comes
from isce3's shipped defaults. Reading the ``.rc.yaml`` alone therefore tells
you what was *changed*, not what was *used*, and the two are easy to confuse.
For example a runconfig that says only::

    phase_unwrap:
        preprocess_wrapped_phase:
            enabled: true
            mask:
                mask_type: subswath_mask

was in fact run with ``filling_enabled: False`` (from the defaults), which means
the masked pixels were **zeroed**, not interpolated -- and with
``algorithm: snaphu``, which the runconfig never mentions.

What it does
------------
Reproduces isce3's own resolution order from
``nisar/workflows/runconfig.py``: load ``share/nisar/defaults/<workflow>.yaml``,
then ``helpers.deep_update`` the user runconfig on top of it. ``deep_update`` is
re-implemented here (it is 10 lines) so this runs without an isce3 install --
only the defaults YAML from an isce3 checkout is needed.

Note the ``flag_none_is_valid=True`` semantics that isce3 uses for ``insar``: a
key present-but-empty in the user runconfig overrides the default with ``None``
rather than falling back to it.

Usage
-----
    python resolve_nisar_runconfig.py PRODUCT.rc.yaml --isce3 ~/repos/isce3

    # just the unwrapping section, which is usually the point
    python resolve_nisar_runconfig.py PRODUCT.rc.yaml --isce3 ~/repos/isce3 \
        --section processing.phase_unwrap

    # show only keys that came from the defaults (i.e. what you'd have missed)
    python resolve_nisar_runconfig.py PRODUCT.rc.yaml --isce3 ~/repos/isce3 \
        --section processing.phase_unwrap --only-defaults
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

WORKFLOWS = ("insar", "gcov", "gslc", "focus", "static")


def deep_update(original: dict, update: dict, flag_none_is_valid: bool = True) -> dict:
    """isce3's ``nisar.workflows.helpers.deep_update``, verbatim in behaviour."""
    for key, val in update.items():
        if isinstance(val, dict) and original.get(key) is not None:
            original[key] = deep_update(original.get(key, {}), val, flag_none_is_valid)
        elif flag_none_is_valid or val is not None:
            original[key] = val
    return original


def defaults_path(isce3_root: Path, workflow: str) -> Path:
    """Locate the shipped defaults YAML in an isce3 checkout or install."""
    candidates = [
        isce3_root / "share" / "nisar" / "defaults" / f"{workflow}.yaml",
        isce3_root / "defaults" / f"{workflow}.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit(
        f"No {workflow}.yaml defaults under {isce3_root}. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def strip_runconfig_wrapper(doc: dict) -> dict:
    """Peel the ``runconfig: groups:`` envelope so sections address cleanly."""
    node = doc.get("runconfig", doc)
    return node.get("groups", node)


def get_section(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(
                f"Section {dotted!r} not found (failed at {part!r}). "
                f"Available here: {sorted(node) if isinstance(node, dict) else type(node)}"
            )
        node = node[part]
    return node


def annotate_provenance(
    resolved: Any, user: Any, path: str = ""
) -> list[tuple[str, Any, str]]:
    """Flatten ``resolved`` into ``(dotted_key, value, 'user'|'default')`` rows."""
    rows: list[tuple[str, Any, str]] = []
    if isinstance(resolved, dict):
        for k, v in resolved.items():
            sub_user = user.get(k) if isinstance(user, dict) else None
            key = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                rows.extend(annotate_provenance(v, sub_user, key))
            else:
                in_user = isinstance(user, dict) and k in user
                rows.append((key, v, "user" if in_user else "default"))
    else:
        rows.append((path, resolved, "user" if user is not None else "default"))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("runconfig", type=Path, help="Delivered PRODUCT.rc.yaml")
    p.add_argument(
        "--isce3",
        type=Path,
        default=Path.home() / "repos" / "isce3",
        help="isce3 checkout (or install prefix) holding share/nisar/defaults",
    )
    p.add_argument(
        "--workflow",
        default="insar",
        choices=WORKFLOWS,
        help="Which defaults file to merge against",
    )
    p.add_argument(
        "--section",
        default=None,
        help="Dotted path to print, e.g. processing.phase_unwrap",
    )
    p.add_argument(
        "--provenance",
        action="store_true",
        help="Flat listing tagging each leaf as [user] or [default]",
    )
    p.add_argument(
        "--only-defaults",
        action="store_true",
        help="With --provenance, show only the keys the runconfig never set.",
    )
    args = p.parse_args()

    yaml = YAML(typ="safe")
    resolved = yaml.load(defaults_path(args.isce3, args.workflow).read_text())
    user_doc = yaml.load(args.runconfig.read_text())

    resolved = strip_runconfig_wrapper(resolved)
    user = strip_runconfig_wrapper(user_doc)
    resolved = deep_update(resolved, user, flag_none_is_valid=True)

    if args.section:
        resolved = get_section(resolved, args.section)
        try:
            user = get_section(user, args.section)
        except SystemExit:
            user = {}

    if args.provenance or args.only_defaults:
        rows = annotate_provenance(resolved, user)
        if args.only_defaults:
            rows = [r for r in rows if r[2] == "default"]
        width = max((len(k) for k, _, _ in rows), default=0)
        for key, val, src in rows:
            print(f"{key:<{width}}  {val!r:<28} [{src}]")
    else:
        out = YAML()
        out.default_flow_style = False
        out.dump(resolved, sys.stdout)


if __name__ == "__main__":
    main()
