from __future__ import annotations

from typing import Tuple


def resolve_contact_indices(contact_sensor, asset, name_keys) -> Tuple[list[int], list[int], list[str]]:
    """Resolve both contact-sensor indices and asset body indices for body patterns.

    MJLab ContactSensor does not expose ``find_bodies``. Its output is indexed by
    the sensor primary list built from ``cfg.primary``, so we reconstruct that
    primary ordering here and map requested asset bodies onto it.
    """

    primary_names = None
    try:
        primary = contact_sensor.cfg.primary
        mode = getattr(primary, "mode", None)
        pattern = getattr(primary, "pattern", None)
        if mode in ("subtree", "body"):
            _, primary_names = asset.find_bodies(pattern)
        elif mode == "geom":
            _, primary_names = asset.find_geoms(pattern)
        else:
            _, primary_names = asset.find_bodies(pattern)
    except Exception:
        primary_names = None

    if primary_names is None:
        raise RuntimeError(
            "Failed to resolve contact sensor primary names from contact_sensor.cfg.primary. "
            "Contact indices must be mapped against the sensor primary ordering explicitly."
        )

    target_ids, target_names = asset.find_bodies(name_keys)
    name_to_index = {n: i for i, n in enumerate(primary_names)}
    indices: list[int] = []
    missing = []
    for name in target_names:
        if name not in name_to_index:
            missing.append(name)
            continue
        indices.append(name_to_index[name])

    if missing:
        raise ValueError(
            f"Contact sensor primaries do not include bodies: {missing}. "
            f"Primary set size={len(primary_names)}."
        )

    return indices, target_ids, target_names
