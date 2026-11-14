"""
Python-specific module: irradiance plot
Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

===============================================================================
IRRADIANCE PLOT -- an irradiance map, its two sections, and the 2D profile
===============================================================================
``plot_irradiance_map`` draws an ``IrradianceMap``
(``ray_tracing_shapely.analysis.irradiance``):

- **the map** on the plane, u across (the 2D scene's line), v out of the 2D
  plane, in flux per mm^2, with dashed lines where the sections are cut;
- **its two sections** through (0, 0), along u and along v, in the same
  unit;
- given a 2D ``LineProfile`` on the same plane, **the flux per mm of radius**
  in 3D (``radial_profile``) against the 2D profile folded about the axis,
  each over its own peak. A 2D section carries dPhi/dr, not the irradiance
  (``analysis.irradiance``), and its flux is per ray, so the two compare in
  shape.

``save_irradiance_plot`` writes the figure as SVG and PNG and returns a
descriptor shaped as ``analysis.render_result``'s (paths, size, a
description), with the map's ``summary()``: what a tool returns.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ray_tracing_shapely.analysis.irradiance import IrradianceMap, LineProfile

#: The 3D curves' colour and the 2D curve's (the validated palette's slots 1 and 2).
COLOUR_3D, COLOUR_2D = "#2a78d6", "#eb6834"


def plot_irradiance_map(imap: IrradianceMap, *, profile_2d: Optional[LineProfile] = None,
                        title: Optional[str] = None, section_half_width_mm: Optional[float] = None,
                        cmap: str = "magma", figsize=(13.0, 8.5)):
    """The map, its sections through (0, 0) and, with ``profile_2d``, dPhi/dr in 3D against 2D.

    ``section_half_width_mm``: the band each section averages
    (``IrradianceMap.section``); two pixels each side by default.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 (only a caller who draws needs matplotlib)

    unit = f"{imap.flux_unit}/mm²"
    band = 2.0 * float(imap.u_edges_mm[1] - imap.u_edges_mm[0]) if section_half_width_mm is None \
        else float(section_half_width_mm)
    fig = plt.figure(figsize=figsize, layout="constrained")
    grid = fig.add_gridspec(3 if profile_2d is not None else 2, 2, width_ratios=(1.25, 1.0))
    ax_map = fig.add_subplot(grid[:, 0])
    extent = (imap.u_edges_mm[0], imap.u_edges_mm[-1], imap.v_edges_mm[0], imap.v_edges_mm[-1])
    image = ax_map.imshow(imap.values, origin="lower", extent=extent, cmap=cmap, interpolation="nearest",
                          aspect="equal")
    fig.colorbar(image, ax=ax_map, label=f"irradiance ({unit})", shrink=0.8)
    for line in (ax_map.axhline, ax_map.axvline):
        line(0.0, color="white", lw=0.7, ls="--", alpha=0.7)
    for edge in (-band, band):
        ax_map.axhline(edge, color="white", lw=0.4, ls=":", alpha=0.5)
        ax_map.axvline(edge, color="white", lw=0.4, ls=":", alpha=0.5)
    ax_map.set_xlabel("u, across the axis in the 2D plane (mm)")
    ax_map.set_ylabel("v, out of the 2D plane (mm)")
    s = imap.summary()
    ax_map.set_title(f"{s['share_on_map']:.1%} of the light that left lands on the map")

    for row, axis in enumerate(("u", "v")):
        ax = fig.add_subplot(grid[row, 1])
        x, y = imap.section(axis, 0.0, band)
        ax.plot(x, y, color=COLOUR_3D, lw=1.5, drawstyle="steps-mid")
        other = "v" if axis == "u" else "u"
        ax.set_title(f"section along {axis}, |{other}| <= {band:.2g} mm", fontsize=10)
        ax.set_xlabel(f"{axis} (mm)")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
        ax.set_xlim(extent[0], extent[1])

    if profile_2d is not None:
        ax = fig.add_subplot(grid[2, 1])
        r_edges = np.arange(0.0, float(extent[1]) + 1e-9, float(profile_2d.u_edges_mm[1] - profile_2d.u_edges_mm[0]))
        for (x, y), colour, label in ((imap.radial_profile(r_edges), COLOUR_3D, "3D, the map about the axis"),
                                      (profile_2d.folded(r_edges), COLOUR_2D, "2D trace, both sides")):
            peak = float(np.max(y)) if len(y) and np.max(y) > 0 else 1.0
            ax.plot(x, y / peak, color=colour, lw=1.5, drawstyle="steps-mid", label=label)
        ax.set_title("flux per mm of radius, each over its own peak (what a 2D section carries)", fontsize=10)
        ax.set_xlabel("distance from the axis (mm)")
        ax.set_ylabel("relative")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(0.0, extent[1])

    if title:
        fig.suptitle(title)
    return fig


def save_irradiance_plot(imap: IrradianceMap, path_stem: Any, *, profile_2d: Optional[LineProfile] = None,
                         title: Optional[str] = None, dpi: int = 110, **kwargs) -> Dict[str, Any]:
    """Draw and write the plot as ``<stem>.svg`` and ``<stem>.png``; return a descriptor.

    Returns:
        ``{"svg_path", "png_path", "png_available", "width", "height",
        "description", "summary"}``, plain JSON: ``width``/``height`` in
        pixels of the PNG, ``summary`` the map's.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_irradiance_map(imap, profile_2d=profile_2d, title=title, **kwargs)
    svg, png = stem.with_suffix(".svg"), stem.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=dpi)
    width, height = (int(round(v * dpi)) for v in fig.get_size_inches())
    plt.close(fig)
    s = imap.summary()
    return {"svg_path": str(svg), "png_path": str(png), "png_available": True, "width": width, "height": height,
            "description": (f"irradiance on the plane at {s['plane']['origin']} (axis {s['plane']['direction_deg']} deg), "
                            f"{s['extent_mm']['u'][1] * 2:g} mm square; {s['share_on_map']:.1%} of the light that left"),
            "summary": s}


__all__ = ["plot_irradiance_map", "save_irradiance_plot"]
