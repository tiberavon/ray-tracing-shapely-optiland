"""
Python-specific module: test rayfile source
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
A rayfile as an Optiland source: the file's rays, placed, with their flux
===============================================================================
Synthetic files only. A run's rays must be the file's records in order,
moved into the world by the placement the 2D scene gives; unequal per-ray
flux must survive; and a Lambertian file must trace like Optiland's own
Lambertian ExtendedSource: the same share within 30 deg, sin^2(30) = 1/4.
===============================================================================
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import ExtendedSourceConfig, IrradianceDetectorConfig, NSQScene, Spectrum
from optiland.nonsequential.backends.numpy_backend import NumpyBackend

from ray_tracing_shapely.optical_elements.sources.rayfile import read_rayfile, write_tm25
from ray_tracing_shapely.optical_elements.sources.source_spec import source_axes
from ray_tracing_shapely_optiland.events import split_rays
from ray_tracing_shapely_optiland.sources import add_rayfile_source_3d, scene_to_world

N = 4000
SPECTRUM = [(850.0, 50.0), (854.0, 100.0), (858.0, 50.0)]


def _lambertian(n, seed=7):
    rng = np.random.default_rng(seed)
    pos = np.c_[(rng.random(n) - 0.5) * 2.0, (rng.random(n) - 0.5) * 1.0, np.zeros(n)]
    cos_t = np.sqrt(rng.random(n))
    phi = 2 * np.pi * rng.random(n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    return pos, np.c_[sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t]


@pytest.fixture
def rayfile(tmp_path):
    pos, dirs = _lambertian(N)
    flux = np.where(np.arange(N) % 2 == 0, 1.0, 3.0) * (2.0 / (2.0 * N))     # 1:3 alternating, 2 W in all
    return read_rayfile(write_tm25(tmp_path / "f.TM25RAY", pos, dirs, flux, spectrum_nm=SPECTRUM))


def _trace(sources, num_rays, seed=1):
    scene = NSQScene()
    for add in sources:
        add(scene)
    scene.add_detector("far", CoordinateSystem(x=1e4, ry=math.pi / 2),
                       IrradianceDetectorConfig(width=1e6, height=1e6, num_pixels_x=2, num_pixels_y=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scene.trace(num_rays=num_rays, seed=seed, max_depth=4, backend=NumpyBackend(seed=seed),
                             record_paths=True)
    return split_rays(result.ray_paths["events"])


def test_a_run_emits_the_files_rays_in_order_where_the_scene_puts_them(rayfile):
    where, direction, roll = (40.0, 7.0), 30.0, 20.0
    recs = _trace([lambda sc: add_rayfile_source_3d(sc, "led", rayfile, where, direction, roll)], N)
    ax = source_axes({"direction_deg": direction, "roll_deg": roll})
    rot = np.column_stack([scene_to_world(ax["e_out"]), scene_to_world(ax["e_in"]), scene_to_world(ax["n"])])
    want_p = rayfile.positions @ rot.T + scene_to_world([*where, 0.0])
    want_d = rayfile.directions @ rot.T
    got_p = np.array([r.points[0] for r in recs])
    got_d = np.array([r.directions[0] for r in recs])
    assert all(r.source == "led" for r in recs) and len(recs) == N
    assert np.abs(got_p - want_p).max() < 1e-12 and np.abs(got_d - want_d).max() < 1e-12


def test_per_ray_flux_survives_and_sums_to_the_file(rayfile):
    recs = _trace([lambda sc: add_rayfile_source_3d(sc, "led", rayfile, (0.0, 0.0), 0.0)], N)
    birth = np.array([r.flux[0] for r in recs])
    assert birth.sum() == pytest.approx(rayfile.total_flux, rel=1e-12) == pytest.approx(2.0, rel=1e-6)
    assert np.allclose(birth[1::2] / birth[0::2], 3.0)


def test_wavelengths_come_from_the_files_spectrum_or_the_override(rayfile):
    from ray_tracing_shapely_optiland.sources import RayfileSource, rayfile_frame

    src = RayfileSource(rayfile_frame((0.0, 0.0), 0.0), rayfile)
    assert set(np.round(src.spectrum.wavelengths, 6)) == {0.85, 0.854, 0.858}
    mono = RayfileSource(rayfile_frame((0.0, 0.0), 0.0), rayfile, wavelength_um=0.6)
    assert list(mono.spectrum.wavelengths) == [0.6]


def test_a_lambertian_file_traces_like_optilands_lambertian_source(tmp_path):
    n = 40_000
    pos, dirs = _lambertian(n, seed=11)
    lamb = read_rayfile(write_tm25(tmp_path / "l.TM25RAY", pos, dirs, np.full(n, 1.0 / n), spectrum_nm=SPECTRUM))
    file_recs = _trace([lambda sc: add_rayfile_source_3d(sc, "file", lamb, (0.0, 0.0), 0.0)], n)
    own_recs = _trace([lambda sc: sc.add_source("own", CoordinateSystem(ry=math.pi / 2), ExtendedSourceConfig(
        Spectrum.monochromatic(0.854), total_flux=1.0, width=1.0, height=2.0, half_angle_deg=90.0))], n)
    sigma = math.sqrt(0.25 * 0.75 / n)
    for recs in (file_recs, own_recs):
        cos_x = np.array([r.directions[0][0] for r in recs])                 # emission along world +x
        within = np.mean(cos_x > math.cos(math.radians(30.0)))
        assert abs(within - 0.25) < 4 * sigma
        assert abs(cos_x.mean() - 2.0 / 3.0) < 4 * math.sqrt(1.0 / 18.0 / n)


def test_a_rayfile_spec_adds_every_ray_of_its_file(tmp_path):
    from ray_tracing_shapely.optical_elements.sources import rayfile_source_spec
    from ray_tracing_shapely_optiland.sources import add_source_3d, build_source_3d

    pos, dirs = _lambertian(N)
    write_tm25(tmp_path / "f.TM25RAY", pos, dirs, np.full(N, 1.0 / N), spectrum_nm=SPECTRUM)
    spec = rayfile_source_spec("led", "f.TM25RAY", (40.0, 7.0), 30.0, roll_deg=20.0, max_offset_mm=0.1,
                               base_dir=tmp_path)
    by_spec = _trace([lambda sc: add_source_3d(sc, spec, base_dir=tmp_path)], N)
    by_hand = _trace([lambda sc: add_rayfile_source_3d(sc, "led", read_rayfile(tmp_path / "f.TM25RAY"),
                                                       (40.0, 7.0), 30.0, 20.0)], N)
    assert len(by_spec) == N                                   # every ray, not the slice's
    assert np.array_equal([r.points[0] for r in by_spec], [r.points[0] for r in by_hand])
    assert np.array_equal([r.directions[0] for r in by_spec], [r.directions[0] for r in by_hand])
    with pytest.raises(ValueError, match="add_source_3d"):
        build_source_3d(spec)
