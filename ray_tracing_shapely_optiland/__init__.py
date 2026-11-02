"""
Python-specific module: ray_tracing_shapely_optiland package
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

Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
ray_tracing_shapely_optiland
===============================================================================
3D verification satellite: assembles ray_tracing_shapely's 2D nonimaging-optics
designs into Optiland non-sequential (NSQ) scenes, visualizes them, and persists
their ray/run data -- everything except the entry-point-discovered SMS
geometry plugin itself, which lives in the separate `optiland-sms`
distribution. [KB7: one bridge, two directions; KC7: the batch half of it
never imports the viewer.]

scene_bridge.py, visualization/renderers.py, visualization/jupyter_viewer.py,
and nsq_db.py are implemented and tested. See
dev_doc/architecture/components/optiland_scene_bridge.md and
dev_doc/architecture/components/optiland_nsq_run_registry.md for how each came to be, and
dev_doc/architecture/components/optiland_scene_bridge.md
for the one remaining open piece (a reverse, 3D-to-2D import entry point).
===============================================================================
"""

__version__ = "0.0.0"
