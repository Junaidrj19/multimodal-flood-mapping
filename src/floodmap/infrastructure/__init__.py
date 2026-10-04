"""Infrastructure exposure from predicted event footprint and pre-event OSM.

Scientific responsibility
-------------------------
Intersect the segmentation mask with pre-event OSM buildings, roads and
bridges, and report spatial relationships (architecture.md §7).

Key constraints
---------------
* A flood mask intersecting a road is evidence of spatial exposure, NOT
  evidence of physical destruction (AGENTS.md §8).
* Preserve the distinction between geometrically exposed, potentially
  disrupted, and inferred inaccessible (architecture.md §7).
* Permitted vocabulary: exposed, intersected, potentially disrupted,
  potentially inaccessible, potentially cut off. Avoid destroyed,
  structurally failed, inaccessible, isolated unless the evidence genuinely
  supports the stronger claim (AGENTS.md §8).

Status: NOT IMPLEMENTED.
"""
