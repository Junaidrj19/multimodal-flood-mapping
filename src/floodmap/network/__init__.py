"""Road-network graph construction and disruption analysis.

Scientific responsibility
-------------------------
Represent roads as a graph, assign each segment a state or cost from the
predicted event footprint, and recompute settlement-to-hub reachability
(architecture.md §8, §9).

Key constraints
---------------
* All graph assumptions must be explicit and documented: road segmentation,
  disruption threshold, graph construction, settlement representation,
  reference hubs, routing algorithm, and treatment of disconnected
  components (AGENTS.md §9, PRD.md §9).
* Conclusions must not depend on one arbitrary overlap threshold; threshold
  sensitivity should be testable (architecture.md §8).
* Output status vocabulary: CONNECTED, POTENTIALLY DISRUPTED,
  POTENTIALLY CUT OFF, UNKNOWN / INSUFFICIENT DATA (architecture.md §9).
* Network disconnection is an inference about modelled accessibility, not
  confirmed human isolation (architecture.md §18).

Modules
-------
* ``graph``: road pieces between junctions as an undirected multigraph.
* ``connectivity``: before/after reachability from settlements to hub groups,
  disruption-threshold sensitivity, routes and settlement statuses, driven by
  the predicted flood class.
* ``candidate_disruption`` (M6.5): the same graph and routing kernel driven by
  M6.4 candidate exposure instead, under an explicit *hypothetical* closure
  rule. Candidate intersection is radar-change evidence rather than observed
  impassability, so its outputs are named and documented as a scenario
  comparison against the baseline graph.
"""
