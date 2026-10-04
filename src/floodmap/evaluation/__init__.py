"""Independent evaluation of model and pipeline outputs.

Scientific responsibility
-------------------------
Compute segmentation metrics (IoU, Dice/F1, precision, recall), class-specific
performance where labels permit, error pattern analysis, and the post-hoc
comparison against validation-only references (architecture.md §11).

Key constraints
---------------
* Evaluation is independent of training (AGENTS.md §6).
* Report confusion and error patterns, not a single headline score.
* A strong score on an easy random split is not evidence of Himalayan
  generalisation (AGENTS.md §6).
* This is the ONLY layer permitted to read validation-only references such as
  EMSR927, and only AFTER the production prediction exists. It must not feed
  anything back into training, features, preprocessing, inference or
  threshold selection (architecture.md §11, §18).

Status: NOT IMPLEMENTED. See docs/evaluation-protocol.md.
"""
