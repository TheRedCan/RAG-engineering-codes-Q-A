"""Shared infrastructure used by every pipeline stage.

Modules here MUST NOT import from any pipeline-stage package (ingest, retrieval,
generation, eval, app). The dependency direction is one-way: stages depend on
common; common depends on nothing project-internal.
"""
