"""Live data connectors — pull long-panel data straight from an external
source instead of a CSV upload. Each connector returns a plain
pandas.DataFrame in the same [metric, <dims...>, n, c] shape the graph
already expects; the investigation pipeline itself is connector-agnostic.
"""
