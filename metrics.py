"""
metrics.py — agrégation d'une décomposition.

aggregate() recompose les totaux R0/R1, somme les contributions et expose
le résidu (doit être ~0). Sort la logique du fichier 'reference' (oracle)
pour que le code de production n'en dépende pas.
"""
from __future__ import annotations
import pandas as pd


def aggregate(out: pd.DataFrame) -> dict:
    R0 = float((out.w0 * out.r0).sum())
    R1 = float((out.w1 * out.r1).sum())
    return {
        "R0": R0, "R1": R1, "delta_R": R1 - R0,
        "rate": float(out.rate.sum()),
        "mix": float(out.mix.sum()),
        "interaction": float(out.interaction.sum()),
        "sum_contrib": float(out.contribution.sum()),
        "residual": (R1 - R0) - float(out.contribution.sum()),
    }
