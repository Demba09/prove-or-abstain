"""
attribution.py — TA décomposition rate/mix/interaction.

Signature identique à attribution_reference.decompose().
"""
import numpy as np
import pandas as pd


def decompose(base: pd.DataFrame, curr: pd.DataFrame, dims, n_col="n", c_col="c") -> pd.DataFrame:
    """
    base, curr : DataFrames avec colonnes [*dims, n_col, c_col].
    
    Retourne un DataFrame indexé par segment avec :
    - w0, w1 : parts de mix (baseline, current)
    - r0, r1 : taux (baseline, current)
    - rate, mix, interaction : décomposition exacte
    - contribution : leur somme
    
    Identité : w1*r1 - w0*r0 = w0*dr + r0*dw + dw*dr
    """
    # ÉTAPE 1 : normaliser dims en liste
    if isinstance(dims, str):
        dims = [dims]
    
    # ÉTAPE 2 : indexer par segment(s)
    b = base.set_index(dims)
    c = curr.set_index(dims)
    
    # ÉTAPE 3 : fusionner les indices (segments dans base OU curr)
    idx = b.index.union(c.index)
    b = b.reindex(idx, fill_value=0)
    c = c.reindex(idx, fill_value=0)
    
    # ÉTAPE 4 : calcul des totaux
    N0 = b[n_col].sum()   # total baseline
    N1 = c[n_col].sum()   # total current
    
    # ÉTAPE 5 : parts de mix w0, w1
    w0 = b[n_col] / N0
    w1 = c[n_col] / N1
    
    # ÉTAPE 6 : taux r0, r1 (avec gestion division par zéro)
    r0 = (b[c_col] / b[n_col]).replace([np.inf, -np.inf], 0).fillna(0)
    r1 = (c[c_col] / c[n_col]).replace([np.inf, -np.inf], 0).fillna(0)
    
    # ÉTAPE 7 : changements différentiels
    dw = w1 - w0  # changement de mix
    dr = r1 - r0  # changement de taux
    
    # ÉTAPE 8 : décomposition exacte
    rate = w0 * dr          # mix figé au baseline
    mix = r0 * dw           # taux figé au baseline
    interaction = dw * dr   # effet combiné
    
    # ÉTAPE 9 : construire le DataFrame de sortie
    # (n0/n1/c0/c1 : comptes bruts, nécessaires au test z des gates)
    out = pd.DataFrame({
        "n0": b[n_col], "n1": c[n_col],
        "c0": b[c_col], "c1": c[c_col],
        "w0": w0, "w1": w1,
        "r0": r0, "r1": r1,
        "rate": rate,
        "mix": mix,
        "interaction": interaction,
    })

    # ÉTAPE 10 : contribution totale (somme des trois)
    out["contribution"] = out[["rate", "mix", "interaction"]].sum(axis=1)

    return out


def decompose_sum(base: pd.DataFrame, curr: pd.DataFrame, dims,
                  n_col="n", c_col="c") -> pd.DataFrame:
    """
    Décomposition volume/taux pour une métrique SOMME V = Σₛ nₛ·rₛ
    (ex. revenu : n = clients, c = montant total, r = panier moyen).

    Même algèbre que decompose(), mais avec les effectifs bruts à la place
    des parts de mix :
        ΔVₛ = n₀·Δr (rate) + r₀·Δn (mix/volume) + Δn·Δr (interaction)
    contributionₛ = c₁ₛ - c₀ₛ, et Σ contributions = ΔV exactement.

    w0/w1 sont posés aux effectifs bruts pour que metrics.aggregate()
    (R = Σ w·r) recompose V0/V1 sans changement.
    """
    if isinstance(dims, str):
        dims = [dims]

    b = base.set_index(dims)
    c = curr.set_index(dims)
    idx = b.index.union(c.index)
    b = b.reindex(idx, fill_value=0)
    c = c.reindex(idx, fill_value=0)

    r0 = (b[c_col] / b[n_col]).replace([np.inf, -np.inf], 0).fillna(0)
    r1 = (c[c_col] / c[n_col]).replace([np.inf, -np.inf], 0).fillna(0)
    dn = c[n_col] - b[n_col]
    dr = r1 - r0

    out = pd.DataFrame({
        "n0": b[n_col], "n1": c[n_col],
        "c0": b[c_col], "c1": c[c_col],
        "w0": b[n_col], "w1": c[n_col],
        "r0": r0, "r1": r1,
        "rate": b[n_col] * dr,
        "mix": r0 * dn,
        "interaction": dn * dr,
    })
    out["contribution"] = out[["rate", "mix", "interaction"]].sum(axis=1)
    return out
