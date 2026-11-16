"""Fabriques de graphiques Plotly (palette catégorielle validée, axes et grilles discrets)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Palette catégorielle de référence (ordre fixe, jamais recyclée), rampe séquentielle bleue,
# palette divergente orange / gris / bleu.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
GRILLE = "#e4e3df"
SEQUENTIELLE = [[0.0, "#eef4fc"], [0.5, "#7fb0ea"], [1.0, "#1d4f8c"]]
DIVERGENTE = [[0.0, "#eb6834"], [0.5, "#efeeea"], [1.0, "#2a78d6"]]


def couleur(i: int) -> str:
    return PALETTE[i % len(PALETTE)] if i < len(PALETTE) else "#8a8984"


def _forme(fig: go.Figure, titre: str | None = None, axe_x: str | None = None, axe_y: str | None = None,
           hauteur: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=titre, font=dict(size=15, color=ENCRE)) if titre else None,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=hauteur,
        font=dict(color=ENCRE_2, size=12), margin=dict(l=10, r=10, t=50 if titre else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(title=axe_x, gridcolor=GRILLE, zerolinecolor=GRILLE, linecolor=GRILLE)
    fig.update_yaxes(title=axe_y, gridcolor=GRILLE, zerolinecolor=GRILLE, linecolor=GRILLE)
    return fig


def afficher(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


def barres(etiquettes, valeurs, titre: str | None = None, axe: str | None = None, horizontal: bool = True,
           decimales: int = 4, couleur_unique: str = PALETTE[0], ligne: tuple[float, str] | None = None) -> go.Figure:
    etiquettes, valeurs = list(etiquettes), np.asarray(valeurs, float)
    texte = [f"{v:.{decimales}f}" for v in valeurs]
    if horizontal:
        fig = go.Figure(go.Bar(y=etiquettes, x=valeurs, orientation="h", marker_color=couleur_unique, text=texte,
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y} : %{x:." + str(decimales) + "f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        if ligne:
            fig.add_vline(x=ligne[0], line_dash="dash", line_color=ENCRE_2, annotation_text=ligne[1])
        return _forme(fig, titre, axe, None, hauteur=max(220, 60 + 34 * len(etiquettes)))
    fig = go.Figure(go.Bar(x=etiquettes, y=valeurs, marker_color=couleur_unique, text=texte, textposition="outside",
                           cliponaxis=False, hovertemplate="%{x} : %{y:." + str(decimales) + "f}<extra></extra>"))
    if ligne:
        fig.add_hline(y=ligne[0], line_dash="dash", line_color=ENCRE_2, annotation_text=ligne[1])
    return _forme(fig, titre, None, axe)


def barres_groupees(etiquettes, series: dict, titre: str | None = None, axe: str | None = None,
                    decimales: int = 4) -> go.Figure:
    fig = go.Figure()
    for i, (nom, valeurs) in enumerate(series.items()):
        fig.add_trace(go.Bar(name=nom, x=list(etiquettes), y=np.asarray(valeurs, float), marker_color=couleur(i),
                             hovertemplate=nom + " — %{x} : %{y:." + str(decimales) + "f}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    return _forme(fig, titre, None, axe)


def barres_empilees(etiquettes, series: dict, titre: str | None = None, axe: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (nom, valeurs) in enumerate(series.items()):
        fig.add_trace(go.Bar(name=nom, y=list(etiquettes), x=np.asarray(valeurs, float), orientation="h",
                             marker=dict(color=couleur(i), line=dict(color=SURFACE, width=2)),
                             hovertemplate=nom + " — %{y} : %{x:.4f}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    return _forme(fig, titre, axe, None, hauteur=max(240, 80 + 36 * len(list(etiquettes))))


def heatmap(Z, x, y, titre: str | None = None, divergente: bool = False, zmin=None, zmax=None,
            decimales: int = 3) -> go.Figure:
    Z = np.asarray(Z, float)
    fig = go.Figure(go.Heatmap(
        z=Z, x=list(x), y=list(y), colorscale=DIVERGENTE if divergente else SEQUENTIELLE,
        zmin=zmin, zmax=zmax, text=np.vectorize(lambda v: f"{v:.{decimales}f}")(Z), texttemplate="%{text}",
        hovertemplate="%{y} / %{x} : %{z:." + str(decimales) + "f}<extra></extra>", xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed")
    return _forme(fig, titre, hauteur=max(260, 80 + 40 * len(list(y))))


def courbes(x, series: dict, titre: str | None = None, axe_x: str | None = None, axe_y: str | None = None,
            marqueur_x: float | None = None, escalier: bool = False, hauteur: int = 380) -> go.Figure:
    fig = go.Figure()
    for i, (nom, valeurs) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=np.asarray(x), y=np.asarray(valeurs, float), name=nom, mode="lines",
                                 line=dict(color=couleur(i), width=2, shape="hv" if escalier else "linear"),
                                 hovertemplate=nom + " : %{y:.4f}<extra></extra>"))
    if marqueur_x is not None:
        fig.add_vline(x=marqueur_x, line_dash="dot", line_color=ENCRE_2)
    fig.update_layout(hovermode="x unified")
    return _forme(fig, titre, axe_x, axe_y, hauteur)


def nuage(x, y, etiquettes, titre: str | None = None, axe_x: str | None = None, axe_y: str | None = None,
          couleurs=None, echelle_couleur: str | None = None) -> go.Figure:
    marker = dict(size=12, line=dict(color=SURFACE, width=2))
    if couleurs is not None:
        marker.update(color=np.asarray(couleurs, float), colorscale=SEQUENTIELLE, showscale=True,
                      colorbar=dict(title=echelle_couleur or ""), reversescale=True)
    else:
        marker["color"] = PALETTE[0]
    fig = go.Figure(go.Scatter(x=np.asarray(x, float), y=np.asarray(y, float), mode="markers+text",
                               text=list(etiquettes), textposition="top center", marker=marker,
                               hovertemplate="%{text}<br>" + (axe_x or "x") + " : %{x:.4f}<br>" + (axe_y or "y")
                               + " : %{y:.4f}<extra></extra>"))
    return _forme(fig, titre, axe_x, axe_y)


def rangs_par_methode(alternatives, rangs: dict, titre: str | None = None) -> go.Figure:
    """Graphique des rangs (une ligne par alternative, une colonne par méthode)."""
    methodes = list(rangs)
    fig = go.Figure()
    for i, a in enumerate(alternatives):
        fig.add_trace(go.Scatter(x=methodes, y=[rangs[m][i] for m in methodes], name=str(a), mode="lines+markers",
                                 line=dict(color=couleur(i), width=2), marker=dict(size=10),
                                 hovertemplate=str(a) + " — %{x} : rang %{y}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", dtick=1, title="Rang (1 = meilleure)")
    return _forme(fig, titre, hauteur=420)


def histogrammes(series: dict, titre: str | None = None, axe_x: str | None = None, ligne_x: float | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (nom, valeurs) in enumerate(series.items()):
        fig.add_trace(go.Histogram(x=np.asarray(valeurs, float), name=nom, opacity=0.65, nbinsx=25,
                                   marker=dict(color=couleur(i), line=dict(color=SURFACE, width=1))))
    fig.update_layout(barmode="overlay")
    if ligne_x is not None:
        fig.add_vline(x=ligne_x, line_dash="dash", line_color=ENCRE_2, annotation_text=f"τ = {ligne_x:.3f}")
    return _forme(fig, titre, axe_x, "effectif")


def roc(fpr, tpr, points: dict | None = None, titre: str | None = "Courbe ROC") -> go.Figure:
    """Courbe ROC (TPR en fonction du FPR) ; points = {nom: (FPR, TPR)} mis en évidence."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard", line=dict(color=GRILLE, dash="dot"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.asarray(fpr), y=np.asarray(tpr), mode="lines+markers", name="Seuils testés",
                             line=dict(color=PALETTE[0], width=2, shape="hv"), marker=dict(size=6),
                             hovertemplate="FPR = %{x:.3f}<br>TPR = %{y:.3f}<extra></extra>"))
    for i, (nom, (x, y)) in enumerate((points or {}).items()):
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", name=nom, text=[nom], textposition="bottom right",
                                 marker=dict(size=13, color=couleur(i + 1), line=dict(color=SURFACE, width=2)),
                                 hovertemplate=nom + " : FPR = %{x:.3f}, TPR = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0], y=[1], mode="markers", name="Point idéal (0 ; 1)",
                             marker=dict(size=12, symbol="star", color=ENCRE)))
    fig.update_xaxes(range=[-0.02, 1.02], title="FPR (faux positifs)")
    fig.update_yaxes(range=[-0.02, 1.05], title="TPR (vrais positifs)", scaleanchor="x", scaleratio=1)
    return _forme(fig, titre, hauteur=470)


def fonctions_appartenance(x, series: dict, titre: str | None = None, axe_x: str | None = None,
                           zone: tuple[float, float] | None = None, niveau: float | None = None) -> go.Figure:
    fig = courbes(x, series, titre, axe_x, "μ(x)")
    fig.update_yaxes(range=[-0.05, 1.08])
    if zone is not None:
        fig.add_vrect(x0=zone[0], x1=zone[1], fillcolor=PALETTE[3], opacity=0.15, line_width=0)
    if niveau is not None:
        fig.add_hline(y=niveau, line_dash="dash", line_color=ENCRE_2)
    fig.update_layout(hovermode="x unified")
    return fig


def nombres_flous(series: dict, titre: str | None = None, axe_x: str | None = None) -> go.Figure:
    """Trace des nombres flous (a, b, c) ou (a, b, c, d) sous forme de triangles / trapèzes."""
    fig = go.Figure()
    for i, (nom, p) in enumerate(series.items()):
        p = [float(v) for v in p]
        xs, ys = (p[:3], [0, 1, 0]) if len(p) == 3 else (p[:4], [0, 1, 1, 0])
        fig.add_trace(go.Scatter(x=xs, y=ys, name=nom, mode="lines+markers", line=dict(color=couleur(i), width=2),
                                 hovertemplate=nom + " : x = %{x:.4f}, μ = %{y}<extra></extra>"))
    fig.update_yaxes(range=[-0.05, 1.1], title="μ(x)")
    return _forme(fig, titre, axe_x)


def plan_mu_nu(points: dict, frontieres: dict | None = None, titre: str | None = None) -> go.Figure:
    """Plan (μ, ν) avec frontières μ^q + ν^q = 1."""
    fig = go.Figure()
    for i, (nom, (mus, nus)) in enumerate((frontieres or {}).items()):
        fig.add_trace(go.Scatter(x=mus, y=nus, name=nom, mode="lines", line=dict(color=couleur(i), width=2),
                                 hoverinfo="skip"))
    decal = len(frontieres or {})
    for i, (nom, (mu, nu)) in enumerate(points.items()):
        fig.add_trace(go.Scatter(x=np.atleast_1d(mu), y=np.atleast_1d(nu), name=nom, mode="markers",
                                 marker=dict(size=11, color=couleur(decal + i), line=dict(color=SURFACE, width=2)),
                                 hovertemplate=nom + " : μ = %{x:.4f}, ν = %{y:.4f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.05], title="μ (appartenance)")
    fig.update_yaxes(range=[0, 1.05], title="ν (non-appartenance)", scaleanchor="x", scaleratio=1)
    return _forme(fig, titre, hauteur=460)
