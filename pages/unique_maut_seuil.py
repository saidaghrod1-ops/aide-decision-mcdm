import numpy as np
import pandas as pd
import streamlit as st

from core.alternative_unique.maut import maut, scores_maut
from core.alternative_unique.seuil import CRITERES, courbe_roc, indicateurs, seuil_optimal
from core.commun import ErreurMethode
from core.donnees import chemin, dataset, lire_tableau_csv
from ui import affichage as af
from ui import graphiques as gr
from ui.alternative_unique import editeur_dossier
from ui.donnees import charger_tableau, editeur_tableau

af.entete("MAUT et seuil de décision", "Évaluation d'une alternative unique (évaluation absolue)", "81–88",
          "On évalue une seule alternative A sur plusieurs critères : score S(A), puis décision (accepter / rejeter, "
          "classe, niveau de risque).")

with st.expander("Théorie (cours p. 81–87)", expanded=False):
    st.latex(r"S(A)=\sum_{j=1}^{n} w_j\,u_j(x_j),\qquad \sum_j w_j = 1")
    st.markdown("Fonctions d'utilité d'exemple (p. 82) :")
    st.latex(r"u_j(x)=\frac{x-x_j^{min}}{x_j^{max}-x_j^{min}}\ \text{(critère bénéfice)},\qquad "
             r"u_j(x)=\frac{x-x_j^{max}}{x_j^{min}-x_j^{max}}\ \text{(critère coût)}")
    st.markdown("Règle de décision avec un seuil $\\tau \\in [0 ; 1]$ : accepter si $S(A) \\ge \\tau$, rejeter sinon (p. 83).  \n"
                "**Choix du seuil** : 1) fixé par la politique ou des experts (ex. τ = 0,7) ; 2) issu des données : "
                "on calcule les scores $S_i$ d'alternatives passées dont la décision $y_i$ est connue, et "
                "$\\hat y_i = 1$ si $S_i \\ge \\tau$, 0 sinon (p. 86).")
    st.latex(r"TPR=\frac{TP}{TP+FN},\qquad FPR=\frac{FP}{FP+TN},\qquad d=\sqrt{FPR^2+(1-TPR)^2},\qquad J=TPR-FPR")
    st.markdown("Chaque seuil τ donne un point (FPR, TPR) de la **courbe ROC**. Méthode 1 : minimiser la distance d au "
                "point idéal (0 ; 1). Méthode 2 : maximiser l'indice de Youden J. En cas d'égalité : départager selon le "
                "coût d'un faux positif ou d'un faux négatif (p. 88).")

onglet1, onglet2 = st.tabs(["1. Évaluer une alternative (MAUT)", "2. Choisir le seuil à partir des données"])

with onglet1:
    st.caption("Jeu créé : scoring de crédit (cas d'utilisation cité p. 83).")
    dossier = editeur_dossier("maut_dossier")
    if dossier is None:
        st.stop()
    c1, c2, c3 = st.columns(3)
    tau = c1.slider("Seuil τ (politique)", 0.0, 1.0, 0.7, 0.01)
    seuil_a = c2.number_input("Classe A si S ≥", 0.0, 1.0, 0.8, 0.05)
    seuil_b = c3.number_input("Classe B si S ≥", 0.0, 1.0, 0.6, 0.05)
    try:
        r = maut(dossier.valeurs, dossier.xmin, dossier.xmax, dossier.sens, dossier.poids, tau, (seuil_a, seuil_b))
    except ErreurMethode as exc:
        st.error(str(exc))
        st.stop()
    with af.etape("Utilités uⱼ(xⱼ) et contributions wⱼ uⱼ", ouverte=True):
        st.dataframe(pd.DataFrame({"Valeur": dossier.valeurs, "x_min": dossier.xmin, "x_max": dossier.xmax,
                                   "Sens": ["+" if s > 0 else "−" for s in dossier.sens], "uⱼ": np.round(r.u, 4),
                                   "wⱼ": dossier.poids, "wⱼ uⱼ": np.round(r.contributions, 4)}, index=dossier.criteres))
    if r.hors_bornes.any():
        st.warning("Valeur hors des bornes de référence pour : "
                   + ", ".join(c for c, h in zip(dossier.criteres, r.hors_bornes) if h) + " (utilité hors de [0, 1]).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Score S(A)", f"{r.S:.4f}")
    c2.metric("Décision", "Accepter" if r.accepte else "Rejeter", f"τ = {tau:.2f}", delta_color="off")
    c3.metric("Classe", r.classe)
    gr.afficher(gr.barres(dossier.criteres, r.contributions, "Contribution de chaque critère au score S(A)", "wⱼ uⱼ",
                          ligne=None))
    (st.success if r.accepte else st.warning)(
        f"S(A) = {r.S:.4f} {'≥' if r.accepte else '<'} τ = {tau:.2f} : l'alternative est "
        f"**{'acceptée' if r.accepte else 'rejetée'}** (classe {r.classe}, seuils A ≥ {seuil_a}, B ≥ {seuil_b} ; "
        "seuils de classe réglables, E-18).")

with onglet2:
    st.subheader("Exemple du cours : 6 décisions passées (p. 85–88)")
    table = editeur_tableau("seuil_cours", charger_tableau("C8_decisions_passees_6", "tableau"),
                            colonnes_attendues=["alternative", "score", "decision"], nom_fichier="decisions.csv",
                            column_config={"decision": st.column_config.SelectboxColumn("decision", options=["Accepter", "Rejeter"])})
    try:
        S6 = pd.to_numeric(table["score"]).to_numpy(float)
        y6 = (table["decision"] == "Accepter").to_numpy(int)
        taus_txt = st.text_input("Seuils τ testés (séparés par « ; »)", "0.75 ; 0.65 ; 0.60")
        taus = [float(t.replace(",", ".")) for t in taus_txt.split(";") if t.strip()]
        lignes = [indicateurs(S6, y6, t) for t in taus]
    except (ValueError, ErreurMethode) as exc:
        st.error(f"Données invalides : {exc}")
        st.stop()
    st.dataframe(pd.DataFrame([{"Cas": f"Cas {k + 1}", "Seuil τ": i.tau, "TP": i.TP, "FP": i.FP, "FN": i.FN, "TN": i.TN,
                                "TPR": round(i.TPR, 2), "FPR": round(i.FPR, 2), "d": round(i.d, 2), "J": round(i.J, 2)}
                               for k, i in enumerate(lignes)]), hide_index=True)
    best_d = min(i.d for i in lignes); best_j = max(i.J for i in lignes)
    st.markdown(f"Distance minimale d = {best_d:.2f} pour : " + ", ".join(f"cas {k + 1}" for k, i in enumerate(lignes) if abs(i.d - best_d) < 1e-9)
                + f" ; indice J maximal = {best_j:.2f} pour : " + ", ".join(f"cas {k + 1}" for k, i in enumerate(lignes) if abs(i.J - best_j) < 1e-9)
                + ". En cas d'égalité, on départage selon le coût d'un faux positif ou d'un faux négatif (p. 88).")
    points6 = courbe_roc(S6, y6)
    opt6, _ = seuil_optimal(S6, y6, "youden")
    gr.afficher(gr.roc([p.FPR for p in points6], [p.TPR for p in points6],
                       {**{f"τ = {i.tau:g}": (i.FPR, i.TPR) for i in lignes}}, "Courbe ROC des 6 décisions passées"))
    plus_bas = S6[S6 < opt6.tau - 1e-12]
    intervalle = f"]{plus_bas.max():g} ; {opt6.tau:g}]" if len(plus_bas) else f"]−∞ ; {opt6.tau:g}]"
    st.info(f"Complément (E-30) : en testant tous les seuils, le meilleur indice J = {opt6.J:.2f} est atteint pour "
            f"τ ∈ {intervalle} (TPR = {opt6.TPR:.2f}, FPR = {opt6.FPR:.2f})"
            + (" : les décisions passées sont séparées parfaitement." if opt6.J == 1 else "."))

    st.subheader("Historique de 200 dossiers de crédit (jeu créé)")
    d = dataset("D5_credit")
    entete, lignes_h = lire_tableau_csv(chemin(d["fichiers"]["historique"]))
    X = np.array([[float(v) for v in l[1:6]] for l in lignes_h])
    y = np.array([l[6] == "Accepter" for l in lignes_h], int)
    ref = editeur_dossier("seuil_dossier_ref")
    if ref is None:
        st.stop()
    S = scores_maut(X, ref.xmin, ref.xmax, ref.sens, ref.poids)
    c1, c2 = st.columns(2)
    c_fp = c1.number_input("Coût d'un faux positif (mauvais dossier accepté)", 0.0, 100.0, 5.0, 0.5)
    c_fn = c2.number_input("Coût d'un faux négatif (bon dossier rejeté)", 0.0, 100.0, 1.0, 0.5)
    optima = {cle: seuil_optimal(S, y, cle, c_fp, c_fn)[0] for cle in CRITERES}
    st.dataframe(pd.DataFrame([{"Critère de choix": CRITERES[k][0], "τ optimal": round(i.tau, 4), "TPR": round(i.TPR, 3),
                                "FPR": round(i.FPR, 3), "d": round(i.d, 3), "J": round(i.J, 3),
                                "Exactitude": round(i.exactitude, 3), "Coût": i.cout} for k, i in optima.items()]),
                 hide_index=True)
    points = courbe_roc(S, y, c_fp, c_fn)
    gr.afficher(gr.roc([p.FPR for p in points], [p.TPR for p in points],
                       {CRITERES[k][0].split(" (")[0]: (i.FPR, i.TPR) for k, i in optima.items()},
                       "Courbe ROC de l'historique et seuils optimaux"))
    tau_h = st.slider("Seuil τ appliqué à l'historique", float(S.min()), float(S.max()), float(optima["youden"].tau), 0.005)
    ind = indicateurs(S, y, tau_h, c_fp, c_fn)
    gr.afficher(gr.histogrammes({"Décision réelle : accepter": S[y == 1], "Décision réelle : rejeter": S[y == 0]},
                                "Scores S des dossiers passés", "S", ligne_x=tau_h))
    st.dataframe(pd.DataFrame([[ind.TP, ind.FN], [ind.FP, ind.TN]], index=["Réel : accepter", "Réel : rejeter"],
                              columns=["Prédit : accepter", "Prédit : rejeter"]))
    st.markdown(f"À τ = {tau_h:.3f} : TPR = {ind.TPR:.3f}, FPR = {ind.FPR:.3f}, exactitude = {ind.exactitude:.3f}, "
                f"coût = {ind.cout:g}. Le seuil « politique » τ = 0,7 accepterait {int((S >= 0.7).sum())} dossiers sur {len(S)}.")
