import numpy as np
import pandas as pd
import streamlit as st

from core.alternative_unique.distance_ideal import distance_ideal
from core.alternative_unique.maut import maut
from core.alternative_unique.seuil import courbe_roc, indicateurs
from core.commun import ErreurMethode
from ui import affichage as af
from ui import graphiques as gr
from ui.alternative_unique import editeur_dossier
from ui.donnees import charger_tableau, editeur_tableau

af.entete("Évaluer une alternative unique", "Évaluation absolue",
          "Une seule alternative est évaluée sur plusieurs critères : score, puis décision (accepter / rejeter, classe).")

with st.expander("Théorie", expanded=True):
    st.latex(r"S(A)=\sum_j w_j\,u_j(x_j),\qquad u_j=\frac{x-x_j^{min}}{x_j^{max}-x_j^{min}}\ (+),\qquad "
             r"u_j=\frac{x-x_j^{max}}{x_j^{min}-x_j^{max}}\ (-)")
    st.markdown("Décision : **accepter si $S(A) \\ge \\tau$**, rejeter sinon. Le seuil τ est fixé par la politique "
                "(ex. 0,7) ou déduit de décisions passées avec la courbe ROC :")
    st.latex(r"TPR=\frac{TP}{TP+FN},\quad FPR=\frac{FP}{FP+TN},\quad \min d=\sqrt{FPR^2+(1-TPR)^2},\quad \max J=TPR-FPR")
    st.markdown("Distance à l'idéal (logique de TOPSIS) : $D^\\pm=\\sqrt{\\sum_j (x_j-x_j^\\pm)^2}$, "
                "$C = D^-/(D^+ + D^-)$, calculée ici sur les utilités normalisées ($A^+ = 1$, $A^- = 0$).")

onglet1, onglet2 = st.tabs(["MAUT et distance à l'idéal", "Choix du seuil τ"])

with onglet1:
    st.caption("Exemple : un dossier de crédit et les bornes de référence fixées par la politique de crédit.")
    dossier = editeur_dossier("maut_dossier")
    if dossier is not None:
        tau = st.slider("Seuil τ", 0.0, 1.0, 0.7, 0.01)
        try:
            r = maut(dossier.valeurs, dossier.xmin, dossier.xmax, dossier.sens, dossier.poids, tau)
            dist = distance_ideal(r.u)
        except ErreurMethode as exc:
            st.error(str(exc))
        else:
            st.dataframe(pd.DataFrame({"uⱼ": np.round(r.u, 4), "wⱼ uⱼ": np.round(r.contributions, 4)}, index=dossier.criteres).T)
            c1, c2, c3 = st.columns(3)
            c1.metric("Score S(A)", f"{r.S:.4f}")
            c2.metric("Décision", "Accepter" if r.accepte else "Rejeter")
            c3.metric("Distance à l'idéal C", f"{dist.C:.4f}")
            (st.success if r.accepte else st.warning)(
                f"S(A) = {r.S:.4f} {'≥' if r.accepte else '<'} τ = {tau:.2f} : alternative "
                f"**{'acceptée' if r.accepte else 'rejetée'}** (classe {r.classe} : A si S ≥ 0,8, B si S ≥ 0,6, C sinon).")

with onglet2:
    st.caption("Exemple du cours : 6 décisions passées, trois seuils testés.")
    table = editeur_tableau("seuil_cours", charger_tableau("C8_decisions_passees_6", "tableau"),
                            colonnes_attendues=["alternative", "score", "decision"], nom_fichier="decisions.csv",
                            column_config={"decision": st.column_config.SelectboxColumn("decision", options=["Accepter", "Rejeter"])})
    try:
        S6 = pd.to_numeric(table["score"]).to_numpy(float)
        y6 = (table["decision"] == "Accepter").to_numpy(int)
        lignes = [indicateurs(S6, y6, t) for t in (0.75, 0.65, 0.60)]
    except (ValueError, ErreurMethode) as exc:
        st.error(f"Données invalides : {exc}")
        st.stop()
    st.dataframe(pd.DataFrame([{"Cas": f"Cas {k + 1}", "Seuil τ": i.tau, "TPR": round(i.TPR, 2), "FPR": round(i.FPR, 2),
                                "d": round(i.d, 2), "J": round(i.J, 2)} for k, i in enumerate(lignes)]), hide_index=True)
    st.caption("Cas 1 et 2 à égalité (d et J) : on les départage selon le coût d'un faux positif ou d'un faux négatif.")
    points = courbe_roc(S6, y6)
    gr.afficher(gr.roc([p.FPR for p in points], [p.TPR for p in points],
                       {f"τ = {i.tau:g}": (i.FPR, i.TPR) for i in lignes}, "Courbe ROC"))
