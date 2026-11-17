# Notes sur le cours : méthodes, conventions et écarts

Source unique : support de cours « Aide à la décision » (2026, 154 diapositives). L'application reprend les notations,
les formules et les conventions du cours, et **aucune méthode absente du cours**. Les méthodes seulement nommées dans le
cours (PROMETHEE, ELECTRE III, etc.) ne sont pas implémentées. Quand une formule du cours est ambiguë ou incomplète, le
choix retenu est indiqué au §5 (écarts E-1 à E-32). Les tests (`tests/`) reproduisent les valeurs du cours.

---

## 1. Méthodes et notions du cours

| # | Section (pages) | Méthode / notion | Catégorie | Formules clés (notation du cours) | Exemples / exercices |
|---|---|---|---|---|---|
| 1 | Introduction (1–13) | MCDM, MODM, MADM, taxonomie, choix d'une méthode, processus de décision | — (cadre) | $D_{mn}=(x_{ij})$, $W=(w_1,\dots,w_n)$ ; critères + (bénéfice) / − (coût) ; MODM : $\max/\min F(x)$ s.c. $g_i(x)\le,\ge,=0$ | Exemples qualitatifs (p. 4) : hôpital (4 villes), électrification, fournisseur (5 offres), investissement |
| 2 | Exemples d'applications (14–20) | Sélection de modèle ML, traitement médical, portefeuille, risques, finance | — (cadre) | Matrices génériques | **p. 18 : matrice Vuković et al. (2020), 18 actions × 9 critères, types et poids** ; p. 14, 16, 19 : gabarits sans chiffres ; p. 15 : 9 critères médicaux avec leur type |
| 3 | AHP, poids (23–29) | Comparaisons par paires, vecteur propre approché, cohérence | Pondération subjective | $\alpha_{ji}=1/\alpha_{ij}$ ; $s_j=\sum_i\alpha_{ij}$ ; $w_i=\frac1n\sum_j \alpha_{ij}/s_j$ ; $AW=\lambda_{max}W$ ; $\lambda_{max}=\sum_i s_i w_i$ ; $CI=\frac{\lambda_{max}-n}{n-1}$ ; $CR=CI/RI$ ; table RI (n = 3…10) ; $CR<0{,}1$ | Exercice voiture (p. 32–34) |
| 4 | AHP, classement (30–32) | Priorités locales, priorités finales, sensibilité | Classement compensatoire | Priorités finales $\sum_{j} w_j\,w_{ji}$ ; sensibilité aux poids (sans procédure) | Même exercice |
| 5 | BWM (35–43) | Best-Worst Method, modèle linéaire | Pondération subjective | $\min\xi$ s.c. $\lvert w_B-a_{Bj}w_j\rvert\le\xi$, $\lvert w_j-a_{jW}w_W\rvert\le\xi$, $\sum w_j=1$, $w_j\ge0$ ; forme $Ax\le b$ ; $2n-3$ comparaisons contre $n(n-1)/2$ | Exemple fournisseur (p. 37) + résultat (p. 42) + lignes de contraintes (p. 43) |
| 6 | Entropie (44–45) | Poids par entropie de Shannon | Pondération objective | $p_{ij}=d_{ij}/\sum_i d_{ij}$ ; $E_j=-\frac{1}{\ln m}\sum_i p_{ij}\ln p_{ij}$ ; $w_j=\frac{1-E_j}{n-\sum_k E_k}$ | Aucun |
| 7 | CRITIC (46–51) | Contraste et conflit entre critères | Pondération objective | Min-max selon le sens ; $\rho_{jk}$ (Pearson) ; $\sigma_j$ (dénominateur $m-1$) ; $C_j=\sigma_j\sum_k(1-\lvert\rho_{jk}\rvert)$ ; $w_j=C_j/\sum C_j$ | Exercice machines de découpe (p. 51), sans résultat |
| 8 | WSM (53–56) | Somme pondérée | Classement compensatoire | $A^+,A^-$ ; + : $r_{ij}=x_{ij}/x_j^+$ (= $x_{ij}/\max x_{ij}$) ; − : $r_{ij}=x_j^+/x_{ij}$ (= $\min x_{ij}/x_{ij}$) ; $Q_i^{(1)}=\sum_j w_j r_{ij}$, max | Exercice p. 58 |
| 9 | WPM (56) | Produit pondéré | Classement compensatoire | $Q_i^{(2)}=\prod_j r_{ij}^{w_j}$, max | p. 58 |
| 10 | WASPAS (57–59) | Combinaison WSM/WPM | Classement compensatoire | $Q_i=\frac12(Q_i^{(1)}+Q_i^{(2)})$ ; $Q_i=\lambda Q_i^{(1)}+(1-\lambda)Q_i^{(2)}$, $\lambda\in[0,1]$ ; ordre décroissant | Exercice 4 machines × 5 critères avec poids (p. 58), sans résultat |
| 11 | TOPSIS (60–65) | Proximité à la solution idéale | Classement compensatoire | $r_{ij}=x_{ij}/\sqrt{\sum_i x_{ij}^2}$ ; $v_{ij}=w_j r_{ij}$ ; $I^+$, $I^-$ selon $J$/$J'$ ; $S_i^\pm$ euclidiennes ; $RC_i^*=S_i^-/(S_i^-+S_i^+)$, max | Exercice voitures (p. 65), sans résultat |
| 12 | VIKOR (66–72) | Classement de compromis | Classement compensatoire | $x_j^*$, $x_j^-$ ; $S_i=\sum_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}$ ; $R_i=\max_j[\cdot]$ ; $Q_i=v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*}$, $v=0{,}5$ ; avantage acceptable $Q(A'')-Q(A')\ge\frac1{m-1}$ ; stabilité ; ensembles de compromis | Exercice 3 machines × 2 critères (p. 72), sans résultat |
| 13 | PROMETHEE (74) | — | Surclassement | **Titre seul** | PROMETHEE II n'apparaît que comme classement donné dans l'exemple de Spearman (p. 78) |
| 14 | ELECTRE III (75) | — | Surclassement | **Titre seul** | — |
| 15 | Robustesse (76–79) | Corrélation de Spearman | Robustesse | $\rho=1-\frac{6\sum d_i^2}{n(n^2-1)}$ ; interprétation −1/0/+1 ; n = 6 : $\rho\ge0{,}886$ significatif à 5 % | Rangs de 6 alternatives selon TOPSIS, VIKOR, PROMETHEE II + matrice de corrélation (p. 78–79) |
| 16 | Sensibilité (12, 32, 79) | Faire varier les poids | Robustesse | Principe sans procédure | — |
| 17 | Alternative unique (81–83) | MAUT, règle de décision | Évaluation d'une alternative unique | $S(A)=\sum_j w_j u_j(x_j)$ ; + : $\frac{x-x_j^{min}}{x_j^{max}-x_j^{min}}$ ; − : $\frac{x-x_j^{max}}{x_j^{min}-x_j^{max}}$ ; accepter si $S(A)\ge\tau$ | — |
| 18 | Choix du seuil (84–88) | Seuil fixé (politique), seuil issu des données | Évaluation d'une alternative unique | Ex. $S\ge0{,}7$ ; $TPR=\frac{TP}{TP+FN}$, $FPR=\frac{FP}{FP+TN}$ ; $d=\sqrt{FPR^2+(1-TPR)^2}$ (min) ; $J=TPR-FPR$ (max) | 6 décisions passées (p. 85) + 3 seuils testés avec résultats (p. 88) |
| 19 | Distance à l'idéal (89) | TOPSIS pour une seule alternative | Évaluation d'une alternative unique | $D^\pm=\sqrt{\sum_j(x_j-x_j^\pm)^2}$ ; $C=D^-/(D^++D^-)$ | — |
| 20 | Incertitude, logique booléenne (96–98) | Aléatoire / épistémique ; ensemble classique | Logique floue | $1_A(x)$ ; complément $1-1_A(x)$ | Adulte / mineur |
| 21 | Ensembles flous (99–104) | Degré d'appartenance, fonctions d'appartenance | Logique floue | Trapézoïdale $(a,b,c,d)$, triangulaire $(a,b,c)$, gaussienne $e^{-(x-m)^2/2\sigma^2}$ | « Âge moyen » (25, 35, 55, 65) : μ(45) = 1, μ(30) = 0,5 (p. 104) |
| 22 | Propriétés (105–107) | Hauteur, support, noyau, α-coupe, convexité, variable linguistique | Logique floue | $h(A)=\sup\mu_A$ ; $supp$ ; $noy$ ; $\alpha$-cut ; convexité ; triplet $(V,X,T_V)$ | « Average » (10, 15, 20) avec noyau, hauteur, support, 0,5-coupe (p. 106) ; pourboire faible/moyen/élevé (p. 107) |
| 23 | Nombres flous (108–109) | Trapézoïdaux, triangulaires, arithmétique | Logique floue | $A\pm B$, $A\otimes B$, $\lambda A$, $A\oslash B$ (nombres positifs) | N(4, 5, 6, 7) « moyenne » (p. 108) |
| 24 | Nombres d'intervalle (110–111) | Intervalles | Logique floue | $m(A)$, $w(A)=(a_2-a_1)/2$ ; $+,-,\times,\div$ ; $A\le B\Leftrightarrow m(A)\le m(B)$ | — |
| 25 | Flou intuitionniste (112–117) | NFI, opérations, IFWA, score, précision | Logique floue | $\pi=1-\mu-\nu$ ; $\oplus$, $\otimes$, $\lambda\alpha$, $\alpha^\lambda$ ; $IFWA_\sigma=(1-\prod(1-\mu_j)^{\sigma_j},\prod\nu_j^{\sigma_j})$ ; $S=\mu-\nu$, $H=\mu+\nu$ | Vote 5 pour / 3 contre / 2 absents (p. 112) ; α = (0,6 ; 0,3) (p. 114) |
| 26 | Pythagoricien, q-ROF, IVq-ROF (118–121) | Extensions orthopaires | Logique floue | $\mu^2+\nu^2\le1$ ; $\mu^q+\nu^q\le1$ ; $\pi$ ; version intervalle | Figure des espaces IFN/PFN/q-ROFN (p. 120) |
| 27 | Flou pictural (122–124) | Oui / abstention / non / refus | Logique floue | $\mu+\nu+\pi\le1$ ; refus $1-(\mu+\nu+\pi)$ ; $\oplus$, $\lambda A$, $A^\lambda$ | Système de vote |
| 28 | Flou hésitant (125–130) | HFE, opérations, score, déviation, comparaison, distance | Logique floue | $s(h)=\frac1l\sum\gamma$ ; $\sigma(h)=[\frac1l\sum(\gamma-s)^2]^{1/2}$ ; distance de Hamming hésitante | $h_1=(0{,}2;0{,}3;0{,}5;0{,}8)$, $h_2=(0{,}4;0{,}6;0{,}8)$, $h_3=(0{,}3;0{,}45;0{,}6)$ (p. 129), sans résultat |
| 29 | MCDM flou, alternative unique (132–134) | Matrice floue, normalisation ÷ $c_{max}$, agrégation, défuzzification | MCDM flou | $\tilde S=\sum w_i\tilde x_i$ ; $S=(a+b+c)/3$ ; largeur $c-a$ | ROA / Liquidité avec résultats (p. 134) |
| 30 | TOPSIS flou (135–142) | Chen (2000), groupe de décideurs | MCDM flou | Moyenne des K décideurs ; normalisation + : $x/x_j^{3*}$ ; − : $x_j^{1-}/x$ ; $v=r\otimes w$ ; $v^*=(1,1,1)$, $v^-=(0,0,0)$ ; distance des sommets $\sqrt{\frac13\sum(\cdot)^2}$ ; $CC_i=D^-/(D^-+D^*)$ | **Exercice recrutement : 3 candidats × 3 décideurs × 5 critères, Tableaux 1–3** (p. 135, 140–142), sans résultat |
| 31 | TOPSIS intuitionniste (143–150) | IF-TOPSIS en groupe | MCDM flou | IFWA (poids $\sigma_k$ des décideurs) ; $D'=D\otimes W$ ; $I^+$, $I^-$ selon $J_1$/$J_2$ ; $S_i^\pm=\frac{1}{2m}\sum_j\sqrt{(\Delta\mu)^2+(\Delta\nu)^2+(\Delta\pi)^2}$ ; $C_i=S_i^-/(S_i^-+S_i^+)$ | Aucun |

Les sections « MCDM et IA » (p. 90–94) et « MCDM et IA générative » (p. 151–154) du cours ne sont pas traitées dans
l'application.

## 2. Conventions du cours (respectées dans l'application)

- **Matrice** : lignes = alternatives $A_i$ ($i=1..m$), colonnes = critères $C_j$ ($j=1..n$). **Exception** : la section IF-TOPSIS (p. 143–150) utilise n = alternatives et m = critères ; cette notation est conservée sur sa page, avec un avertissement.
- **Sens des critères** : + = bénéfice (maximiser), − = coût (minimiser). Le cours note le sens au-dessus de chaque colonne ; l'application aussi.
- **Normalisations** : elles diffèrent selon la méthode, et le cours le fait explicitement :
  - linéaire par le max (WSM, WPM, WASPAS, p. 55) : + : $x/\max$ ; − : $\min/x$ ;
  - min-max (CRITIC p. 47, utilités MAUT p. 82) ;
  - vectorielle (TOPSIS p. 62), où le sens n'est traité qu'à l'étape $I^+/I^-$ ;
  - division par $c_{max}$ (flou, p. 134 et 136) ;
  - entropie : « matrice normalisée » sans précision (voir E-10).
- **Règles de classement** : le plus grand score est le meilleur pour WSM, WPM, WASPAS, TOPSIS ($RC$), MAUT ($S$), TOPSIS flou ($CC$) et IF-TOPSIS ($C$). Pour VIKOR ($S$, $R$, $Q$), le **plus petit** est le meilleur. Spearman : rang 1 = meilleure alternative.
- **Seuils et paramètres** :

  | Méthode | Seuil ou paramètre |
  |---|---|
  | AHP | $CR<0{,}1$, table RI de la p. 29 |
  | BWM | $\xi^*$ « proche de 0 » ; $a_{BB}=a_{WW}=1$ ; la plus forte note de BO porte sur W |
  | VIKOR | $v=0{,}5$ ; $DQ=1/(m-1)$ |
  | WASPAS | $\lambda\in[0,1]$ |
  | Alternative unique | $\tau\in[0,1]$ (exemple 0,7) |
  | Spearman | $\rho\ge0{,}886$ pour n = 6 à 5 % |
  | Partout | $\sum w_j=1$ |
- **AHP** : méthode approchée du cours ($\lambda_{max}=\sum s_iw_i$).

## 3. Exemples et exercices chiffrés : vérification

| Exemple | Pages | Données | Résultat donné par le cours | Recalcul | Statut |
|---|---|---|---|---|---|
| AHP voiture | 32–34 | Critères 3×3 (1, 7, 3 / 1/7, 1, 1/3 / 1/3, 3, 1) ; matrices 2×2 coût (7), confort (0,2), sécurité (1/9) | Sommes de colonnes seulement : (1,143 ; 8), (6 ; 1,2), (10 ; 1,111) | w = (0,6687 ; 0,0882 ; 0,2431) ; $\lambda_{max}$ = 3,0108 ; CI = 0,0054 ; CR = 0,0093 ; priorités locales V1 = (0,875 ; 0,167 ; 0,100) ; finales **V1 = 0,6241**, V2 = 0,3759 (identiques avec 0,143 et 0,111 arrondis) | Sommes **reproduites** ; le reste sert de référence (non publié dans le cours) |
| BWM fournisseur | 37–43 | B = C1, W = C4 ; BO = (1, 3, 4, 8) ; OW = (8, 5, 3, 1) | w = (0,552 ; 0,207 ; 0,155 ; 0,086) ; ξ* = 0,08 ; lignes de contraintes pour $w_2$ | Lignes de la p. 43 identiques. Programme linéaire du cours (solution unique) : **w = (0,5629 ; 0,2156 ; 0,1617 ; 0,0599), ξ* = 0,0838** | ξ* **reproduit** ; poids **non reproduits** → E-1 |
| CRITIC | 51 | 4×4, sens (−, −, +, −) | Aucun | w = (0,1436 ; 0,1595 ; 0,2639 ; 0,4331) | Référence |
| WASPAS | 58 | 4×5, sens (−, −, +, −, +), w = (0,331 ; 0,181 ; 0,369 ; 0,072 ; 0,047) | Aucun | $Q^{(1)}$ = (0,7443 ; 0,8310 ; 0,8746 ; 0,9110) ; $Q^{(2)}$ = (0,7289 ; 0,8037 ; 0,8630 ; 0,9060) ; Q(λ = 0,5) = (0,7366 ; 0,8174 ; 0,8688 ; 0,9085) → A4 > A3 > A2 > A1 | Référence |
| TOPSIS | 65 | 4×4, sens (+, +, +, −), w = (0,2 ; 0,1 ; 0,4 ; 0,3) | Aucun | RC = (0,4545 ; 0,5678 ; 0,3703 ; 0,5527) → M2 > M4 > M1 > M3 | Référence |
| VIKOR | 72 | 3×2, sens (−, +), poids égaux | Aucun | S = (0,5 ; 0,375 ; 0,5) ; R = (0,5 ; 0,25 ; 0,5) ; Q = (1 ; 0 ; 1) → **A2 solution de compromis**, conditions 1 (1 ≥ 0,5) et 2 vérifiées ; A1 et A3 ex æquo | Référence |
| Spearman | 78–79 | Rangs de 6 alternatives selon 3 méthodes | ρ = 0,77 ; 0,94 ; 0,60 ; seuil 0,886 | 0,7714 ; 0,9429 ; 0,6000. Seuil exact bilatéral à 5 % par loi de permutation exacte (n = 6) : **0,886** | **Reproduit** |
| Seuil ROC | 85–88 | 6 scores et décisions ; τ ∈ {0,75 ; 0,65 ; 0,60} | TPR, FPR, d, J (tableau p. 88) | Identiques | **Reproduit** (remarque E-30) |
| Flou, alternative unique | 134 | (1,0 ; 1,2 ; 1,5), (110 ; 120 ; 130) ; w = (0,6 ; 0,4) | R̃ ≈ (0,67 ; 0,80 ; 1,00), (0,85 ; 0,92 ; 1,00) ; S̃ = (0,74 ; 0,85 ; 1,00) ; S = 0,86 ; largeur 0,26 | (0,6667 ; 0,8 ; 1), (0,8462 ; 0,9231 ; 1) ; (0,7385 ; 0,8492 ; 1) ; 0,8626 ; 0,2615 | **Reproduit** |
| TOPSIS flou (Chen 2000) | 135–142 | Échelles (Tableau 1), évaluations (Tableau 2), poids (Tableau 3) | Aucun | D* = (1,9456 ; 1,2589 ; 1,6020) ; D⁻ = (3,4295 ; 4,1057 ; 3,7706) ; **CC = (0,6380 ; 0,7653 ; 0,7018)** → A2 > A3 > A1 | Référence |
| Nombres hésitants | 129 | h1, h2, h3 | Aucun | s = (0,45 ; 0,60 ; 0,45) ; σ = (0,2291 ; 0,1633 ; 0,1225) → h2 > h3 > h1 | Référence |
| Âge moyen | 104 | Trapèze (25, 35, 55, 65) | μ(45) = 1 ; μ(30) = 0,5 | Identiques | **Reproduit** |
| Average | 106 | Triangle (10, 15, 20) | Noyau {15}, hauteur 1, support ]10 ; 20[, 0,5-coupe [12,5 ; 17,5] | Identiques | **Reproduit** |
| Vote intuitionniste | 112, 114 | 5/3/2 ; (0,6 ; 0,3) | (0,5 ; 0,3 ; 0,2) ; π = 0,1 | Identiques | **Reproduit** |
| Portefeuille Vuković | 18 | 18×9, types, poids | Poids (0,0370 … 0,1089), d'origine non expliquée | CRITIC et entropie ne redonnent pas ces poids | Données transcrites ; poids utilisés « tels que publiés » → E-14 |

## 4. Méthodes seulement nommées ou annoncées sans contenu

| Élément | Page | Ce que contient le cours | Dans l'application |
|---|---|---|---|
| **PROMETHEE (II)** | 74, 78 | Titre seul ; un classement PROMETHEE II est utilisé comme donnée dans l'exemple de Spearman | **Non implémenté.** Les rangs PROMETHEE II de la p. 78 restent utilisés tels quels comme données de l'exemple de Spearman. |
| **ELECTRE III** | 75 | Titre seul | **Non implémenté.** |
| ANP | 9 | Nommé (critères dépendants) | Non implémenté ; mentionné sur l'accueil |
| SWARA | 22 | Nommé | Non implémenté |
| Poids hybrides | 22 | « On combine souvent les deux », sans formule | Non implémenté (la phrase du cours est reprise telle quelle) |
| MODM | 7 | Définition seule | Texte sur l'accueil, pas de solveur |
| « Techniques MCDM hybrides » | 3 (plan, point 6) | Annoncé, sans section dédiée | Pas de page dédiée |
| Group MCDM | 10 | Définition | Couvert par l'agrégation des décideurs (TOPSIS flou, IF-TOPSIS) |
| QUALIFLEX–TODIM | 15 | Source citée | Non implémenté |
| Procédure d'analyse de sensibilité | 12, 32 | Principe seulement : « faire varier les poids et vérifier la stabilité du classement » | Mise en œuvre directe du principe : un poids varie à la fois (les autres sont réajustés proportionnellement pour garder $\sum w_j=1$) |
| Échelle de Saaty détaillée | 24 | Seules les bornes 1 (importances égales) et 9 (extrêmement important) sont décrites | Seule la description du cours est affichée |

## 5. Écarts, ambiguïtés et choix retenus

| ID | Page | Constat | Choix retenu |
|---|---|---|---|
| **E-1** | 42 | **Les poids BWM du cours ne se reproduisent pas.** Le modèle linéaire du cours (p. 38–40) avec BO = (1, 3, 4, 8) et OW = (8, 5, 3, 1) a une solution unique : (0,5629 ; 0,2156 ; 0,1617 ; 0,0599), ξ* = 0,0838. Les poids du cours (0,552 ; 0,207 ; 0,155 ; 0,086) violent les contraintes avec un écart maximal de 0,223 (pour $\lvert w_2-5w_4\rvert$). Une recherche exhaustive sur les valeurs entières voisines de BO et OW ne redonne pas ces poids. Le modèle non linéaire de Rezaei (2015) donne (0,540 ; 0,252 ; 0,147 ; 0,061) avec ξ* = 0,86 : pas mieux. Le ξ* du cours (0,08) correspond à 0,0838 arrondi. L'ordre des critères (C1 > C2 > C3 > C4) est le même. | L'application affiche la solution du programme linéaire du cours, avec une note qui signale l'écart. Le test correspondant est en `xfail(strict=True)` avec l'explication. |
| E-2 | 119 | Hésitation q-ROF écrite $\sqrt[q]{1-\mu^q+\nu^q}$ (signe + erroné) | $\sqrt[q]{1-\mu^q-\nu^q}$, cohérent avec la p. 118 (q = 2) et la p. 121 |
| E-3 | 124 | Ensembles picturaux : $\lambda A$ et $A^\lambda$ utilisent l'exposant n au lieu de λ | Remplacer n par λ |
| E-4 | 136 | La normalisation floue des critères négatifs $(x_j^{1-}/x^1_{ij},\,x_j^{1-}/x^2_{ij},\,x_j^{1-}/x^3_{ij})$ donne un triplet décroissant, donc un nombre flou invalide | Ordre de Chen (2000) : $(x_j^{1-}/x^3_{ij},\,x_j^{1-}/x^2_{ij},\,x_j^{1-}/x^1_{ij})$. L'exercice (que des critères +) n'est pas affecté. |
| E-5 | 70–71 | « Classer par ordre décroissant de S, R, Q » alors que « plus petit = meilleur » | Tri croissant : rang 1 = plus petite valeur |
| E-6 | 67 | $x_j^*=\max[(x_{ij})\,\vert\, j=1..n]$ : l'indice devrait parcourir les alternatives | Max/min sur $i=1..m$ |
| E-7 | 70 | Stabilité : « meilleure en S **et** R » | Condition du cours appliquée telle quelle |
| E-8 | 149 | Séparations IF-TOPSIS : racine **par critère**, puis somme multipliée par 1/(2m), alors que la forme usuelle de la littérature prend la racine de la somme | Formule du cours uniquement |
| E-9 | 150 | « 1 ≤ C_i ≤ 1 » | 0 ≤ C_i ≤ 1 |
| E-10 | 45 | Entropie : la « matrice normalisée D » n'est pas précisée, ni le traitement des critères −, ni la convention 0·ln 0 | Défaut : normalisation linéaire du cours (p. 55). Au choix : les autres normalisations du cours, min-max (p. 47) et vectorielle (p. 62). Convention 0·ln 0 = 0. Message d'erreur si une valeur est négative. |
| E-11 | 29 | RI(10) = 1,56, alors que la table usuelle de Saaty donne 1,49 | Garder 1,56 (cours) |
| E-12 | 29–30 | RI n'est pas défini pour n ≤ 2, mais le cours demande CR < 0,1 pour des matrices 2×2 | n ≤ 2 : matrice toujours cohérente, CI = 0, CR affiché « sans objet » |
| E-13 | 30–31 | La légende dit « w_ij = priorité de A_i selon C_j », mais le tableau utilise $w_{ji}$ et $\sum_j w_jw_{ji}$ | Notation du tableau, légende corrigée |
| E-14 | 18 | Origine des poids Vuković non expliquée ; CRITIC et entropie ne les redonnent pas | Poids utilisés « tels que publiés », remplaçables |
| E-15 | 89 | Distance à l'idéal : formule non pondérée sur valeurs brutes (unités hétérogènes) | Calcul sur les utilités normalisées $u_j\in[0,1]$ (standardisation recommandée p. 54), donc $A^+=1$ et $A^-=0$ ; formule non pondérée, comme dans le cours |
| E-18 | 81 | Sortie « classe A/B/C, niveau de risque » sans seuils | Classe A si S ≥ 0,8 ; B si S ≥ 0,6 ; C sinon |
| E-20 | 130 | Distance de Hamming hésitante : l = max(l(h1), l(h2)), mais l'extension de l'ensemble le plus court n'est pas précisée | Extension pessimiste (ajout du minimum) |
| E-21 | 77 | La formule de Spearman suppose l'absence d'ex æquo ; VIKOR en produit (exercice p. 72) | Formule du cours appliquée aux rangs. En cas d'ex æquo, rangs moyens, avec un avertissement : la formule devient approximative. |
| E-22 | 143–150 | Inversion de notation (n = alternatives, m = critères) | Conservée sur la page IF-TOPSIS et signalée |
| E-23 | 146 | Indices $\mu_{ij}$ dans l'IFWA des poids | $\mu_j^k$, $\nu_j^k$ |
| E-24 | 147 | Le produit ⊗ entre nombres flous intuitionnistes n'est pas rappelé | Produit de la p. 115 : $(\mu_a\mu_w,\ \nu_a+\nu_w-\nu_a\nu_w)$ |
| E-25 | 51 | C1 « épaisseur maximale de la pièce » est noté −, alors qu'on l'attendrait + | Garder −. Les poids CRITIC ne dépendent pas du sens (vérifié : poids identiques) ; seul le classement ultérieur change. |
| E-27 | 110 | « a1 < a2 » puis « si a1 = a2 » | a1 ≤ a2 |
| E-28 | 41 | Les contraintes $A_{ub}x\le b_{ub}$ et $A_{eq}x=b_{eq}$ manquent dans l'encadré | Ajoutées, d'après la liste des éléments de la même page |
| E-29 | 64 | « RC = … = 0 ≤ RC ≤ 1 » | Coquille ; formule lue comme $RC\in[0,1]$ |
| E-30 | 88 | Seuls 3 seuils sont testés. Un balayage complet montre que τ ∈ ]0,69 ; 0,71] sépare parfaitement les classes (TPR = 1, FPR = 0). | Le tableau du cours est reproduit, puis la courbe ROC complète est montrée. C'est un complément, pas une correction. |
| E-31 | 82 | Bornes $x_j^{min}$, $x_j^{max}$ d'une alternative isolée non précisées | Bornes de référence saisies (politique ou réglementation), fournies avec le jeu de données |
| E-32 | 124 | Les opérations $\lambda A$ et $A^\lambda$ des ensembles picturaux peuvent produire un triplet qui viole $\mu+\nu+\pi\le1$ (ex. $A=(0{,}3 ; 0{,}3 ; 0{,}3)$, $\lambda=2$ : $A^\lambda=(0{,}09 ; 0{,}51 ; 0{,}51)$) | Formules du cours appliquées telles quelles ; résultat signalé comme invalide dans la page et vérifié par un test |

Les numéros E-16, E-17, E-19 et E-26 concernaient la partie « MCDM et IA », retirée de l'application.
