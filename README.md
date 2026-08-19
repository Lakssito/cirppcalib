# Autocalibration CIR++ sur SOFR — spécification complète, étape par étape

Ce document décrit EXACTEMENT la procédure implémentée dans ce projet pour
autocalibrer un modèle CIR++ (Brigo & Mercurio) sur (1) l'historique complet
du SOFR overnight depuis 2018 et (2) les quotes de marché du jour. Il est
autonome : toutes les formules, conventions, algorithmes et seuils de
validation sont donnés, ainsi que les résultats numériques attendus sur les
données de référence. Un développeur (ou un LLM) peut reproduire le module
intégralement à partir de ce seul document.

Périmètre : CALIBRATION UNIQUEMENT. Pas de Monte Carlo, pas de pricing
d'instruments. Les livrables sont les paramètres (kappa, theta, sigma, x0) et
une fonction phi(t) évaluable en tout t, sérialisés en JSON pour réutilisation
dans un framework de simulation externe.

Stack : Python, numpy / scipy / pandas / matplotlib uniquement. Pas de QuantLib.

---

## 0. Le modèle

CIR++ (Brigo & Mercurio, "Interest Rate Models", ch. 3.9) :

    r(t) = x(t) + phi(t)
    dx(t) = kappa (theta - x(t)) dt + sigma sqrt(x(t)) dW(t),   x(0) = x0

- x est un processus CIR classique, calibré sur la DYNAMIQUE historique.
- phi est une fonction déterministe du temps qui absorbe EXACTEMENT la courbe
  de taux du jour (fit parfait des discount factors de marché).

Formules fermées CIR utilisées (h = sqrt(kappa^2 + 2 sigma^2)) :

    B(T) = 2 (e^{hT} - 1) / (2h + (kappa + h)(e^{hT} - 1))
    A(T) = [ 2h e^{(kappa+h)T/2} / (2h + (kappa + h)(e^{hT} - 1)) ]^{2 kappa theta / sigma^2}
    P_CIR(0,T) = A(T) e^{-B(T) x0}

Forward instantané CIR (B&M eq. 3.77), avec D(t) = 2h + (kappa+h)(e^{ht}-1) :

    f_CIR(0,t) = 2 kappa theta (e^{ht} - 1) / D(t)  +  x0 · 4 h^2 e^{ht} / D(t)^2

Notes d'implémentation : calculer A en log (log_num - log_den puis exp) pour
la stabilité aux grands T ; utiliser expm1(h·tau) pour e^{hT}-1 ; vérifier
les limites B(0)=0, A(0)=1, f_CIR(0,0)=x0.

Loi stationnaire du CIR (pour le diagnostic graphique) :
Gamma(shape = 2 kappa theta / sigma^2, scale = sigma^2 / (2 kappa)).

---

## 1. Données d'entrée (formats exacts)

### 1.1 `SOFR.csv` — historique SOFR overnight (source FRED)

    observation_date,SOFR
    2018-04-03,1.83
    2018-04-04,1.74
    ...
    2026-08-17,3.66

- Taux en POURCENT. 2186 lignes, dont 94 lignes vides (jours fériés, champ
  taux vide) → 2091 fixings exploitables après suppression des NaN.
- Période couverte : 2018-04-03 → 2026-08-17. Contient la période ZIRP
  2020-2022 avec des fixings à 0.01% (quasi nuls).

### 1.2 `swapsofrrates.txt` — quotes du jour

    ON : 3.62%
    1M : 3.80%
    3M : 4.00%
    6M : 3.83%
    1Y : 3.908%
    2Y : 3.961%
    3Y : 3.980%
    5Y : 4.025%
    7Y : 4.103%
    10Y : 4.228%
    15Y : 4.411%
    30Y : 4.483%

- `ON` = fixing SOFR spot overnight (3.62%). Utilisé comme x0 ET comme ancre
  du très court terme de la courbe.
- Ténors en `M` = OIS monétaires à paiement unique à maturité.
- Ténors en `Y` = par swap rates OIS SOFR, jambe fixe ANNUELLE.
- Parser tolérant : accepter `~`, virgules décimales (`3,8%`), espaces.
  Regex : `(ON|\d+(?:[.,]\d+)?\s*[MY])\s*:\s*~?\s*([\d.,]+)\s*%`.
- Tout est en pourcent → diviser par 100.

### 1.3 Conventions

- Axe temps du modèle : années ACT/365F (t = jours calendaires / 365).
- Jambe fixe des OIS : accruals ACT/360 entre dates de paiement.
- Date de valorisation : aujourd'hui (2026-08-18 pour les résultats de
  référence), configurable.
- Pas d'ajustement business-day (pas de calendrier de jours fériés) : les
  dates de paiement sont val_date + i ans (ou + m mois) calendaires.

---

## 2. ÉTAPE 1 — Calibration CIR (kappa, theta, sigma) par MLE exact sur l'historique

### 2.1 Préprocessing

1. Charger `SOFR.csv`, supprimer les lignes sans taux (jours fériés).
2. Convertir % → décimal (1.83 → 0.0183).
3. Appliquer un floor eps = 1e-4 (0.01%) : `x = max(x, eps)`. Le CIR exige
   x > 0 ; la période 2020-2022 contient des fixings à exactement 0.01%.
4. Construire les pas de temps réels entre fixings consécutifs :
   dt_i = (date_{i+1} - date_i) en jours / 365. (Les week-ends donnent
   dt = 3/365, les jours ouvrés consécutifs 1/365.)

### 2.2 Initialisation : régression AR(1) (pseudo-MLE gaussien)

Régression OLS x_{i+1} = a + b x_i + e_i (numpy.polyfit degré 1), puis :

    kappa_0 = -ln(b) / dt_moyen          (clipper b dans (1e-6, 1-1e-6))
    theta_0 = a / (1 - b)
    sigma_0^2 = mean( e_i^2 / (x_i · dt_i) )   car Var[e_i] ≈ sigma^2 x_i dt_i

Clipper dans des bornes raisonnables : kappa dans [1e-3, 50], theta dans
[1e-5, 0.5], sigma dans [1e-4, 5].

### 2.3 MLE exact via la densité de transition chi-2 non centrale

La transition exacte du CIR : sachant x_i, avec e = e^{-kappa·dt_i},

    c  = 2 kappa / (sigma^2 (1 - e))
    df = 4 kappa theta / sigma^2          (degrés de liberté)
    nc = 2 c x_i e                        (paramètre de non-centralité)
    alors  2 c x_{i+1} ~ chi2 non centrale(df, nc)

Log-vraisemblance (scipy.stats.ncx2) :

    LL = somme_i [ ln(2 c_i) + ncx2.logpdf(2 c_i x_{i+1}, df, nc_i) ]

(le terme ln(2c) est le jacobien du changement de variable x → 2cx ; les
dt_i variables sont pris en compte individuellement.)

Optimisation : scipy.optimize.minimize, méthode Nelder-Mead, sur les
LOG-paramètres (ln kappa, ln theta, ln sigma) pour garantir la positivité,
point de départ = init AR(1), xatol=fatol=1e-8, maxiter=5000. Retourner 1e12
si LL non finie.

### 2.4 x0 et condition de Feller

- x0 = fixing SOFR spot du jour fourni dans `swapsofrrates.txt` (ligne `ON`),
  soit 0.0362. (À défaut de ligne ON : dernier fixing du CSV.)
- Afficher la condition de Feller 2·kappa·theta >= sigma^2 et AVERTIR
  clairement si elle est violée (le processus peut alors toucher 0).

### 2.5 Résultats attendus sur les données de référence (2091 obs, full 2018→2026)

    kappa  = 0.247161
    theta  = 0.035598
    sigma  = 0.130239
    x0     = 0.036200
    log-vraisemblance = 11916.27
    init AR(1) : kappa=0.3684, theta=0.03264, sigma=0.14266
    Feller : 2kt = 1.7597e-02 >= s2 = 1.6962e-02  [OK, de justesse]

---

## 3. ÉTAPE 2a — Bootstrap des discount factors P_market(0,T)

Single curve SOFR : la même courbe sert au discount et à la projection.

### 3.1 Échéanciers

Pour un ténor N années (N >= 1) : paiements fixes annuels aux dates
val_date + 1an, ..., val_date + N ans.
Pour un ténor m mois (< 1 an) : paiement unique à val_date + m mois.
Pour chaque date de paiement d_i :

    t_i   = (d_i - val_date).days / 365        (temps de discount, ACT/365F)
    tau_i = (d_i - d_{i-1}).days / 360         (accrual jambe fixe, ACT/360)

### 3.2 Relation de par

    S = (1 - P(0, T_N)) / somme_i [ tau_i · P(0, t_i) ]

(valable aussi pour les ténors monétaires : un seul terme dans l'annuité,
ce qui équivaut à P = 1/(1 + S·tau).)

### 3.3 Ancre overnight

Si le fixing spot est fourni : ajouter un pilier en t = 1/365 avec

    P(0, 1j) = 1 / (1 + spot/360)

(implémenter en log : -log1p(spot/360).) Cela ancre le forward instantané en
0 sur le fixing du jour (≈ spot·365/360 en composé continu).

### 3.4 Bootstrap séquentiel

Interpolation : LOG-LINÉAIRE sur les DF (ln P linéaire par morceaux en t →
forwards instantanés constants par morceaux), extrapolation flat-forward
au-delà du dernier pilier.

Piliers traités par ténor croissant : 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y,
15Y, 30Y. Pour chaque ténor, résoudre en 1D (scipy.optimize.brentq,
bornes [1e-8, 1.5], xtol=1e-16, rtol=1e-15) le DF du pilier P(0,T_N) tel que

    S_quote · annuité(P) - (1 - P(0,T_N)) = 0

où les DF des dates de paiement intermédiaires non encore bootstrappées
(ex : paiement 4Y du swap 5Y) sont interpolés log-linéairement entre le
dernier pilier connu et le pilier candidat courant.

### 3.5 Piliers attendus (val_date = 2026-08-18)

    O/N  T=0.0027397260   P=0.9998994546   zero=3.6701%
    1M   T=0.0849315068   P=0.9967384503   zero=3.8465%
    3M   T=0.2520547945   P=0.9898812143   zero=4.0350%
    6M   T=0.5041095890   P=0.9808002894   zero=3.8457%
    1Y   T=1.0000000000   P=0.9618873512   zero=3.8858%
    2Y   T=2.0027397260   P=0.9241546102   zero=3.9384%
    3Y   T=3.0027397260   P=0.8879591780   zero=3.9574%
    5Y   T=5.0027397260   P=0.8184996063   zero=4.0035%
    7Y   T=7.0054794521   P=0.7510358529   zero=4.0868%
    10Y  T=10.0082191781  P=0.6551328626   zero=4.2257%
    15Y  T=15.0109589041  P=0.5132894158   zero=4.4429%
    30Y  T=30.0219178082  P=0.2579262492   zero=4.5136%

(zero rate = -ln P / T, composé continu.)

---

## 4. ÉTAPE 2b — Construction de phi(t)

Définition : phi(t) = f_market(0,t) - f_CIR(0,t). DEUX représentations
cohérentes, c'est le point clé de l'implémentation :

### 4.1 phi ponctuel (la fonction évaluable, pour la simulation)

1. Grille fine : t de 0 à T_max (= dernier ténor + 1 an) au pas 1/365.
2. f_market brut = dérivée numérique (numpy.gradient) de -ln P_market(0,t)
   sur la grille. Comme ln P est linéaire par morceaux, ce forward est
   constant par morceaux avec des sauts aux piliers.
3. Lissage : scipy.ndimage.gaussian_filter1d, écart-type 10 jours de grille
   (paramètre configurable), mode='nearest'.
4. phi_grid = f_market_lissé - f_CIR(grille) (formule fermée 3.77).
5. phi(t) = interpolation linéaire dans phi_grid (np.interp).

### 4.2 Intégrale de phi EXACTE (pour le repricing)

Ne PAS intégrer numériquement le phi lissé. Utiliser l'identité fermée :

    Phi(t) := int_0^t phi(s) ds = ln P_CIR(0,t) - ln P_market(0,t)

d'où le ZCB CIR++ :

    P_CIR++(0,T) = e^{-Phi(T)} · P_CIR(0,T) = P_market(0,T)   exactement.

C'est ce qui garantit le repricing à la précision machine (les seuils du §5
sont inatteignables en intégrant numériquement un forward lissé).

### 4.3 Valeurs de contrôle de phi (paramètres du §2.5, lissage 10j)

    phi(0)  = +13.4 bp   → r(0) = x0 + phi(0) = 3.7542%
    phi(6M) = +0.1692%
    phi(1Y) = +0.3759%
    phi(5Y) = +0.8309%
    phi(10Y)= +1.4646%

phi est positif et croissant : le CIR full-period a un theta bas (3.56%,
tiré par la période ZIRP) et une mean-reversion lente, son forward reste
sous la courbe de marché. NB : phi(0) ≠ 0 car le forward marché lissé en 0
capte le segment 1M (~3.85%) et pas seulement le segment O/N (1 jour de
large) ; avec un lissage ~0 on retrouve phi(0) ≈ spot·365/360 - x0 ≈ +5 bp.

---

## 5. Vérification OBLIGATOIRE du repricing (fait échouer le run sinon)

Pour chaque ténor quoté, afficher un tableau avec :

1. |P_CIR++(0,T) - P_market(0,T)| aux piliers — seuil : < 1e-10.
   (P_CIR++ calculé via e^{-Phi(T)}·P_CIR(0,T), Phi exacte du §4.2.)
2. Par swap rate recalculé avec les DF DU MODÈLE (formule du §3.2, mêmes
   échéanciers) vs quote d'entrée — seuil : < 0.1 bp.

Résultats attendus : erreur ZCB = 0.0 (précision machine) sur les 11 piliers ;
erreur swap max = 5.7e-12 bp (sur le 1M). Si un seuil est dépassé → lever une
exception, la calibration est invalide.

---

## 6. Graphiques (matplotlib, PNG, backend Agg)

1. `discount_factors.png` : P_market(0,T) bootstrappés (points) vs
   P_CIR++(0,T) (ligne continue sur [0, 30Y]) — superposition parfaite.
2. `zero_rates.png` : taux zéro-coupon continus, marché (points aux piliers)
   vs modèle (ligne).
3. `phi.png` : phi(t) sur [0, 10Y], avec f_market lissé et f_CIR en appui.
4. `historical_diagnostics.png` (2 panneaux) :
   a. histogramme (densité) du SOFR historique vs pdf de la loi stationnaire
      Gamma du CIR calibré ;
   b. QQ-plot vs N(0,1) des incréments standardisés
      z_i = (x_{i+1} - x_i - kappa (theta - x_i) dt_i) / (sigma sqrt(x_i dt_i)).
   Attendu : queues très épaisses sur le QQ-plot (les sauts FOMC de ±25/75 bp
   ne sont pas gaussiens) — c'est un diagnostic, pas un échec.

---

## 7. Structure du projet et sérialisation

    cirpp/
      __init__.py       # exports publics
      model.py          # CIRParams, A/B/P/f_CIR fermés, DiscountCurve,
                        # PhiFunction, CIRPPModel, loi stationnaire
      calibration.py    # chargement données, AR(1), MLE ncx2, échéanciers,
                        # bootstrap brentq, verify_repricing
      scenario.py       # calibration sur scénario forward utilisateur (§9)
      plots.py          # les 4 graphiques + scenario_fit
      main.py           # pipeline complet (CLI)
    tests/
      test_model.py     # limites en 0 ; A,B vs intégration numérique des ODE
                        # de Riccati (B'=1-kB-0.5s²B², (lnA)'=-kθB, solve_ivp
                        # rtol=1e-12) ; f_CIR vs différence finie de -ln P ;
                        # moments stationnaires ; recovery du MLE sur une
                        # trajectoire simulée par échantillonnage ncx2 exact
                        # (simulation dans le TEST uniquement)
      test_bootstrap.py # roundtrip DF→par rates→bootstrap→DF (rtol 1e-12) ;
                        # seuils de repricing ; cohérence Phi exacte vs
                        # trapèzes du phi non lissé ; roundtrip de
                        # sérialisation ; parsing ON/M/Y ; ancre spot
    requirements.txt    # numpy, scipy, pandas, matplotlib, pytest
    SOFR.csv, swapsofrrates.txt, README.md

Sérialisation : JSON (pas de pickle). Le fichier `output/cirpp_model.json`
contient {params: {kappa, theta, sigma, x0}, curve: {times, dfs}, t_max,
grid_step, smooth_days} — la courbe aux piliers suffit à reconstruire
intégralement phi (ponctuel ET intégrale exacte) au chargement.

API de réutilisation :

    from cirpp import CIRPPModel
    m = CIRPPModel.load("output/cirpp_model.json")
    m.params.kappa, m.params.theta, m.params.sigma, m.params.x0
    m.phi(t)           # phi ponctuel, scalaire ou array
    m.phi.integral(t)  # Phi(t) exacte
    m.zcb_price(T)     # P_CIR++(0,T)

Exécution :

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m cirpp.main --data-dir . --out-dir output
    .venv/bin/python -m pytest tests/          # 13 tests attendus verts

Options CLI : --eps 1e-4, --valuation-date YYYY-MM-DD (défaut : aujourd'hui),
--smooth-days 10, --sofr-file, --swaps-file, --out-dir.

---

## 8. Pièges numériques rencontrés (à ne pas reproduire)

- Calculer A(T) directement (sans log) overflow/perd de la précision à 30Y.
- Intégrer numériquement le phi lissé pour le repricing : erreur ~1e-4,
  seuil 1e-10 impossible. Toujours utiliser l'identité du §4.2.
- Oublier le jacobien ln(2c) dans la log-vraisemblance ncx2 : le MLE part
  dans le décor.
- Utiliser un dt constant 1/252 au lieu des jours calendaires réels : biaise
  kappa et sigma (les week-ends comptent pour 3 jours de diffusion).
- Le floor eps doit être appliqué AVANT le calcul des incréments (sqrt(x)
  au dénominateur de la vraisemblance).
- brentq avec xtol par défaut ne donne pas la précision machine sur les DF :
  serrer xtol=1e-16, rtol=1e-15.

---

## 9. Mode scénario — calibration sur les vues forward de l'utilisateur

Activé par `--scenario fichier.csv` (cf. `scenario_example.csv`). L'utilisateur
fournit, pour chaque ténor (ON, 1Y, ..., 30Y) et chaque horizon h (6M, 1Y,
2Y, ...), sa vision du niveau forward du taux, plus des poids par ténor :

    horizon,ON,1Y,2Y,5Y,10Y,30Y
    0,3.62,3.908,3.961,4.025,4.228,4.483
    6M,3.45,3.75,3.85,4.00,4.25,4.50
    1Y,3.30,3.60,3.75,3.95,4.28,4.52
    weight,1,1,1,1,5,1

La ligne `0` (optionnelle) remplace `swapsofrrates.txt` comme courbe initiale ;
cellules vides = vue non contrainte ; lignes d'horizon dupliquées autorisées
(scénarios divergents superposés).

### 9.1 Formulation (et pourquoi celle-là)

Les vues sont traitées comme des instruments FORWARD-STARTING sur la courbe :

    swap NY vu en h : S(h) = (P(0,h) - P(0,T_N)) / sum_i tau_i P(0,t_i)
                      (échéancier annuel ACT/360 démarrant en h)
    ON vu en h      : (P(0,h)/P(0,h+1j) - 1) * 360   (taux simple 1 jour)

et la calibration ajuste les log-DF de la courbe aux nœuds (horizons +
maturités finales, fusionnés au jour près, interpolation log-linéaire) par
moindres carrés pondérés (scipy least_squares, résidus en bp) :

    residus = sqrt(w_tenor) · (S_courbe - S_vue)·1e4          (vues, h > 0)
            + sqrt(base_weight) · (idem quotes h=0)           (défaut 50)
            + sqrt(lambda / delta_t_mid) · saut_de_forward·1e4 (lissage)

Le CIR++ final = paramètres HISTORIQUES (kappa, theta, sigma du MLE, x0 =
spot) + phi recalé EXACTEMENT sur la courbe scénario (identité du §4.2). Les
vues cohérentes sont repricées à < 0.5 bp ; les vues divergentes donnent le
compromis pondéré (un poids 5 sur le 10Y colle les forwards 10Y au détriment
des ténors en conflit) ; la RMSE pondérée et le tableau vue/fit/erreur sont
affichés, plus `scenario_fit.png`.

NE PAS faire à la place : recalibrer (kappa, theta) pour que les taux le long
du chemin espéré E[x(h)] = theta + (x0-theta)e^{-kappa h} collent aux vues, en
gardant phi sur la courbe de marché. Par Jensen, f(0,t) <= E[r(t)] : le taux
impliqué sur le chemin espéré est structurellement AU-DESSUS du forward absorbé
par phi, donc toute vue sous les forwards de marché est inatteignable —
l'optimisation dégénère (kappa -> 20, theta -> 0, Feller violée, RMSE ~30 bp,
constaté empiriquement). Face à des vues de forwards, l'inconnue est la
courbe (donc phi), pas la dynamique.

### 9.2 Pièges spécifiques

- Pénalité de lissage NON pondérée par la longueur des segments : le solveur
  fitte les vues ON par des creux d'un jour aux nœuds (spikes invisibles dans
  la RMSE, catastrophiques pour les forwards). D'où le facteur
  1/sqrt(delta_t_mid) (Sobolev discret) qui rend un spike d'1 jour ~sqrt(365)
  fois plus coûteux qu'une inflexion annuelle.
- En mode scénario, l'écart aux quotes h=0 peut dépasser 0.1 bp si les vues
  les contredisent : c'est le compromis pondéré (piloté par --base-weight),
  pas un échec — le seuil dur ne s'applique qu'au repricing de la courbe
  scénario elle-même (toujours précision machine).
- Bornes sur les log-DF dans least_squares : [-20, 0.5].

CLI : --scenario, --base-weight 50, --smooth-lambda 0.05.
API : `load_scenario`, `fit_scenario_curve`, `forward_par_rate`,
`scenario_schedule` (exportés par `cirpp`).
