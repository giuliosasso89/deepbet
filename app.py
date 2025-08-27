# app.py
# Avvio: streamlit run app.py
# Requisiti: pip install streamlit pandas numpy scipy plotly openpyxl

import os
import json
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson
import plotly.express as px

st.set_page_config(page_title="Football Predictor — Esatti & 1X2", page_icon="⚽", layout="wide")

# ======= Percorsi file =======
DATA_PATH = "historical_dataset.xlsx"           # dataset multi-league
CAL_SERIEA_PATH = "Calendario Serie A.xlsx"     # calendario con colonna "Giornata"

# ======= Loader dataset principale =======
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"File non trovato: {path}")
        st.stop()
    df = pd.read_excel(path)

    # normalizza alias tipici Football-Data
    ren = {}
    if "HG" in df.columns and "FTHG" not in df.columns: ren["HG"] = "FTHG"
    if "AG" in df.columns and "FTAG" not in df.columns: ren["AG"] = "FTAG"
    if "Res" in df.columns and "FTR"  not in df.columns: ren["Res"] = "FTR"
    df = df.rename(columns=ren)

    needed = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    for c in needed:
        if c not in df.columns:
            st.error(f"Colonna mancante nel dataset: {c}")
            st.stop()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    if "FTR" not in df.columns:
        df["FTR"] = np.where(df["FTHG"] > df["FTAG"], "H",
                      np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    # ripulisci whitespace nomi squadre
    for col in ["HomeTeam", "AwayTeam"]:
        df[col] = df[col].astype(str).str.strip()
    return df

# ======= Loader calendario Serie A =======
@st.cache_data(show_spinner=False)
def load_calendar_serieA(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    cal = pd.read_excel(path)
    for col in ["HomeTeam", "AwayTeam"]:
        if col in cal.columns:
            cal[col] = cal[col].astype(str).str.strip()
    # rileva colonne alternative se necessario
    cols = {c.lower(): c for c in cal.columns}
    home_col = None
    away_col = None
    if "hometeam" in cols: home_col = cols["hometeam"]
    if "awayteam" in cols: away_col = cols["awayteam"]
    if home_col is None and "home" in cols: home_col = cols["home"]
    if away_col is None and "away" in cols: away_col = cols["away"]
    if home_col is None and "squadra casa" in cols: home_col = cols["squadra casa"]
    if away_col is None and "squadra ospite" in cols: away_col = cols["squadra ospite"]

    if "Giornata" not in cal.columns:
        return None
    if home_col and home_col != "HomeTeam":
        cal = cal.rename(columns={home_col: "HomeTeam"})
    if away_col and away_col != "AwayTeam":
        cal = cal.rename(columns={away_col: "AwayTeam"})
    if "HomeTeam" not in cal.columns or "AwayTeam" not in cal.columns:
        return None

    cal["HomeTeam"] = cal["HomeTeam"].astype(str).str.strip()
    cal["AwayTeam"] = cal["AwayTeam"].astype(str).str.strip()
    return cal

df_all = load_data(DATA_PATH)
cal_sa = load_calendar_serieA(CAL_SERIEA_PATH)

# ======= Mapping campionati =======
DIV_MAP = {"D1":"Bundesliga", "E0":"Premier League", "F1":"Ligue 1", "I1":"Serie A", "SP1":"La Liga"}
DIV_INV = {v:k for k, v in DIV_MAP.items()}

# ======= Helpers modello =======
@st.cache_data(show_spinner=False)
def league_means(df_league: pd.DataFrame):
    return df_league["FTHG"].mean(), df_league["FTAG"].mean()

@st.cache_data(show_spinner=False)
def fit_simple(df_league: pd.DataFrame, reg: float = 0.1):
    teams = sorted(set(df_league["HomeTeam"]).union(df_league["AwayTeam"]))
    idx = {t:i for i,t in enumerate(teams)}
    n = len(teams)
    gfH = gaH = nH = np.zeros(n)
    gfA = gaA = nA = np.zeros(n)
    for _, r in df_league.iterrows():
        hi = idx[r["HomeTeam"]]; ai = idx[r["AwayTeam"]]
        gfH[hi] += r["FTHG"]; gaH[hi] += r["FTAG"]; nH[hi] += 1
        gfA[ai] += r["FTAG"]; gaA[ai] += r["FTHG"]; nA[ai] += 1
    muH = gfH.sum()/max(nH.sum(),1)
    muA = gfA.sum()/max(nA.sum(),1)

    att = np.zeros(n); deff = np.zeros(n)
    for i in range(n):
        aH = (gfH[i]+reg*muH)/(nH[i]+reg); aA = (gfA[i]+reg*muA)/(nA[i]+reg)
        dH = (gaH[i]+reg*muA)/(nH[i]+reg); dA = (gaA[i]+reg*muH)/(nA[i]+reg)
        att[i]  = 0.5*(aH/muH + aA/muA)
        deff[i] = 0.5*(dH/muA + dA/muH)
    att  *= n/att.sum()
    deff *= n/deff.sum()
    return {"teams":teams,"idx":idx,"att":att,"def":deff,"muH":muH,"muA":muA}

@st.cache_data(show_spinner=False)
def compute_elo(df_league: pd.DataFrame, K=20, home_adv_elo=60):
    teams = sorted(set(df_league["HomeTeam"]).union(df_league["AwayTeam"]))
    elo = {t:1500.0 for t in teams}
    df_ord = df_league.sort_values("Date") if "Date" in df_league.columns else df_league
    for _, r in df_ord.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        if h not in elo or a not in elo: continue
        Rh = elo[h] + home_adv_elo
        Ra = elo[a]
        Eh = 1/(1 + 10**((Ra-Rh)/400))
        Ea = 1 - Eh
        if r["FTHG"] > r["FTAG"]:
            Sh, Sa = 1, 0
        elif r["FTHG"] < r["FTAG"]:
            Sh, Sa = 0, 1
        else:
            Sh, Sa = 0.5, 0.5
        elo[h] += K*(Sh-Eh)
        elo[a] += K*(Sa-Ea)
    return elo

def pois_means_simple(h, a, simple_params):
    hi = simple_params["idx"].get(h); ai = simple_params["idx"].get(a)
    if hi is None or ai is None: return None, None
    lh = max(0.05, simple_params["muH"]*simple_params["att"][hi]*simple_params["def"][ai])
    la = max(0.05, simple_params["muA"]*simple_params["att"][ai]*simple_params["def"][hi])
    return lh, la

def elo_to_means(home, away, elo_now, mu_total):
    if home not in elo_now or away not in elo_now: return None, None
    Rh = elo_now[home] + 60
    Ra = elo_now[away]
    pH = 1/(1 + 10**((Ra-Rh)/400))
    pA = 1 - pH
    pD = 0.25
    wH = pH + 0.5*pD
    wA = pA + 0.5*pD
    s = wH + wA
    return max(0.05, mu_total*(wH/s)), max(0.05, mu_total*(wA/s))

def dc_corr_cell(i, j, rho):
    if (i, j) == (0, 0): return 1 - rho
    if (i, j) == (1, 1): return 1 - rho
    if (i, j) == (1, 0): return 1 + rho
    if (i, j) == (0, 1): return 1 + rho
    return 1.0

def exact_matrix_indep(lh, la, N):
    i = np.arange(0, N+1)
    M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    return M / M.sum()

def exact_matrix_dc_lite(lh, la, rho, goal_max=6):
    M = np.zeros((goal_max+1, goal_max+1))
    for i in range(goal_max+1):
        for j in range(goal_max+1):
            p = poisson.pmf(i, lh) * poisson.pmf(j, la) * dc_corr_cell(i, j, rho)
            M[i, j] = p
    return M / M.sum()

@st.cache_data(show_spinner=False)
def fit_rho_dc_lite_from_simple(train_df: pd.DataFrame, simple_params: dict, grid=np.linspace(-0.2, 0.2, 41)):
    def means_fn(h, a):
        hi = simple_params["idx"].get(h); ai = simple_params["idx"].get(a)
        if hi is None or ai is None: return None, None
        lh = max(0.05, simple_params["muH"]*simple_params["att"][hi]*simple_params["def"][ai])
        la = max(0.05, simple_params["muA"]*simple_params["att"][ai]*simple_params["def"][hi])
        return lh, la
    best_rho, best_ll = 0.0, -np.inf
    for rho in grid:
        ll = 0.0
        for _, r in train_df.iterrows():
            lh, la = means_fn(r["HomeTeam"], r["AwayTeam"])
            if lh is None or la is None:
                continue
            gh, ga = int(r["FTHG"]), int(r["FTAG"])
            p = poisson.pmf(gh, lh) * poisson.pmf(ga, la) * dc_corr_cell(gh, ga, rho)
            ll += np.log(max(p, 1e-12))
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return float(best_rho)

def oneXtwo(M):
    if M is None:
        return None
    pH = np.tril(M, -1).sum()
    pX = np.trace(M)
    pA = np.triu(M, 1).sum()
    return float(pH), float(pX), float(pA)

def heatmap(M, title):
    if M is None:
        st.info("—")
        return
    dfp = pd.DataFrame(M, index=[str(i) for i in range(M.shape[0])],
                          columns=[str(i) for i in range(M.shape[1])])
    fig = px.imshow(dfp, text_auto=".1%", aspect="auto",
                    labels=dict(x="Gol Away", y="Gol Home", color="Prob"),
                    title=title)
    fig.update_traces(hovertemplate="Home %{y} - Away %{x}: %{z:.3%}<extra></extra>")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def top_exact(M, k, home, away):
    if M is None:
        st.info("—")
        return
    lst = [((i, j), float(M[i, j])) for i in range(M.shape[0]) for j in range(M.shape[1])]
    lst.sort(key=lambda x: x[1], reverse=True)
    rows = [{"Score": f"{home} {i}-{j} {away}", "Prob": f"{p*100:.2f}%"} for (i, j), p in lst[:k]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def metric_cards(one_tuple, label):
    if not one_tuple:
        return
    pH, pX, pA = one_tuple
    c1, c2, c3 = st.columns(3)
    with c1: st.metric(f"{label} — 1 (Home)", f"{pH*100:.1f}%")
    with c2: st.metric(f"{label} — X", f"{pX*100:.1f}%")
    with c3: st.metric(f"{label} — 2 (Away)", f"{pA*100:.1f}%")

# ======= Sidebar filtri =======
with st.sidebar:
    st.header("⚙️ Filtri")
    leagues_present = [d for d in DIV_MAP.keys() if d in df_all["Div"].unique()]
    league_names = [DIV_MAP[d] for d in leagues_present]
    default_idx = league_names.index(DIV_MAP.get("I1", "Serie A")) if "I1" in leagues_present else 0
    league_name_sel = st.selectbox("Campionato", league_names, index=default_idx)
    div_sel = DIV_INV[league_name_sel]
    df = df_all[df_all["Div"] == div_sel].copy()
    st.caption(f"Partite caricate: **{len(df):,}** in {league_name_sel}")

    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
    home_sel = st.selectbox("🏟️ Home", teams, index=0 if len(teams) else 0)
    away_sel = st.selectbox("🧳 Away", teams, index=1 if len(teams) > 1 else 0)

# ======= Tabs principali =======
tab_match, tab_round = st.tabs(["🎯 Scontro Diretto", "🗓️ Giornata (Serie A)"])

# --------------------- TAB MATCH ---------------------
with tab_match:
    st.title(f"{league_name_sel} — {home_sel} vs {away_sel}")

    # H2H gate nella stessa direzione e campionato
    h2h = df[(df["HomeTeam"] == home_sel) & (df["AwayTeam"] == away_sel)]
    st.write(f"**H2H trovati {home_sel} (H) vs {away_sel} (A): {len(h2h)}**")
    if len(h2h) < 7:
        st.warning("⚠️ Servono almeno 7 precedenti H2H (stessa direzione).")
        st.stop()

    # Parametri del campionato selezionato
    mu_home, mu_away = league_means(df)
    simple = fit_simple(df)
    elo_now = compute_elo(df)
    rho_star = fit_rho_dc_lite_from_simple(df, simple)

    # slider matrice
    goal_max = st.slider("Range gol per matrice", 3, 7, 5, key="match_goalmax")

    # Calcolo modelli
    lh_s, la_s = pois_means_simple(home_sel, away_sel, simple)
    M_s = exact_matrix_indep(lh_s, la_s, goal_max) if (lh_s and la_s) else None
    one_s = oneXtwo(M_s)

    lh_e, la_e = elo_to_means(home_sel, away_sel, elo_now, mu_home + mu_away)
    M_e = exact_matrix_indep(lh_e, la_e, goal_max) if (lh_e and la_e) else None
    one_e = oneXtwo(M_e)

    M_dcl = exact_matrix_dc_lite(lh_s, la_s, rho_star, goal_max) if (lh_s and la_s) else None
    one_dcl = oneXtwo(M_dcl)

    # Ensemble con pesi da JSON se disponibile
    weights = None
    if os.path.exists("ensemble_weights.json"):
        try:
            with open("ensemble_weights.json", "r", encoding="utf-8") as f:
                weights = json.load(f).get("weights", None)
        except Exception:
            weights = None

    mats, ws = [], []
    if M_s is not None:  mats.append(M_s);  ws.append(weights.get("Poisson", 1.0) if weights else 1.0)
    if M_e is not None:  mats.append(M_e);  ws.append(weights.get("Elo",     1.0) if weights else 1.0)
    if M_dcl is not None: mats.append(M_dcl); ws.append(weights.get("DC",      1.0) if weights else 1.0)

    M_ens, one_ens = None, None
    if mats:
        total_w = sum(ws)
        W = [w/total_w for w in ws] if total_w > 0 else [1/len(mats)]*len(mats)
        M_ens = sum(wi*Mi for wi, Mi in zip(W, mats))
        one_ens = oneXtwo(M_ens)

    # KPI 1X2
    st.subheader("📊 Probabilità 1X2")
    metric_cards(one_s,   "Poisson")
    metric_cards(one_e,   "Elo→xG")
    metric_cards(one_dcl, "DC-lite")
    st.markdown("---")
    metric_cards(one_ens, f"Ensemble ({'pesati' if weights else 'media'})")

    # Heatmaps
    st.subheader("🎯 Risultati esatti — Heatmap (0..N)")
    ht = st.tabs(["Poisson", "Elo→xG", "DC-lite", "Ensemble"])
    with ht[0]:
        st.markdown(f"λH={lh_s:.2f}, λA={la_s:.2f}" if M_s is not None else "")
        heatmap(M_s, "Poisson")
    with ht[1]:
        st.markdown(f"λH={lh_e:.2f}, λA={la_e:.2f}" if M_e is not None else "")
        heatmap(M_e, "Elo→xG")
    with ht[2]:
        st.markdown(f"λH={lh_s:.2f}, λA={la_s:.2f} | ρ*={rho_star:.3f}" if M_dcl is not None else "")
        heatmap(M_dcl, "DC-lite")
    with ht[3]:
        heatmap(M_ens, "Ensemble")

    # Top esatti
    st.subheader("🏆 Top risultati esatti")
    k = st.slider("Quanti mostrare", 3, 10, 5, key="match_topk")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown("**Poisson**");  top_exact(M_s,   k, home_sel, away_sel)
    with c2: st.markdown("**Elo→xG**");  top_exact(M_e,   k, home_sel, away_sel)
    with c3: st.markdown("**DC-lite**"); top_exact(M_dcl, k, home_sel, away_sel)
    with c4: st.markdown("**Ensemble**"); top_exact(M_ens, k, home_sel, away_sel)

    # --------- 💡 Value vs Mercato (AvgH/AvgD/AvgA) ----------
    st.subheader("💡 Value vs mercato (se disponibili)")
    def implied(p): return (1.0/p) if (p and p > 0) else None

    odds_row = None
    if all(c in df.columns for c in ["AvgH", "AvgD", "AvgA"]):
        tmp = df[(df["HomeTeam"] == home_sel) & (df["AwayTeam"] == away_sel)].dropna(subset=["AvgH", "AvgD", "AvgA"])
        if len(tmp) > 0:
            odds_row = tmp.sort_values("Date").iloc[-1]

    if odds_row is not None and one_ens:
        avgH, avgD, avgA = float(odds_row["AvgH"]), float(odds_row["AvgD"]), float(odds_row["AvgA"])
        imp = np.array([implied(avgH), implied(avgD), implied(avgA)], dtype=float)
        if np.isfinite(imp).all() and imp.sum() > 0:
            imp = imp / imp.sum()  # rimuove overround
            pH, pX, pA = one_ens
            model = np.array([pH, pX, pA])
            edge = (model - imp) * 100

            model_pct  = [f"{x:.2f}%" for x in (model*100)]
            market_pct = [f"{x:.2f}%" for x in (imp*100)]

            df_val = pd.DataFrame({
                "Esito": ["1 (Home)","X (Draw)","2 (Away)"],
		"Prob modello": model_pct,
    		"Prob mercato": market_pct,
		"Edge (p.p.)": np.round(edge, 2)
            })
            st.dataframe(df_val, use_container_width=True, hide_index=True)
        else:
            st.info("Quote non valide per calcolare le probabilità implicite.")
    else:
        st.info("Quote AvgH/AvgD/AvgA non presenti per questo scontro o ensemble non disponibile.")

    # Storico H2H
    st.subheader("📚 Storico H2H (solo Home→Away selezionate)")
    show_cols = [c for c in ["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG","FTR","AvgH","AvgD","AvgA"] if c in h2h.columns]
    st.dataframe(h2h.sort_values("Date", ascending=False)[show_cols], use_container_width=True)

# --------------------- TAB GIORNATA (SERIE A) ---------------------
with tab_round:
    st.title("Serie A — Calcolo per Giornata")
    if cal_sa is None:
        st.warning("Non trovo 'Calendario Serie A.xlsx' oppure mancano le colonne 'Giornata'/'HomeTeam'/'AwayTeam'.")
    else:
        giornate = sorted([g for g in cal_sa["Giornata"].dropna().unique()])
        giornata_sel = st.selectbox("Seleziona la Giornata", giornate, index=0)

        # Dataset & parametri SOLO SERIE A
        df_sa = df_all[df_all["Div"] == "I1"].copy()
        muH_sa, muA_sa = league_means(df_sa)
        simple_sa = fit_simple(df_sa)
        elo_sa = compute_elo(df_sa)
        rho_sa = fit_rho_dc_lite_from_simple(df_sa, simple_sa)

        # slider matrice
        goal_max_round = st.slider("Range gol per matrice (Giornata)", 3, 7, 5, key="round_goalmax")

        # pesi ensemble (opzionali)
        weights = None
        if os.path.exists("ensemble_weights.json"):
            try:
                with open("ensemble_weights.json", "r", encoding="utf-8") as f:
                    weights = json.load(f).get("weights", None)
            except Exception:
                weights = None

        # partite della giornata selezionata
        fixtures = cal_sa[cal_sa["Giornata"] == giornata_sel][["HomeTeam", "AwayTeam"]].copy()
        fixtures["HomeTeam"] = fixtures["HomeTeam"].astype(str).str.strip()
        fixtures["AwayTeam"] = fixtures["AwayTeam"].astype(str).str.strip()

        # batch
        rows = []
        for _, row in fixtures.iterrows():
            h, a = row["HomeTeam"], row["AwayTeam"]
            # H2H gate nella Serie A
            h2h_sa = df_sa[(df_sa["HomeTeam"] == h) & (df_sa["AwayTeam"] == a)]
            if len(h2h_sa) < 7:
                rows.append({
                    "Match": f"{h} - {a}",
                    "Note": "H2H < 7 (skip)",
                    "1": None, "X": None, "2": None,
                    "Top1": None, "Top2": None,
                    "λH": None, "λA": None
                })
                continue

            # λ dal Poisson semplice
            lh_s, la_s = pois_means_simple(h, a, simple_sa)
            if lh_s is None or la_s is None:
                rows.append({
                    "Match": f"{h} - {a}",
                    "Note": "Team non trovato nello storico Serie A",
                    "1": None, "X": None, "2": None,
                    "Top1": None, "Top2": None,
                    "λH": None, "λA": None
                })
                continue

            # Matrici dei modelli
            M_s = exact_matrix_indep(lh_s, la_s, goal_max_round)
            lh_e, la_e = elo_to_means(h, a, elo_sa, muH_sa + muA_sa)
            M_e = exact_matrix_indep(lh_e, la_e, goal_max_round) if (lh_e and la_e) else None
            M_dcl = exact_matrix_dc_lite(lh_s, la_s, rho_sa, goal_max_round)

            mats = [m for m in [M_s, M_e, M_dcl] if m is not None]
            ws = []
            if weights:
                if M_s is not None: ws.append(weights.get("Poisson", 1.0))
                if M_e is not None: ws.append(weights.get("Elo", 1.0))
                if M_dcl is not None: ws.append(weights.get("DC", 1.0))
            else:
                ws = [1.0]*len(mats)

            if mats:
                total_w = sum(ws)
                W = [w/total_w for w in ws] if total_w > 0 else [1/len(mats)]*len(mats)
                M_ens = sum(wi*Mi for wi, Mi in zip(W, mats))
                pH, pX, pA = oneXtwo(M_ens)
            else:
                M_ens = None
                pH = pX = pA = None

            # Top due risultati dall’ensemble
            def top2(M):
                lst = []
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        lst.append(((i, j), float(M[i, j])))
                lst.sort(key=lambda x: x[1], reverse=True)
                if not lst: return None, None
                t1 = f"{h} {lst[0][0][0]}-{lst[0][0][1]} {a} ({lst[0][1]*100:.1f}%)"
                t2 = f"{h} {lst[1][0][0]}-{lst[1][0][1]} {a} ({lst[1][1]*100:.1f}%)" if len(lst) > 1 else None
                return t1, t2

            t1, t2 = top2(M_ens) if M_ens is not None else (None, None)

            rows.append({
                "Match": f"{h} - {a}",
                "Note": "",
                "1": round(pH*100, 1) if pH is not None else None,
                "X": round(pX*100, 1) if pX is not None else None,
                "2": round(pA*100, 1) if pA is not None else None,
                "Top1": t1, "Top2": t2,
                "λH": round(lh_s, 2), "λA": round(la_s, 2)
            })

        df_round = pd.DataFrame(rows)
        st.dataframe(df_round, use_container_width=True, hide_index=True)

        # download CSV
        if len(df_round):
            csv = df_round.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Scarica CSV Giornata",
                               data=csv,
                               file_name=f"pronostici_giornata_{giornata_sel}.csv",
                               mime="text/csv")

# ======= Footer =======
st.info(
    "ℹ️ Modelli: **Poisson** (forze att/def), **Elo→xG** (rating dinamico), **DC-lite** (correzione rapida punteggi bassi). "
    "Il box **Value vs mercato** usa `AvgH/AvgD/AvgA` se disponibili."
)
