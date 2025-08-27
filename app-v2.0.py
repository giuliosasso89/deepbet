# app.py
# Avvio: streamlit run app.py
# Requisiti: pip install streamlit pandas numpy scipy plotly openpyxl

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson
import plotly.express as px

st.set_page_config(page_title="Football Predictor — Esatti & 1X2", page_icon="⚽", layout="wide")

# ===================== DATA LOADING =====================
DATA_PATH = "historical_dataset.xlsx"  # caricato automaticamente

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

    # colonne minime
    needed = ["Div","Date","HomeTeam","AwayTeam","FTHG","FTAG"]
    for c in needed:
        if c not in df.columns:
            st.error(f"Colonna mancante nel dataset: {c}")
            st.stop()

    # parse date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FTHG","FTAG"]).copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    if "FTR" not in df.columns:
        df["FTR"] = np.where(df["FTHG"]>df["FTAG"],"H",np.where(df["FTHG"]<df["FTAG"],"A","D"))
    return df

df_all = load_data(DATA_PATH)

# mappa codici campionato -> nome leggibile
DIV_MAP = {
    "D1": "Bundesliga",
    "E0": "Premier League",
    "F1": "Ligue 1",
    "I1": "Serie A",
    "SP1": "La Liga",
}
DIV_INV = {v:k for k,v in DIV_MAP.items()}

# ===================== SIDEBAR FILTRI =====================
with st.sidebar:
    st.header("⚙️ Filtri")
    leagues_present = [d for d in DIV_MAP.keys() if d in df_all["Div"].unique()]
    league_names = [DIV_MAP[d] for d in leagues_present]
    league_name_sel = st.selectbox("Campionato", league_names, index=league_names.index(DIV_MAP.get("I1","Serie A")) if "I1" in leagues_present else 0)
    div_sel = DIV_INV[league_name_sel]

    # filtra dataset per campionato
    df = df_all[df_all["Div"] == div_sel].copy()
    st.caption(f"Partite caricate: **{len(df):,}** in {league_name_sel}")

    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
    home_sel = st.selectbox("🏟️ Home", teams, index=0 if len(teams)==0 else 0)
    away_sel = st.selectbox("🧳 Away", teams, index=1 if len(teams)>1 else 0)

# ===================== H2H GATE =====================
h2h = df[(df["HomeTeam"]==home_sel) & (df["AwayTeam"]==away_sel)]
st.title(f"⚽ {league_name_sel} — {home_sel} vs {away_sel}")
st.caption("Pronostici su scontro specifico con Poisson, Elo→xG e DC-lite. Richiede ≥7 H2H nella stessa direzione (Home→Away).")

st.write(f"**H2H trovati {home_sel} (H) vs {away_sel} (A) in {league_name_sel}: {len(h2h)}**")
if len(h2h) < 7:
    st.warning("⚠️ Servono almeno 7 precedenti H2H (stessa direzione Home→Away) per effettuare il pronostico.")
    st.stop()

# ===================== LEAGUE STATS =====================
@st.cache_data(show_spinner=False)
def league_means(df_league: pd.DataFrame):
    return df_league["FTHG"].mean(), df_league["FTAG"].mean()
mu_home, mu_away = league_means(df)

# ===================== MODELLI =====================
# Poisson semplice (attack/defense)
@st.cache_data(show_spinner=False)
def fit_simple(df_league: pd.DataFrame, reg: float = 0.1):
    teams = sorted(set(df_league["HomeTeam"]).union(df_league["AwayTeam"]))
    idx = {t:i for i,t in enumerate(teams)}; n=len(teams)
    gfH=gaH=nH=np.zeros(n); gfA=gaA=nA=np.zeros(n)
    for _,r in df_league.iterrows():
        hi=idx[r["HomeTeam"]]; ai=idx[r["AwayTeam"]]
        gfH[hi]+=r["FTHG"]; gaH[hi]+=r["FTAG"]; nH[hi]+=1
        gfA[ai]+=r["FTAG"]; gaA[ai]+=r["FTHG"]; nA[ai]+=1
    muH = gfH.sum()/max(nH.sum(),1); muA = gfA.sum()/max(nA.sum(),1)
    att=np.zeros(n); deff=np.zeros(n)
    for i in range(n):
        aH=(gfH[i]+reg*muH)/(nH[i]+reg); aA=(gfA[i]+reg*muA)/(nA[i]+reg)
        dH=(gaH[i]+reg*muA)/(nH[i]+reg); dA=(gaA[i]+reg*muH)/(nA[i]+reg)
        att[i]=0.5*(aH/muH + aA/muA)
        deff[i]=0.5*(dH/muA + dA/muH)
    att*=n/att.sum(); deff*=n/deff.sum()
    return {"teams":teams,"idx":idx,"att":att,"def":deff,"muH":muH,"muA":muA}

simple = fit_simple(df)

def pois_means_simple(h, a, p=simple):
    hi=p["idx"].get(h); ai=p["idx"].get(a)
    if hi is None or ai is None: return None, None
    lh = max(0.05, p["muH"]*p["att"][hi]*p["def"][ai])
    la = max(0.05, p["muA"]*p["att"][ai]*p["def"][hi])
    return lh, la

# Elo → xG → Poisson
@st.cache_data(show_spinner=False)
def compute_elo(df_league: pd.DataFrame, K=20, home_adv_elo=60):
    teams = sorted(set(df_league["HomeTeam"]).union(df_league["AwayTeam"]))
    elo = {t:1500.0 for t in teams}
    df_ord = df_league.sort_values("Date") if "Date" in df_league.columns else df_league
    for _,r in df_ord.iterrows():
        h,a=r["HomeTeam"], r["AwayTeam"]
        if h not in elo or a not in elo: continue
        Rh=elo[h]+home_adv_elo; Ra=elo[a]
        Eh=1/(1+10**((Ra-Rh)/400)); Ea=1-Eh
        if r["FTHG"]>r["FTAG"]: Sh,Sa=1,0
        elif r["FTHG"]<r["FTAG"]: Sh,Sa=0,1
        else: Sh,Sa=0.5,0.5
        elo[h]+=K*(Sh-Eh); elo[a]+=K*(Sa-Ea)
    return elo

elo_now = compute_elo(df)

def elo_to_means(home, away, mu_total=None):
    if mu_total is None: mu_total = mu_home + mu_away
    if home not in elo_now or away not in elo_now: return None, None
    Rh=elo_now[home]+60; Ra=elo_now[away]
    pH = 1/(1+10**((Ra-Rh)/400)); pA = 1-pH; pD = 0.25
    wH = pH + 0.5*pD; wA = pA + 0.5*pD
    s = wH + wA
    return max(0.05, mu_total*(wH/s)), max(0.05, mu_total*(wA/s))

# DC-lite (ρ stimato velocemente su storico del campionato)
def dc_corr_cell(i, j, rho):
    if (i, j) == (0, 0): return 1 - rho
    if (i, j) == (1, 1): return 1 - rho
    if (i, j) == (1, 0): return 1 + rho
    if (i, j) == (0, 1): return 1 + rho
    return 1.0

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

# stima rho* per il campionato selezionato (cache)
rho_star = fit_rho_dc_lite_from_simple(df, simple)

# ===================== UTILITIES =====================
def exact_matrix_indep(lh, la, N):
    i=np.arange(0,N+1)
    M=np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    return M/M.sum()

def oneXtwo(M):
    ph=np.tril(M,-1).sum(); pd=np.trace(M); pa=np.triu(M,1).sum()
    return float(ph),float(pd),float(pa)

def metric_cards(one_tuple, label):
    ph,pd,pa = one_tuple
    c1,c2,c3=st.columns(3)
    with c1: st.metric(f"{label} — 1 (Home)", f"{ph*100:.1f}%")
    with c2: st.metric(f"{label} — X", f"{pd*100:.1f}%")
    with c3: st.metric(f"{label} — 2 (Away)", f"{pa*100:.1f}%")

def heatmap(M, title):
    if M is None:
        st.info("—"); return
    dfp=pd.DataFrame(M, index=[str(i) for i in range(M.shape[0])],
                        columns=[str(i) for i in range(M.shape[1])])
    fig=px.imshow(dfp, text_auto=".1%", aspect="auto",
                  labels=dict(x="Gol Away", y="Gol Home", color="Prob"),
                  title=title)
    fig.update_traces(hovertemplate="Home %{y} - Away %{x}: %{z:.3%}<extra></extra>")
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def top_exact(M, k, home, away):
    if M is None:
        st.info("—"); return
    lst=[((i,j), float(M[i,j])) for i in range(M.shape[0]) for j in range(M.shape[1])]
    lst.sort(key=lambda x:x[1], reverse=True)
    rows=[{"Score": f"{home} {i}-{j} {away}", "Prob": f"{p*100:.2f}%"} for (i,j),p in lst[:k]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ===================== CALCOLO MODELLI (LEGA) =====================
# slider per matrice
goal_max = st.slider("Range gol per matrice", 3, 7, 5, help="Limite massimo gol per lato (0..N)")

# Poisson
lh_s, la_s = pois_means_simple(home_sel, away_sel)
M_s = exact_matrix_indep(lh_s, la_s, goal_max) if (lh_s and la_s) else None
one_s = oneXtwo(M_s) if M_s is not None else None

# Elo→xG
lh_e, la_e = elo_to_means(home_sel, away_sel)
M_e = exact_matrix_indep(lh_e, la_e, goal_max) if (lh_e and la_e) else None
one_e = oneXtwo(M_e) if M_e is not None else None

# DC-lite (usa λ del Poisson + ρ* del campionato)
M_dcl = exact_matrix_dc_lite(lh_s, la_s, rho_star, goal_max) if (lh_s and la_s) else None
one_dcl = oneXtwo(M_dcl) if M_dcl is not None else None

# Ensemble pesato da JSON (se disponibile), altrimenti media semplice
weights = None
if os.path.exists("ensemble_weights.json"):
    try:
        with open("ensemble_weights.json","r",encoding="utf-8") as f:
            j = json.load(f)
            weights = j.get("weights", None)
    except Exception:
        weights = None

mats = []; ws = []
if M_s is not None: 
    mats.append(M_s); ws.append(weights.get("Poisson", 1.0) if weights else 1.0)
if M_e is not None: 
    mats.append(M_e); ws.append(weights.get("Elo", 1.0) if weights else 1.0)
if M_dcl is not None: 
    mats.append(M_dcl); ws.append(weights.get("DC", 1.0) if weights else 1.0)

M_ens = None; one_ens = None
if mats:
    W = np.array(ws) / np.sum(ws)
    M_ens = sum(wi*Mi for wi,Mi in zip(W, mats))
    one_ens = oneXtwo(M_ens)

# ===================== UI =====================
st.subheader("📊 Probabilità 1X2")
if one_s:   metric_cards(one_s, "Poisson")
if one_e:   metric_cards(one_e, "Elo→xG")
if one_dcl: metric_cards(one_dcl, "DC-lite")
st.markdown("---")
if one_ens:
    metric_cards(one_ens, f"Ensemble ({'pesati' if weights else 'media'})")
else:
    st.info("Ensemble disponibile quando almeno un modello è calcolato.")

st.subheader("🎯 Risultati esatti — Heatmap (0..N)")
tabs = st.tabs(["Poisson","Elo→xG","DC-lite","Ensemble"])
with tabs[0]:
    st.markdown(f"λH={lh_s:.2f}, λA={la_s:.2f}" if M_s is not None else "")
    heatmap(M_s, "Poisson")
with tabs[1]:
    st.markdown(f"λH={lh_e:.2f}, λA={la_e:.2f}" if M_e is not None else "")
    heatmap(M_e, "Elo→xG")
with tabs[2]:
    st.markdown(f"λH={lh_s:.2f}, λA={la_s:.2f} | ρ*={rho_star:.3f}" if M_dcl is not None else "")
    heatmap(M_dcl, "DC-lite")
with tabs[3]:
    heatmap(M_ens, "Ensemble")

st.subheader("🏆 Top risultati esatti")
k = st.slider("Quanti mostrare", 3, 10, 5)
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown("**Poisson**");  top_exact(M_s,   k, home_sel, away_sel)
with c2:
    st.markdown("**Elo→xG**");  top_exact(M_e,   k, home_sel, away_sel)
with c3:
    st.markdown("**DC-lite**"); top_exact(M_dcl, k, home_sel, away_sel)
with c4:
    st.markdown("**Ensemble**"); top_exact(M_ens, k, home_sel, away_sel)

st.markdown("---")
st.subheader("📚 Storico H2H (solo selezione Home→Away nel campionato)")
show_cols = [c for c in ["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG","FTR","AvgH","AvgD","AvgA"] if c in h2h.columns]
if "Date" in h2h.columns:
    st.dataframe(h2h.sort_values("Date", ascending=False)[show_cols], use_container_width=True)
else:
    st.dataframe(h2h[show_cols], use_container_width=True)

st.info(
    "ℹ️ Modelli:\n"
    "- **Poisson**: forze attacco/difesa + vantaggio casa implicito; indipendenza gol.\n"
    "- **Elo→xG**: rating dinamico convertito in gol attesi.\n"
    "- **DC-lite**: correzione rapida sui punteggi bassi (stima ρ su storico del campionato)."
)
