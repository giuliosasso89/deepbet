# app.py
# Avvio: streamlit run app.py
# Requisiti: streamlit pandas numpy scipy plotly openpyxl

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson
from scipy.optimize import minimize
import plotly.express as px

st.set_page_config(page_title="Serie A Predictor — Esatti & 1X2", page_icon="⚽", layout="wide")

# ============ UI ============
st.title("⚽ Serie A Predictor — Esatti & 1X2")
st.caption("Poisson • Elo→xG • DC-lite (veloce). Richiede ≥7 H2H nella stessa direzione Home→Away.")

# ============ DATA ============
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_excel(file) if file is not None else pd.read_excel("Serie_A_2014_2025.xlsx")
    ren = {}
    if "HG" in df.columns and "FTHG" not in df.columns: ren["HG"] = "FTHG"
    if "AG" in df.columns and "FTAG" not in df.columns: ren["AG"] = "FTAG"
    if "Res" in df.columns and "FTR"  not in df.columns: ren["Res"] = "FTR"
    df = df.rename(columns=ren)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    needed = ["HomeTeam","AwayTeam","FTHG","FTAG"]
    for c in needed:
        if c not in df.columns: st.error(f"Colonna mancante: {c}"); st.stop()
    df = df.dropna(subset=["FTHG","FTAG"]).copy()
    df["FTHG"] = df["FTHG"].astype(int); df["FTAG"]=df["FTAG"].astype(int)
    if "FTR" not in df.columns:
        df["FTR"] = np.where(df["FTHG"]>df["FTAG"],"H",np.where(df["FTHG"]<df["FTAG"],"A","D"))
    return df

with st.sidebar:
    st.header("📁 Dati")
    file_in = st.file_uploader("Carica Excel Serie A (2014–2025)", type=["xlsx"])
    df = load_data(file_in)
    st.success(f"Partite caricate: {len(df):,}")

teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
c1, c2, c3 = st.columns([1,1,2])
with c1: home_sel = st.selectbox("🏟️ Home", teams)
with c2: away_sel = st.selectbox("🧳 Away", teams, index=min(1, len(teams)-1))
with c3: goal_max = st.slider("Range gol matrice", 3, 7, 5)

# ============ H2H gate ============
h2h = df[(df["HomeTeam"]==home_sel) & (df["AwayTeam"]==away_sel)]
st.write(f"**H2H trovati {home_sel} (H) vs {away_sel} (A): {len(h2h)}**")
if len(h2h) < 7:
    st.warning("⚠️ Servono almeno 7 precedenti H2H nella stessa direzione Home→Away.")
    st.stop()

# ============ League stats ============
@st.cache_data(show_spinner=False)
def league_means(df):
    return df["FTHG"].mean(), df["FTAG"].mean()
mu_home, mu_away = league_means(df)

# ============ Poisson semplice (attack/defense) ============
@st.cache_data(show_spinner=False)
def fit_simple(df, reg=0.1):
    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
    idx = {t:i for i,t in enumerate(teams)}; n=len(teams)
    gfH=gaH=nH=np.zeros(n); gfA=gaA=nA=np.zeros(n)
    for _,r in df.iterrows():
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

# ============ Elo → xG → Poisson ============
@st.cache_data(show_spinner=False)
def compute_elo(train_df, K=20, home_adv_elo=60):
    teams = sorted(set(train_df["HomeTeam"]).union(train_df["AwayTeam"]))
    elo = {t:1500.0 for t in teams}
    df_ord = train_df.sort_values("Date") if "Date" in train_df.columns else train_df
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

# ============ DC-lite (ρ veloce) ============
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
def fit_rho_dc_lite_from_simple(train_df, simple_params, goal_max=6, grid=np.linspace(-0.2, 0.2, 41)):
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


# stima rho* una volta su tutto lo storico
rho_star = fit_rho_dc_lite_from_simple(df, simple, goal_max=goal_max)


# ============ Utilità comuni ============
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
    dfp=pd.DataFrame(M, index=[str(i) for i in range(M.shape[0])],
                        columns=[str(i) for i in range(M.shape[1])])
    fig=px.imshow(dfp, text_auto=".1%", aspect="auto",
                  labels=dict(x="Gol Away", y="Gol Home", color="Prob"),
                  title=title)
    fig.update_traces(hovertemplate="Home %{y} - Away %{x}: %{z:.3%}<extra></extra>")
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def top_exact(M, k, home, away):
    lst=[((i,j), float(M[i,j])) for i in range(M.shape[0]) for j in range(M.shape[1])]
    lst.sort(key=lambda x:x[1], reverse=True)
    rows=[{"Score": f"{home} {i}-{j} {away}", "Prob": f"{p*100:.2f}%"} for (i,j),p in lst[:k]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============ Calcolo modelli veloci ============
# Poisson
lh_s, la_s = pois_means_simple(home_sel, away_sel)
M_s = exact_matrix_indep(lh_s, la_s, goal_max) if (lh_s and la_s) else None
one_s = oneXtwo(M_s) if M_s is not None else None

# Elo→xG
lh_e, la_e = elo_to_means(home_sel, away_sel)
M_e = exact_matrix_indep(lh_e, la_e, goal_max) if (lh_e and la_e) else None
one_e = oneXtwo(M_e) if M_e is not None else None

# DC-lite (usa λ del Poisson semplice + rho* globale)
M_dcl = exact_matrix_dc_lite(lh_s, la_s, rho_star, goal_max) if (lh_s and la_s) else None
one_dcl = oneXtwo(M_dcl) if M_dcl is not None else None

# Ensemble: pesi da JSON se presente
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

# ============ UI metriche ============
st.subheader("📊 Probabilità 1X2")
if one_s:  metric_cards(one_s, "Poisson")
if one_e:  metric_cards(one_e, "Elo→xG")
if one_dcl: metric_cards(one_dcl, "DC-lite")
st.markdown("---")
if one_ens:
    tag = "pesati" if weights else "media"
    metric_cards(one_ens, f"Ensemble ({tag})")
else:
    st.info("Ensemble disponibile quando almeno un modello è calcolato.")

# ============ Heatmap ============
st.subheader("🎯 Risultati esatti — Heatmap (0..N)")
tabs = st.tabs(["Poisson","Elo→xG","DC-lite","Ensemble"])
with tabs[0]:
    st.markdown(f"λH={lh_s:.2f}, λA={la_s:.2f}" if M_s is not None else "")
    heatmap(M_s, "Poisson") if M_s is not None else st.info("—")
with tabs[1]:
    st.markdown(f"λH={lh_e:.2f}, λA={la_e:.2f}" if M_e is not None else "")
    heatmap(M_e, "Elo→xG") if M_e is not None else st.info("—")
with tabs[2]:
    st.markdown(f"λH={lh_s:.2f}, λA={la_s:.2f} | ρ*={rho_star:.3f}" if M_dcl is not None else "")
    heatmap(M_dcl, "DC-lite") if M_dcl is not None else st.info("—")
with tabs[3]:
    heatmap(M_ens, "Ensemble") if M_ens is not None else st.info("—")

# ============ Top risultati esatti ============
st.subheader("🏆 Top risultati esatti")
k = st.slider("Quanti mostrare", 3, 10, 5)
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown("**Poisson**"); top_exact(M_s, k, home_sel, away_sel) if M_s is not None else st.info("—")
with c2:
    st.markdown("**Elo→xG**"); top_exact(M_e, k, home_sel, away_sel) if M_e is not None else st.info("—")
with c3:
    st.markdown("**DC-lite**"); top_exact(M_dcl, k, home_sel, away_sel) if M_dcl is not None else st.info("—")
with c4:
    st.markdown("**Ensemble**"); top_exact(M_ens, k, home_sel, away_sel) if M_ens is not None else st.info("—")

# ============ Avanzato: DC classico on-demand ============
with st.expander("⚙️ Avanzato — Dixon-Coles classico (lento, on-demand)"):
    if st.button("Calcola DC classico ORA"):
        def dc_corr(i,j,rho):
            if (i,j)==(0,0): return 1 - rho
            if (i,j)==(0,1): return 1 + rho
            if (i,j)==(1,0): return 1 + rho
            if (i,j)==(1,1): return 1 - rho
            return 1.0
        def fit_dc(train_df, reg=0.01, maxiter=150):
            base = simple
            teams=base["teams"]; idx=base["idx"]; n=len(teams)
            att0=np.log(base["att"]); def0=np.log(base["def"])
            ha0=np.log(max(base["muH"]/base["muA"],1.01)); rho0=0.0
            x0=np.r_[att0,def0,[ha0],[rho0]]
            def pack(x): return x[:n], x[n:2*n], x[2*n], x[2*n+1]
            def nll(x):
                att, deff, ha, rho = pack(x); att = att - att.mean()
                ll=0.0
                for _,r in train_df.iterrows():
                    hi=idx.get(r["HomeTeam"]); ai=idx.get(r["AwayTeam"])
                    if hi is None or ai is None: continue
                    lh=np.exp(ha + att[hi] - deff[ai])
                    la=np.exp(att[ai] - deff[hi])
                    gh,ga=int(r["FTHG"]),int(r["FTAG"])
                    p=poisson.pmf(gh,lh)*poisson.pmf(ga,la)*dc_corr(gh,ga,rho)
                    ll += np.log(max(p,1e-12))
                ll -= reg*(np.sum(att**2)+np.sum(deff**2)+ha**2+rho**2)
                return -ll
            res=minimize(nll, x0, method="L-BFGS-B", options={"maxiter":maxiter})
            att,deff,ha,rho=pack(res.x); att=att-att.mean()
            return {"teams":teams,"idx":idx,"att":np.exp(att),"def":np.exp(deff),"ha":float(np.exp(ha)),"rho":float(rho)}
        with st.spinner("Stimo modello Dixon-Coles classico…"):
            dc = fit_dc(df, reg=0.01, maxiter=150)
        hi=simple["idx"][home_sel]; ai=simple["idx"][away_sel]
        lh_dc = dc["ha"] * (simple["att"][hi]/simple["def"][ai])
        la_dc = (simple["att"][ai]/simple["def"][hi])
        # matrice con correzione
        M_dc = np.zeros((goal_max+1, goal_max+1))
        for i in range(goal_max+1):
            for j in range(goal_max+1):
                M_dc[i,j]=poisson.pmf(i,lh_dc)*poisson.pmf(j,la_dc)*dc_corr(i,j,dc["rho"])
        M_dc /= M_dc.sum()
        st.success(f"OK — ρ={dc['rho']:.3f} | λH={lh_dc:.2f}, λA={la_dc:.2f}")
        metric_cards(oneXtwo(M_dc), "Dixon-Coles classico")
        heatmap(M_dc, "Dixon-Coles classico")
