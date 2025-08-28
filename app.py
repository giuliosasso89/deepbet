# app.py
import os, json, unicodedata
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson
import plotly.express as px

st.set_page_config(page_title="Football Predictor — Esatti & 1X2", page_icon="⚽", layout="wide")

# ================= THEME / STYLE =================
st.markdown("""
<style>
:root{ --col-1:#0ea84b; --col-x:#6b7280; --col-2:#ef4444; }
html, body, [class*="css"] { font-variant-numeric: tabular-nums; }
.kpi{ border-radius:16px; padding:12px 16px; background:#0b1220; color:#fff; }
.kpi .label{ font-size:.8rem; opacity:.75; margin-bottom:2px; }
.kpi .value{ font-size:1.6rem; font-weight:700; }
.badge{ display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; margin:4px 6px 0 0; border:1px solid transparent; cursor:pointer; }
.badge.home{ background:rgba(14,168,75,.12); color:var(--col-1); border-color:rgba(14,168,75,.28); }
.badge.draw{ background:rgba(107,114,128,.12); color:var(--col-x); border-color:rgba(107,114,128,.28); }
.badge.away{ background:rgba(239,68,68,.12); color:var(--col-2); border-color:rgba(239,68,68,.28); }
.badge.active{ box-shadow:0 0 0 2px currentColor inset; }
.small-muted{ color:#6b7280; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# ================= PATHS =================
DATA_PATH         = "historical_dataset.xlsx"     # storico multi-league
NEXT_PATH         = "next_match.xlsx"             # prossimi match (Div, Giornata, HomeTeam, AwayTeam, ...)
LEAGUE_META_PATH  = "country_league_data.xlsx"    # league.name, league.logo
TEAM_META_PATH    = "team_data.xlsx"              # team.name, team.logo
ALIAS_PATH        = "team_aliases.xlsx"           # alias, canonical

DIV_MAP = {"D1":"Bundesliga", "E0":"Premier League", "F1":"Ligue 1", "I1":"Serie A", "SP1":"La Liga"}
DIV_INV = {v:k for k, v in DIV_MAP.items()}

# ================= NORMALIZZAZIONE / ALIAS =================
def _strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))
def _norm(s: str) -> str:
    s = str(s).strip()
    s = " ".join(s.split())
    s = _strip_accents(s)
    return s.casefold()

@st.cache_data(show_spinner=False)
def load_team_aliases(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path): return None
    df = pd.read_excel(path)
    if not set(["alias","canonical"]).issubset(df.columns): return None
    df = df.copy()
    df["alias_norm"] = df["alias"].astype(str).map(_norm)
    return df[["alias","canonical","alias_norm"]]

def apply_alias_series(series: pd.Series, alias_df: Optional[pd.DataFrame]) -> pd.Series:
    if alias_df is None or alias_df.empty: return series.astype(str)
    alias_map = dict(zip(alias_df["alias_norm"], alias_df["canonical"]))
    return series.astype(str).map(lambda x: alias_map.get(_norm(x), x))

# ================= LOADERS =================
@st.cache_data(show_spinner=False)
def load_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"File non trovato: {path}"); st.stop()
    df = pd.read_excel(path)

    ren = {}
    if "HG" in df.columns and "FTHG" not in df.columns: ren["HG"] = "FTHG"
    if "AG" in df.columns and "FTAG" not in df.columns: ren["AG"] = "FTAG"
    if "Res" in df.columns and "FTR"  not in df.columns: ren["Res"] = "FTR"
    df = df.rename(columns=ren)

    needed = ["Div","Date","HomeTeam","AwayTeam","FTHG","FTAG"]
    for c in needed:
        if c not in df.columns: st.error(f"Colonna mancante: {c}"); st.stop()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FTHG","FTAG"]).copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    if "FTR" not in df.columns:
        df["FTR"] = np.where(df["FTHG"]>df["FTAG"],"H",np.where(df["FTHG"]<df["FTAG"],"A","D"))
    for c in ["HomeTeam","AwayTeam"]:
        df[c] = df[c].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_next(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path): return None
    nxt = pd.read_excel(path)
    needed = ["Div","Giornata","HomeTeam","AwayTeam"]
    if not set(needed).issubset(nxt.columns): return None
    for c in ["HomeTeam","AwayTeam"]:
        nxt[c] = nxt[c].astype(str).str.strip()
    # opzionali: Date/Time
    return nxt

@st.cache_data(show_spinner=False)
def load_league_branding(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path): return None
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("league.name", cols.get("name", None))
    logo_col = cols.get("league.logo", cols.get("logo", None))
    if not name_col or not logo_col: return None
    out = df[[name_col, logo_col]].rename(columns={name_col:"league.name", logo_col:"league.logo"}).copy()
    out["league.name_norm"] = out["league.name"].astype(str).map(_norm)
    return out

@st.cache_data(show_spinner=False)
def load_team_branding(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path): return None
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("team.name", cols.get("name", None))
    logo_col = cols.get("team.logo", cols.get("logo", None))
    if not name_col or not logo_col: return None
    out = df[[name_col, logo_col]].rename(columns={name_col:"team.name", logo_col:"team.logo"}).copy()
    out["team.name_norm"] = out["team.name"].astype(str).map(_norm)
    return out

@st.cache_data(show_spinner=False)
def get_league_logo(league_display_name: str, league_meta: Optional[pd.DataFrame]) -> Optional[str]:
    if league_meta is None: return None
    hit = league_meta.loc[league_meta["league.name_norm"] == _norm(league_display_name)]
    return str(hit.iloc[0]["league.logo"]) if len(hit) else None

@st.cache_data(show_spinner=False)
def get_team_logo(team_name: str, team_meta: Optional[pd.DataFrame]) -> Optional[str]:
    if team_meta is None: return None
    hit = team_meta.loc[team_meta["team.name_norm"] == _norm(team_name)]
    return str(hit.iloc[0]["team.logo"]) if len(hit) else None

# Load
df_all   = load_history(DATA_PATH)
next_df  = load_next(NEXT_PATH)
lg_meta  = load_league_branding(LEAGUE_META_PATH)
tm_meta  = load_team_branding(TEAM_META_PATH)
alias_df = load_team_aliases(ALIAS_PATH)

# Applica alias (se presenti)
if alias_df is not None:
    df_all["HomeTeam"] = apply_alias_series(df_all["HomeTeam"], alias_df)
    df_all["AwayTeam"] = apply_alias_series(df_all["AwayTeam"], alias_df)
    if next_df is not None:
        next_df["HomeTeam"] = apply_alias_series(next_df["HomeTeam"], alias_df)
        next_df["AwayTeam"] = apply_alias_series(next_df["AwayTeam"], alias_df)

# ================= MODELLI =================
@st.cache_data(show_spinner=False)
def league_means(df_league: pd.DataFrame):
    return df_league["FTHG"].mean(), df_league["FTAG"].mean()

@st.cache_data(show_spinner=False)
def fit_simple(df_league: pd.DataFrame, reg: float = 0.1):
    teams = sorted(set(df_league["HomeTeam"]).union(df_league["AwayTeam"]))
    idx = {t:i for i,t in enumerate(teams)}; n=len(teams)
    gfH=gaH=nH=np.zeros(n); gfA=gaA=nA=np.zeros(n)
    for _,r in df_league.iterrows():
        hi=idx[r["HomeTeam"]]; ai=idx[r["AwayTeam"]]
        gfH[hi]+=r["FTHG"]; gaH[hi]+=r["FTAG"]; nH[hi]+=1
        gfA[ai]+=r["FTAG"]; gaA[ai]+=r["FTHG"]; nA[ai]+=1
    muH=gfH.sum()/max(nH.sum(),1); muA=gfA.sum()/max(nA.sum(),1)
    att=np.zeros(n); deff=np.zeros(n)
    for i in range(n):
        aH=(gfH[i]+reg*muH)/(nH[i]+reg); aA=(gfA[i]+reg*muA)/(nA[i]+reg)
        dH=(gaH[i]+reg*muA)/(nH[i]+reg); dA=(gaA[i]+reg*muH)/(nA[i]+reg)
        att[i]=0.5*(aH/muH + aA/muA); deff[i]=0.5*(dH/muA + dA/muH)
    att*=n/att.sum(); deff*=n/deff.sum()
    return {"teams":teams,"idx":idx,"att":att,"def":deff,"muH":muH,"muA":muA}

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

def pois_means_simple(h, a, simple_params):
    hi=simple_params["idx"].get(h); ai=simple_params["idx"].get(a)
    if hi is None or ai is None: return None, None
    lh = max(0.05, simple_params["muH"]*simple_params["att"][hi]*simple_params["def"][ai])
    la = max(0.05, simple_params["muA"]*simple_params["att"][ai]*simple_params["def"][hi])
    return lh, la

def elo_to_means(home, away, elo_now, mu_total):
    if home not in elo_now or away not in elo_now: return None, None
    Rh=elo_now[home]+60; Ra=elo_now[away]
    pH=1/(1+10**((Ra-Rh)/400)); pA=1-pH; pD=0.25
    wH=pH+0.5*pD; wA=pA+0.5*pD
    s=wH+wA
    return max(0.05, mu_total*(wH/s)), max(0.05, mu_total*(wA/s))

def dc_corr_cell(i, j, rho):
    if (i, j) == (0, 0): return 1 - rho
    if (i, j) == (1, 1): return 1 - rho
    if (i, j) == (1, 0): return 1 + rho
    if (i, j) == (0, 1): return 1 + rho
    return 1.0

def exact_matrix_indep(lh, la, N):
    i=np.arange(0,N+1)
    M=np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    return M/M.sum()

def exact_matrix_dc_lite(lh, la, rho, goal_max=6):
    M=np.zeros((goal_max+1, goal_max+1))
    for i in range(goal_max+1):
        for j in range(goal_max+1):
            p=poisson.pmf(i, lh)*poisson.pmf(j, la)*dc_corr_cell(i, j, rho)
            M[i,j]=p
    return M/M.sum()

@st.cache_data(show_spinner=False)
def fit_rho_dc_lite_from_simple(train_df: pd.DataFrame, simple_params: dict, grid=np.linspace(-0.2, 0.2, 41)):
    def means_fn(h, a):
        hi=simple_params["idx"].get(h); ai=simple_params["idx"].get(a)
        if hi is None or ai is None: return None, None
        lh=max(0.05, simple_params["muH"]*simple_params["att"][hi]*simple_params["def"][ai])
        la=max(0.05, simple_params["muA"]*simple_params["att"][ai]*simple_params["def"][hi])
        return lh, la
    best_rho, best_ll = 0.0, -np.inf
    for rho in grid:
        ll=0.0
        for _,r in train_df.iterrows():
            lh,la=means_fn(r["HomeTeam"], r["AwayTeam"])
            if lh is None or la is None: continue
            gh,ga=int(r["FTHG"]), int(r["FTAG"])
            p=poisson.pmf(gh, lh)*poisson.pmf(ga, la)*dc_corr_cell(gh, ga, rho)
            ll+=np.log(max(p,1e-12))
        if ll>best_ll: best_ll, best_rho = ll, rho
    return float(best_rho)

def oneXtwo(M):
    if M is None: return None
    pH=float(np.tril(M,-1).sum()); pX=float(np.trace(M)); pA=float(np.triu(M,1).sum())
    return pH, pX, pA

def h2h_record(h2h_df: pd.DataFrame) -> str:
    if h2h_df is None or len(h2h_df)==0: return "0-0-0"
    w=int((h2h_df["FTHG"]>h2h_df["FTAG"]).sum())
    d=int((h2h_df["FTHG"]==h2h_df["FTAG"]).sum())
    l=int((h2h_df["FTHG"]<h2h_df["FTAG"]).sum())
    return f"{w}-{d}-{l}"

def form_5g(df_league: pd.DataFrame, team: str) -> Tuple[str, int]:
    if "Date" not in df_league.columns: return ("—", 0)
    df_t=df_league[(df_league["HomeTeam"]==team)|(df_league["AwayTeam"]==team)].sort_values("Date", ascending=False).head(5)
    if len(df_t)==0: return ("—",0)
    seq=[]; pts=0
    for _,r in df_t.iterrows():
        if r["HomeTeam"]==team:
            if r["FTHG"]>r["FTAG"]: seq.append("W"); pts+=3
            elif r["FTHG"]<r["FTAG"]: seq.append("L")
            else: seq.append("D"); pts+=1
        else:
            if r["FTAG"]>r["FTHG"]: seq.append("W"); pts+=3
            elif r["FTAG"]<r["FTHG"]: seq.append("L")
            else: seq.append("D"); pts+=1
    return ("".join(seq), pts)

def get_kickoff(next_df: Optional[pd.DataFrame], div_code: str, home: str, away: str) -> str:
    if next_df is None or "Div" not in next_df.columns: return "—"
    cand=next_df[(next_df["Div"]==div_code)&(next_df["HomeTeam"]==home)&(next_df["AwayTeam"]==away)]
    if len(cand)==0: return "—"
    row=cand.iloc[0]
    dt=row.get("Date", None); tm=row.get("Time", row.get("Kickoff", None))
    try:
        if pd.notna(dt):
            dt=pd.to_datetime(dt, errors="coerce")
            if pd.notna(dt):
                return f"{dt:%d/%m/%Y} {str(tm)[:5]}" if pd.notna(tm) else f"{dt:%d/%m/%Y}"
    except Exception:
        pass
    return str(tm) if pd.notna(tm) else "—"

# Heatmap robusta (con highlight opzionale)
def heatmap(M, title, highlight: Optional[Tuple[int,int]]=None):
    if M is None:
        st.info("—"); return
    dfp = pd.DataFrame(M, index=[str(i) for i in range(M.shape[0])],
                          columns=[str(i) for i in range(M.shape[1])])
    try:
        fig = px.imshow(dfp, text_auto=".1%", aspect="auto",
                        labels=dict(x="Gol Away", y="Gol Home", color="Prob"),
                        title=title)
    except TypeError:
        # fallback per versioni plotly senza text_auto
        fig = px.imshow(dfp, aspect="auto",
                        labels=dict(x="Gol Away", y="Gol Home", color="Prob"),
                        title=title)
    fig.update_traces(hovertemplate="Home %{y} - Away %{x}: %{z:.3%}<extra></extra>")
    if highlight is not None:
        hi, hj = highlight
        fig.add_shape(type="rect", x0=hj-0.5, x1=hj+0.5, y0=hi-0.5, y1=hi+0.5,
                      line=dict(color="#111827", width=3))
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def score_chips(M, home, away, k=5, key_prefix="chips"):
    if M is None:
        return
    lst=[((i,j), float(M[i,j])) for i in range(M.shape[0]) for j in range(M.shape[1])]
    lst.sort(key=lambda x:x[1], reverse=True)

    cols = st.columns(min(k,5))
    for idx, ((i,j), p) in enumerate(lst[:k]):
        outcome = "home" if i>j else ("draw" if i==j else "away")
        active = (st.session_state.get("sel_cell")==(i,j))
        label = f"{i}-{j}  {p*100:.1f}%"
        safe_home = _norm(home).replace(" ","_")
        safe_away = _norm(away).replace(" ","_")
        key = f"{key_prefix}_{safe_home}_{safe_away}_{i}_{j}"
        with cols[idx]:
            clicked = st.button(label, key=key, help="Evidenzia nella heatmap")
            st.markdown(f"<div class='badge {'active' if active else ''} {outcome}'>{label}</div>", unsafe_allow_html=True)
            if clicked:
                st.session_state["sel_cell"] = (i,j)

# ================= SIDEBAR (Scontro Diretto) =================
with st.sidebar:
    st.header("⚙️ Filtri (Scontro Diretto)")
    leagues_present = [d for d in DIV_MAP.keys() if d in df_all["Div"].unique()]
    league_names = [DIV_MAP[d] for d in leagues_present]
    default_idx = league_names.index(DIV_MAP.get("I1","Serie A")) if "I1" in leagues_present else 0
    league_name_sel = st.selectbox("Campionato", league_names, index=default_idx, key="sd_league")
    div_sel = DIV_INV[league_name_sel]
    df = df_all[df_all["Div"]==div_sel].copy()
    st.caption(f"Partite caricate: **{len(df):,}** in {league_name_sel}")
    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
    home_sel = st.selectbox("🏟️ Home", teams, index=0 if len(teams) else 0, key="sd_home")
    away_sel = st.selectbox("🧳 Away", teams, index=1 if len(teams)>1 else 0, key="sd_away")

# ================= TABS =================
tab_match, tab_round = st.tabs(["🎯 Scontro Diretto", "🗓️ Giornata"])

# ----------------- TAB MATCH -----------------
with tab_match:
    lg_logo = get_league_logo(league_name_sel, lg_meta)
    home_logo = get_team_logo(home_sel, tm_meta)
    away_logo = get_team_logo(away_sel, tm_meta)

    c1,c2,c3 = st.columns([3,1,3])
    with c1:
        if home_logo: st.image(home_logo, width=64)
        st.subheader(home_sel)
    with c2:
        if lg_logo: st.image(lg_logo, width=40)
        st.markdown(f"**{league_name_sel}**")
    with c3:
        if away_logo: st.image(away_logo, width=64)
        st.subheader(away_sel)

    h2h = df[(df["HomeTeam"]==home_sel) & (df["AwayTeam"]==away_sel)]
    rec = h2h_record(h2h)
    kickoff_str = get_kickoff(next_df, div_sel, home_sel, away_sel)
    st.markdown(f"<span class='small-muted'>Kickoff: <b>{kickoff_str}</b> • H2H (home perspective): <b>{rec}</b></span>", unsafe_allow_html=True)

    mu_home, mu_away = league_means(df)
    simple = fit_simple(df)
    elo_now = compute_elo(df)
    rho_star = fit_rho_dc_lite_from_simple(df, simple)

    goal_max = st.slider("Range gol per matrice", 3, 7, 5, key="match_goalmax")

    # Calcolo modelli (solo se H2H >= 7 per rispetto del requisito)
    M_s = M_e = M_dcl = M_ens = None
    lh_s = la_s = lh_e = la_e = None
    one_s = one_e = one_dcl = one_ens = None

    if len(h2h) >= 7:
        lh_s, la_s = pois_means_simple(home_sel, away_sel, simple)
        if lh_s is not None and la_s is not None:
            M_s = exact_matrix_indep(lh_s, la_s, goal_max)
            one_s = oneXtwo(M_s)

        lh_e, la_e = elo_to_means(home_sel, away_sel, elo_now, mu_home+mu_away)
        if lh_e is not None and la_e is not None:
            M_e = exact_matrix_indep(lh_e, la_e, goal_max)
            one_e = oneXtwo(M_e)

        if lh_s is not None and la_s is not None:
            M_dcl = exact_matrix_dc_lite(lh_s, la_s, rho_star, goal_max)
            one_dcl = oneXtwo(M_dcl)

        # Ensemble (pesi opzionali)
        mats, ws = [], []
        weights=None
        if os.path.exists("ensemble_weights.json"):
            try:
                with open("ensemble_weights.json","r",encoding="utf-8") as f:
                    weights = json.load(f).get("weights", None)
            except Exception:
                weights=None
        if M_s is not None:  mats.append(M_s);  ws.append(weights.get("Poisson",1.0) if weights else 1.0)
        if M_e is not None:  mats.append(M_e);  ws.append(weights.get("Elo",    1.0) if weights else 1.0)
        if M_dcl is not None: mats.append(M_dcl); ws.append(weights.get("DC",     1.0) if weights else 1.0)
        if mats:
            total_w=sum(ws)
            W=[w/total_w for w in ws] if total_w>0 else [1/len(mats)]*len(mats)
            M_ens = sum(wi*Mi for wi,Mi in zip(W,mats))
            one_ens = oneXtwo(M_ens)
    else:
        st.warning("⚠️ Servono almeno 7 precedenti H2H (stessa direzione) per calcolare un pronostico.")

    # KPI big (mostrati comunque, con '—' se non disponibili)
    fH, ptsH = form_5g(df, home_sel)
    fA, ptsA = form_5g(df, away_sel)
    v_lh = f"{(lh_s if lh_s is not None else 0):.2f}" if lh_s is not None else "—"
    v_la = f"{(la_s if la_s is not None else 0):.2f}" if la_s is not None else "—"
    v_rho = f"{rho_star:.3f}" if rho_star is not None else "—"

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1: st.markdown(f"<div class='kpi'><div class='label'>λH (gol attesi casa)</div><div class='value'>{v_lh}</div></div>", unsafe_allow_html=True)
    with cc2: st.markdown(f"<div class='kpi'><div class='label'>λA (gol attesi trasferta)</div><div class='value'>{v_la}</div></div>", unsafe_allow_html=True)
    with cc3: st.markdown(f"<div class='kpi'><div class='label'>ρ* (DC-lite, lega)</div><div class='value'>{v_rho}</div></div>", unsafe_allow_html=True)
    with cc4: st.markdown(f"<div class='kpi'><div class='label'>Forma 5G (H/A)</div><div class='value'>{fH} ({ptsH}) / {fA} ({ptsA})</div></div>", unsafe_allow_html=True)

    # 1X2 (tema)
    st.subheader("📊 Probabilità 1X2")
    def metric_cards(one_tuple, label):
        if not one_tuple: return
        pH,pX,pA = one_tuple
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(f"<div class='kpi' style='background:rgba(14,168,75,.10);color:var(--col-1)'><div class='label'>{label} — 1</div><div class='value'>{pH*100:.1f}%</div></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='kpi' style='background:rgba(107,114,128,.10);color:var(--col-x)'><div class='label'>{label} — X</div><div class='value'>{pX*100:.1f}%</div></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='kpi' style='background:rgba(239,68,68,.10);color:var(--col-2)'><div class='label'>{label} — 2</div><div class='value'>{pA*100:.1f}%</div></div>", unsafe_allow_html=True)

    metric_cards(one_s,   "Poisson")
    metric_cards(one_e,   "Elo→xG")
    metric_cards(one_dcl, "DC-lite")
    st.markdown("---")
    metric_cards(one_ens, f"Ensemble ({'pesati' if 'weights' in locals() and weights else 'media'})")

    # Heatmaps + Chips (selezione cella condivisa)
    st.subheader("🎯 Risultati esatti — Heatmap (0..N)")
    if "sel_cell" not in st.session_state: st.session_state["sel_cell"] = None

    tabs = st.tabs(["Poisson","Elo→xG","DC-lite","Ensemble"])
    with tabs[0]:
        if M_s is not None:
            score_chips(M_s, home_sel, away_sel, k=5, key_prefix="chip_pois")
        heatmap(M_s, "Poisson", highlight=st.session_state.get("sel_cell"))
    with tabs[1]:
        if M_e is not None:
            score_chips(M_e, home_sel, away_sel, k=5, key_prefix="chip_elo")
        heatmap(M_e, "Elo→xG", highlight=st.session_state.get("sel_cell"))
    with tabs[2]:
        if M_dcl is not None:
            score_chips(M_dcl, home_sel, away_sel, k=5, key_prefix="chip_dcl")
        heatmap(M_dcl, "DC-lite", highlight=st.session_state.get("sel_cell"))
    with tabs[3]:
        if M_ens is not None:
            score_chips(M_ens, home_sel, away_sel, k=5, key_prefix="chip_ens")
        heatmap(M_ens, "Ensemble", highlight=st.session_state.get("sel_cell"))

    # -------- Value vs Mercato --------
    st.subheader("💡 Value vs mercato (se disponibili)")
    def implied(p): return (1.0/p) if (p and p>0) else None
    odds_row=None
    if all(c in df.columns for c in ["AvgH","AvgD","AvgA"]):
        tmp=df[(df["HomeTeam"]==home_sel) & (df["AwayTeam"]==away_sel)].dropna(subset=["AvgH","AvgD","AvgA"])
        if len(tmp)>0: odds_row = tmp.sort_values("Date").iloc[-1]
    if odds_row is not None and one_ens:
        avgH,avgD,avgA=float(odds_row["AvgH"]),float(odds_row["AvgD"]),float(odds_row["AvgA"])
        imp=np.array([implied(avgH),implied(avgD),implied(avgA)],dtype=float)
        if np.isfinite(imp).all() and imp.sum()>0:
            imp=imp/imp.sum()
            pH,pX,pA=one_ens; model=np.array([pH,pX,pA]); edge=(model-imp)*100
            model_pct  = [f"{x:.2f}%" for x in (model*100)]
            market_pct = [f"{x:.2f}%" for x in (imp*100)]
            df_val=pd.DataFrame({"Esito":["1 (Home)","X (Draw)","2 (Away)"],
                                 "Prob modello":model_pct, "Prob mercato":market_pct,
                                 "Edge (p.p.)":np.round(edge,2)})
            st.dataframe(df_val,use_container_width=True,hide_index=True)
        else:
            st.info("Quote non valide per calcolare le probabilità implicite.")
    else:
        st.info("Quote AvgH/AvgD/AvgA non presenti o ensemble non disponibile.")

    # Storico H2H
    st.subheader("📚 Storico H2H (solo Home→Away selezionate)")
    show_cols=[c for c in ["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG","FTR","AvgH","AvgD","AvgA"] if c in h2h.columns]
    st.dataframe(h2h.sort_values("Date", ascending=False)[show_cols], use_container_width=True)

# ----------------- TAB GIORNATA (resta invariato) -----------------
with tab_round:
    st.title("Calcolo per Giornata (next_match.xlsx)")
    next_df_loaded = next_df is not None and set(["Div","Giornata","HomeTeam","AwayTeam"]).issubset(next_df.columns)
    if not next_df_loaded:
        st.warning("Non trovo 'next_match.xlsx' oppure mancano le colonne minime.")
    else:
        leagues_in_next=[d for d in DIV_MAP.keys() if d in next_df["Div"].unique()]
        league_names_next=[DIV_MAP[d] for d in leagues_in_next]
        default_idx_round = league_names_next.index(DIV_MAP.get("I1","Serie A")) if "I1" in leagues_in_next else 0
        league_name_round = st.selectbox("Campionato", league_names_next, index=default_idx_round, key="round_league")
        div_round = DIV_INV[league_name_round]

        lg_logo_round = get_league_logo(league_name_round, lg_meta)
        if lg_logo_round: st.image(lg_logo_round, width=48)
        st.caption(f"Prossimi match: {league_name_round}")

        next_div = next_df[next_df["Div"]==div_round].copy()
        giornate = sorted([g for g in next_div["Giornata"].dropna().unique()])
        if not giornate:
            st.info("Nessuna giornata trovata per il campionato selezionato.")
        else:
            giornata_sel = st.selectbox("Seleziona la Giornata", giornate, index=0, key="round_giornata")

            df_league = df_all[df_all["Div"]==div_round].copy()
            muH_lg, muA_lg = league_means(df_league)
            simple_lg = fit_simple(df_league)
            elo_lg = compute_elo(df_league)
            rho_lg = fit_rho_dc_lite_from_simple(df_league, simple_lg)

            goal_max_round = st.slider("Range gol per matrice (Giornata)", 3, 7, 5, key="round_goalmax")

            weights=None
            if os.path.exists("ensemble_weights.json"):
                try:
                    with open("ensemble_weights.json","r",encoding="utf-8") as f:
                        weights=json.load(f).get("weights",None)
                except Exception:
                    weights=None

            fixtures = next_div[next_div["Giornata"]==giornata_sel][["HomeTeam","AwayTeam"]].copy()
            fixtures["HomeTeam"]=fixtures["HomeTeam"].astype(str).str.strip()
            fixtures["AwayTeam"]=fixtures["AwayTeam"].astype(str).str.strip()

            rows=[]
            for _, row in fixtures.iterrows():
                h,a = row["HomeTeam"], row["AwayTeam"]
                h2h_lg = df_league[(df_league["HomeTeam"]==h) & (df_league["AwayTeam"]==a)]
                if len(h2h_lg) < 7:
                    rows.append({"Home":h,"Away":a,"HomeLogo":get_team_logo(h, tm_meta),"AwayLogo":get_team_logo(a, tm_meta),
                                 "Note":"H2H < 7 (skip)","1":None,"X":None,"2":None,"Top1":None,"Top2":None,"λH":None,"λA":None})
                    continue

                lh_s, la_s = pois_means_simple(h, a, simple_lg)
                if lh_s is None or la_s is None:
                    rows.append({"Home":h,"Away":a,"HomeLogo":get_team_logo(h, tm_meta),"AwayLogo":get_team_logo(a, tm_meta),
                                 "Note":"Team non trovato nello storico","1":None,"X":None,"2":None,"Top1":None,"Top2":None,"λH":None,"λA":None})
                    continue

                M_s = exact_matrix_indep(lh_s, la_s, goal_max_round)
                lh_e, la_e = elo_to_means(h, a, elo_lg, muH_lg+muA_lg)
                M_e = exact_matrix_indep(lh_e, la_e, goal_max_round) if (lh_e is not None and la_e is not None) else None
                M_dcl = exact_matrix_dc_lite(lh_s, la_s, rho_lg, goal_max_round)

                mats=[m for m in [M_s,M_e,M_dcl] if m is not None]
                ws=[]
                if weights:
                    if M_s is not None: ws.append(weights.get("Poisson",1.0))
                    if M_e is not None: ws.append(weights.get("Elo",1.0))
                    if M_dcl is not None: ws.append(weights.get("DC",1.0))
                else:
                    ws=[1.0]*len(mats)

                if mats:
                    total_w=sum(ws); W=[w/total_w for w in ws] if total_w>0 else [1/len(mats)]*len(mats)
                    M_ens=sum(wi*Mi for wi,Mi in zip(W,mats))
                    pH,pX,pA=oneXtwo(M_ens)
                else:
                    pH=pX=pA=None

                # Top-2 scorelines dall'ensemble
                def top2(M):
                    if M is None: return None, None
                    lst=[]
                    for i in range(M.shape[0]):
                        for j in range(M.shape[1]):
                            lst.append(((i,j), float(M[i,j])))
                    lst.sort(key=lambda x:x[1], reverse=True)
                    if not lst: return None,None
                    t1=f"{h} {lst[0][0][0]}-{lst[0][0][1]} {a} ({lst[0][1]*100:.1f}%)"
                    t2=f"{h} {lst[1][0][0]}-{lst[1][0][1]} {a} ({lst[1][1]*100:.1f}%)" if len(lst)>1 else None
                    return t1,t2
                t1,t2 = top2(M_ens)

                rows.append({"Home":h,"Away":a,
                             "HomeLogo":get_team_logo(h, tm_meta), "AwayLogo":get_team_logo(a, tm_meta),
                             "Note":"", "1":round(pH*100,1) if pH is not None else None,
                             "X":round(pX*100,1) if pX is not None else None,
                             "2":round(pA*100,1) if pA is not None else None,
                             "Top1":t1, "Top2":t2, "λH":round(lh_s,2), "λA":round(la_s,2)})

            df_round=pd.DataFrame(rows)
            try:
                st.dataframe(df_round, use_container_width=True, hide_index=True,
                             column_config={
                                 "HomeLogo": st.column_config.ImageColumn(" ", width="small"),
                                 "AwayLogo": st.column_config.ImageColumn("  ", width="small"),
                                 "1": st.column_config.NumberColumn("1", format="%.1f"),
                                 "X": st.column_config.NumberColumn("X", format="%.1f"),
                                 "2": st.column_config.NumberColumn("2", format="%.1f"),
                             })
            except Exception:
                st.dataframe(df_round, use_container_width=True, hide_index=True)

            if len(df_round):
                csv=df_round.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Scarica CSV Giornata", data=csv,
                                   file_name=f"pronostici_{league_name_round}_giornata_{giornata_sel}.csv", mime="text/csv")

# ======= Footer: spiegazione modelli =======
st.markdown("---")
st.info(
    "### Modelli utilizzati\n"
    "- **Poisson**: stima forze att/def e i gol attesi (λH, λA) → matrice esatti con Poisson indipendenti.\n"
    "- **Elo → xG**: rating Elo cronologico (bonus casa) → probabilità 1X2 → gol attesi sul totale di lega.\n"
    "- **DC-lite**: correzione rapida Dixon–Coles sui punteggi bassi con parametro **ρ** stimato su tutta la lega."
)
