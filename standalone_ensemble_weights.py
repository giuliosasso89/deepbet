# standalone_ensemble_weights.py
# Uso: python standalone_ensemble_weights.py
# Requisiti: pandas numpy scipy openpyxl

import json
import numpy as np
import pandas as pd
from scipy.stats import poisson

EXCEL_PATH = "Serie_A_2014_2025.xlsx"
GOAL_MAX = 6
SEASONS_TARGET = ["2023/24", "2024/25"]

def load_data(path):
    df = pd.read_excel(path)
    ren = {}
    if "HG" in df.columns and "FTHG" not in df.columns: ren["HG"] = "FTHG"
    if "AG" in df.columns and "FTAG" not in df.columns: ren["AG"] = "FTAG"
    if "Res" in df.columns and "FTR"  not in df.columns: ren["Res"] = "FTR"
    df = df.rename(columns=ren)
    if "Date" not in df.columns:
        raise ValueError("Colonna 'Date' mancante")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for c in ["HomeTeam","AwayTeam","FTHG","FTAG"]:
        if c not in df.columns: raise ValueError(f"Colonna mancante: {c}")
    df = df.dropna(subset=["FTHG","FTAG"]).copy()
    df["FTHG"] = df["FTHG"].astype(int); df["FTAG"] = df["FTAG"].astype(int)
    if "FTR" not in df.columns:
        df["FTR"] = np.where(df["FTHG"]>df["FTAG"],"H",np.where(df["FTHG"]<df["FTAG"],"A","D"))
    def season_label(d):
        if pd.isna(d): return None
        y = d.year
        start = y if d.month >= 7 else y-1
        return f"{start}/{str((start+1)%100).zfill(2)}"
    df["Season"] = df["Date"].apply(season_label)
    return df

# --- Poisson att/def ---
def fit_simple_att_def(train_df, reg=0.1):
    teams = sorted(set(train_df["HomeTeam"]).union(train_df["AwayTeam"]))
    idx = {t:i for i,t in enumerate(teams)}; n=len(teams)
    gfH=gaH=nH=np.zeros(n); gfA=gaA=nA=np.zeros(n)
    for _,r in train_df.iterrows():
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

def means_poisson_simple(params, home, away):
    hi=params["idx"].get(home); ai=params["idx"].get(away)
    if hi is None or ai is None: return None, None
    lh = max(0.05, params["muH"]*params["att"][hi]*params["def"][ai])
    la = max(0.05, params["muA"]*params["att"][ai]*params["def"][hi])
    return lh, la

# --- Elo → xG → Poisson ---
def compute_elo(train_df, K=20, home_adv_elo=60):
    teams = sorted(set(train_df["HomeTeam"]).union(train_df["AwayTeam"]))
    elo = {t:1500.0 for t in teams}
    df_ord = train_df.sort_values("Date")
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

def elo_to_means(elo_now, home, away, mu_total):
    if home not in elo_now or away not in elo_now: return None, None
    Rh=elo_now[home]+60; Ra=elo_now[away]
    pH = 1/(1+10**((Ra-Rh)/400)); pA = 1-pH; pD = 0.25
    wH = pH + 0.5*pD; wA = pA + 0.5*pD
    s = wH + wA
    lh = max(0.05, mu_total*(wH/s))
    la = max(0.05, mu_total*(wA/s))
    return lh, la

# --- DC-lite (ρ veloce) ---
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

def fit_rho_dc_lite(train_df, means_fn, goal_max=6, grid=np.linspace(-0.2, 0.2, 41)):
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

# --- Utility comuni ---
def exact_matrix(lh, la, goal_max=6):
    i=np.arange(0,goal_max+1)
    M=np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    return M/M.sum()

def argmax_score(M):
    idx = np.unravel_index(np.argmax(M), M.shape)
    return int(idx[0]), int(idx[1])

def h2h_count_before(train_df, home, away):
    return len(train_df[(train_df["HomeTeam"]==home) & (train_df["AwayTeam"]==away)])

def season_start_date(season_label):
    start_year = int(season_label.split("/")[0])
    return pd.Timestamp(year=start_year, month=7, day=1)

def evaluate_season(df, season_label, verbose=True):
    start = season_start_date(season_label)
    train_df = df[df["Date"] < start].copy()
    test_df  = df[df["Season"] == season_label].copy()

    if verbose:
        print(f"\n=== Stagione {season_label} ===")
        print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    simp = fit_simple_att_def(train_df)
    elo  = compute_elo(train_df)
    mu_total = train_df["FTHG"].mean() + train_df["FTAG"].mean()
    # stimo una sola volta rho* per la stagione
    rho_star = fit_rho_dc_lite(train_df, lambda h,a: means_poisson_simple(simp, h, a), goal_max=GOAL_MAX)

    correct = {"Poisson":0, "Elo":0, "DC":0}
    total   = {"Poisson":0, "Elo":0, "DC":0}

    for _, r in test_df.iterrows():
        home, away = r["HomeTeam"], r["AwayTeam"]
        if h2h_count_before(train_df, home, away) < 7:
            continue
        gh, ga = int(r["FTHG"]), int(r["FTAG"])

        # Poisson
        lh, la = means_poisson_simple(simp, home, away)
        if lh and la:
            M = exact_matrix(lh, la, goal_max=GOAL_MAX)
            ph, pa = argmax_score(M)
            total["Poisson"] += 1
            if (ph,pa)==(gh,ga): correct["Poisson"] += 1

        # Elo
        lh, la = elo_to_means(elo, home, away, mu_total)
        if lh and la:
            M = exact_matrix(lh, la, goal_max=GOAL_MAX)
            ph, pa = argmax_score(M)
            total["Elo"] += 1
            if (ph,pa)==(gh,ga): correct["Elo"] += 1

        # DC-lite
        lh, la = means_poisson_simple(simp, home, away)
        if lh and la:
            M = exact_matrix_dc_lite(lh, la, rho_star, goal_max=GOAL_MAX)
            ph, pa = argmax_score(M)
            total["DC"] += 1
            if (ph,pa)==(gh,ga): correct["DC"] += 1

    acc = {}
    for k in ["Poisson","Elo","DC"]:
        acc[k] = (correct[k]/total[k]) if total[k] > 0 else None

    if verbose:
        for k in ["Poisson","Elo","DC"]:
            if acc[k] is None:
                print(f"  {k:8s}: n/a (0 match validi)")
            else:
                print(f"  {k:8s}: {acc[k]*100:5.2f}%  (su {total[k]} partite)")

    return acc, total

def main():
    df = load_data(EXCEL_PATH)

    seasonal_acc, seasonal_tot = [], []
    for s in SEASONS_TARGET:
        acc, tot = evaluate_season(df, s, verbose=True)
        seasonal_acc.append(acc); seasonal_tot.append(tot)

    models = ["Poisson","Elo","DC"]
    num = {m:0.0 for m in models}; den = {m:0 for m in models}
    for acc, tot in zip(seasonal_acc, seasonal_tot):
        for m in models:
            if acc[m] is not None:
                num[m] += acc[m]*tot[m]
                den[m] += tot[m]

    overall_acc = {m: (num[m]/den[m] if den[m]>0 else None) for m in models}

    print("\n=== Accuratezze complessive (media pesata su #match validi) ===")
    for m in models:
        print(f"  {m:8s}: {'n/a' if overall_acc[m] is None else f'{overall_acc[m]*100:5.2f}%'}")

    used = [m for m in models if overall_acc[m] is not None]
    if not used:
        print("\n[ERRORE] Nessun modello con accuratezza valida."); return
    raw = np.array([overall_acc[m] for m in used], dtype=float)
    weights = raw / raw.sum()
    best_model = used[int(np.argmax(raw))]

    print("\n=== Pesi suggeriti per ENSEMBLE (normalizzati) ===")
    weights_dict = {m: float(w) for m, w in zip(used, weights)}
    for m in used:
        print(f"  w_{m:8s} = {weights_dict[m]:.3f}")
    print(f"\n➡️ Modello più accurato (2023/24 + 2024/25): **{best_model}**")

    out = {
        "weights": weights_dict,
        "overall_accuracy": {m:(None if overall_acc[m] is None else float(overall_acc[m])) for m in models},
        "goal_max": GOAL_MAX,
        "seasons_used": SEASONS_TARGET,
        "best_model": best_model
    }
    with open("ensemble_weights.json","w",encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\nSalvato: ensemble_weights.json')

if __name__ == "__main__":
    main()
