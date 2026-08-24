
import os, math, requests, pandas as pd, numpy as np, streamlit as st
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson

st.set_page_config(page_title="Betting Robot", page_icon="⚽", layout="wide")

LEAGUES = {
    "Premier League": ("E0", "soccer_epl"),
    "Bundesliga": ("D1", "soccer_germany_bundesliga"),
    "Serie A": ("I1", "soccer_italy_serie_a"),
    "La Liga": ("SP1", "soccer_spain_la_liga"),
    "Ligue 1": ("F1", "soccer_france_laliga"),
}
# Correct the Ligue 1 API key after defining the dictionary.
LEAGUES["Ligue 1"] = ("F1", "soccer_france_ligue_one")
SEASON_CODES = ["2122","2223","2324","2425","2526"]

MIN_EV = 0.04
MAX_EV = 0.20
MIN_ODDS = 1.45
MAX_ODDS = 4.50
MAX_BETS = 3

def football_data_url(season, code):
    return f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

@st.cache_data(ttl=60*60*12)
def load_history(league):
    code, _ = LEAGUES[league]
    frames=[]
    for season in SEASON_CODES:
        try:
            r=requests.get(football_data_url(season,code),timeout=20)
            r.raise_for_status()
            from io import BytesIO
            frames.append(pd.read_csv(BytesIO(r.content)))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df=pd.concat(frames,ignore_index=True)
    df["Date"]=pd.to_datetime(df["Date"],dayfirst=True,errors="coerce")
    df["FTHG"]=pd.to_numeric(df["FTHG"],errors="coerce")
    df["FTAG"]=pd.to_numeric(df["FTAG"],errors="coerce")
    return df.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"]).sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=60*30)
def get_events(league, api_key):
    _, sport=LEAGUES[league]
    url=f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    r=requests.get(url,params={"apiKey":api_key,"regions":"eu","markets":"h2h,totals","oddsFormat":"decimal"})
    r.raise_for_status()
    return r.json()

def fit_model(history):
    if len(history)<80:
        return None
    h=history.tail(300).copy()
    teams=sorted(set(h.HomeTeam)|set(h.AwayTeam))
    idx={t:i for i,t in enumerate(teams)}
    n=len(teams)
    hi=np.array([idx[x] for x in h.HomeTeam]); ai=np.array([idx[x] for x in h.AwayTeam])
    hg=h.FTHG.to_numpy(float); ag=h.FTAG.to_numpy(float)
    age=(h.Date.max()-h.Date).dt.days.fillna(0).to_numpy()
    w=np.power(0.996,age)
    x0=np.zeros(2*n+1); x0[-1]=math.log(1.2)
    def unpack(p):
        att=p[:n]-np.mean(p[:n])
        deff=p[n:2*n]-np.mean(p[n:2*n])
        return att,deff,math.exp(p[-1])
    def loss(p):
        att,deff,ha=unpack(p)
        lh=np.exp(np.clip(np.log(ha)+att[hi]-deff[hi],-5,3))
        la=np.exp(np.clip(att[ai]-deff[ai],-5,3))
        return -np.sum(w*(hg*np.log(lh)-lh+ag*np.log(la)-la))
    res=minimize(loss,x0,method="L-BFGS-B",options={"maxiter":400})
    if not res.success:
        return None
    att,deff,ha=unpack(res.x)
    return {"teams":teams,"att":dict(zip(teams,att)),"def":dict(zip(teams,deff)),"ha":ha}

def model_probs(model,home,away):
    if model is None or home not in model["teams"] or away not in model["teams"]:
        return None
    lh=math.exp(np.clip(math.log(model["ha"])+model["att"][home]-model["def"][home],-5,3))
    la=math.exp(np.clip(model["att"][away]-model["def"][away],-5,3))
    m=np.outer(poisson.pmf(np.arange(13),lh),poisson.pmf(np.arange(13),la))
    return {
        "HOME":float(np.tril(m,-1).sum()),
        "DRAW":float(np.trace(m)),
        "AWAY":float(np.triu(m,1).sum()),
        "OVER25":float(1-poisson.cdf(2,lh+la)),
        "UNDER25":float(poisson.cdf(2,lh+la)),
        "BTTS_YES":float(1-math.exp(-lh)-math.exp(-la)+math.exp(-(lh+la))),
        "BTTS_NO":float(math.exp(-lh)+math.exp(-la)-math.exp(-(lh+la))),
    }

def ev(p,o):
    return p*o-1

def collect_candidates(league, events, model):
    rows=[]
    for e in events:
        home=e.get("home_team"); away=e.get("away_team")
        p=model_probs(model,home,away)
        if not p: continue
        for b in e.get("bookmakers",[]):
            for m in b.get("markets",[]):
                key=m.get("key")
                for o in m.get("outcomes",[]):
                    name=o.get("name"); odds=float(o.get("price"))
                    label=None; prob=None
                    if key=="h2h":
                        if name==home: label="HOME"
                        elif name==away: label="AWAY"
                        elif name and name.lower()=="draw": label="DRAW"
                        if label: prob=p[label]
                    elif key=="totals" and o.get("point")==2.5:
                        label="OVER25" if name.lower()=="over" else "UNDER25"; prob=p[label]
                    elif key=="btts":
                        label="BTTS_YES" if name.lower() in ("yes","btts yes") else "BTTS_NO"; prob=p[label]
                    if prob is None or not(MIN_ODDS<=odds<=MAX_ODDS): continue
                    value=ev(prob,odds)
                    if MIN_EV<=value<=MAX_EV:
                        rows.append({
                            "Liga":league,"Match":f"{home} – {away}",
                            "Spel":name if key!="totals" else f"{name} {o.get('point')}",
                            "Marknad":label,"Odds":odds,"Modell":prob,
                            "Fair odds":1/prob,"EV":value,
                            "Bookmaker":b.get("title","")
                        })
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows)
    # Best price for each actual selection.
    df=df.sort_values("Odds",ascending=False).drop_duplicates(["Liga","Match","Marknad"])
    return df.sort_values(["EV","Modell"],ascending=False).head(MAX_BETS)

st.title("⚽ Betting Robot")
st.caption("Premier League • Bundesliga • Serie A • La Liga • Ligue 1")

with st.sidebar:
    st.header("Inställningar")
    api_key = st.secrets["ODDS_API_KEY"]
    selected = st.multiselect("Ligorna", list(LEAGUES), default=list(LEAGUES))
    run = st.button("🔎 Hitta dagens bästa spel", use_container_width=True)
    st.divider()
    st.write("**Regler**")
    st.write("Min EV: 4 %")
    st.write("Max EV: 20 %")
    st.write("Max 3 spel")
    st.write("Riktiga spel: AV")

if not run:
    st.info("Tryck på **Hitta dagens bästa spel**. Robotens jobb är att leta efter värde — inte att hitta ett spel varje dag.")
    st.markdown("### Så fungerar det")
    st.markdown("1. Hämtar aktuella odds")
    st.markdown("2. Räknar sannolikheter från historiska matcher")
    st.markdown("3. Jämför sannolikheten med oddset")
    st.markdown("4. Rankar de bästa värdena")
    st.markdown("5. Säger **NO BET** om inget håller måttet")
else:
    if not api_key:
        st.error("Du behöver lägga in din Odds API-nyckel i rutan till vänster.")
        st.stop()
    all_rows=[]
    progress=st.progress(0)
    for i,league in enumerate(selected):
        try:
            hist=load_history(league)
            model=fit_model(hist)
            events=get_events(league,api_key)
            x=collect_candidates(league,events,model)
            if not x.empty: all_rows.append(x)
        except Exception as e:
            st.warning(f"{league}: {e}")
        progress.progress((i+1)/len(selected))
    progress.empty()

    if not all_rows:
        st.error("🔴 NO BET TODAY")
        st.write("Inga spel passerade robotens krav.")
    else:
        df=pd.concat(all_rows,ignore_index=True).sort_values("EV",ascending=False).head(MAX_BETS)
        st.success(f"🟢 {len(df)} spel hittades")
        for i,(_,r) in enumerate(df.iterrows(),1):
            with st.container(border=True):
                st.subheader(f"#{i}  {r['Liga']} — {r['Match']}")
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Spel",r["Spel"])
                c2.metric("Odds",f"{r['Odds']:.2f}")
                c3.metric("Modell",f"{r['Modell']:.1%}")
                c4.metric("EV",f"{r['EV']:.1%}")
                st.write(f"Fair odds: **{r['Fair odds']:.2f}**  •  Bäst odds hos: **{r['Bookmaker']}**")
        st.caption("Detta är en forsknings-/paper-bettingmodell. Den placerar inga riktiga spel.")
