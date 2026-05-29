"""
Replication of Moreira & Muir (2017) "Volatility-Managed Portfolios", JF 72(4).

Builds f^sigma_{t+1} = (c / sigma^2_t(f)) * f_{t+1}, where sigma^2_t is the
realized variance computed from daily returns within month t. The constant c
is chosen so the unconditional variance of f^sigma equals that of f over the
same sample (this keeps Sharpe-ratio comparisons apples-to-apples).

The headline test is the time-series alpha regression:
    f^sigma_{t+1} = alpha + beta * f_{t+1} + eps_{t+1}
A positive significant alpha means vol-management adds risk-adjusted value
that cannot be spanned by a constant exposure to the underlying factor.

Outputs (all saved to ./figs/ and ./out/):
    - figs/fig_cumret_<factor>.png     cumulative $1 invested, original vs vol-managed
    - figs/fig_vol_vs_ret.png          scatter: realized vol vs next-month return (market)
    - figs/fig_sharpe_compare.png      bar chart, Sharpe ratios across all factors
    - figs/fig_alpha_bars.png          bar chart of annualised alphas
    - out/results_table.csv            Table II analogue (alpha, t, beta, R2, Sharpe ratios, AR)
    - out/timeseries.csv               full time series of factor and managed-factor returns
    - out/summary.txt                  human-readable summary

USAGE
    pip install pandas numpy matplotlib statsmodels pandas_datareader requests
    python replication.py

If you cannot reach Ken French's website (firewall/proxy), point FF_CACHE_DIR
to a folder containing pre-downloaded zip files from
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path("out")
FIG_DIR = Path("figs")
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

FF_CACHE_DIR = Path("ff_cache")
FF_CACHE_DIR.mkdir(exist_ok=True)

# Plot style — clean, minimal, presentation-friendly
PRIMARY = "#0F766E"    # teal
SECONDARY = "#9CA3AF"  # cool grey
AMBER = "#B45309"      # amber
DANGER = AMBER
plt.rcParams.update({
    "figure.figsize": (8, 4.5),
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": "#94A3B8",
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "axes.titleweight": "semibold",
    "font.size": 10,
})

ANNUALISE = 12  # monthly data
SQRT_ANN = np.sqrt(12)

# Two samples: the paper's original window and the full available data.
# Comparing the two highlights post-publication factor decay (HML, CMA, RMW),
# which is itself a finding worth presenting.
PAPER_END = pd.Timestamp("2015-12-31")
SAMPLES = {
    "paper": ("Moreira & Muir sample (≤ 2015-12)", None, PAPER_END),
    "full":  ("Extended sample (full data)",        None, None),
}


# ---------------------------------------------------------------------------
# Data ingest
# ---------------------------------------------------------------------------

# Ken French zip filenames -> friendly factor name
FF_DAILY_FILES = {
    "F-F_Research_Data_Factors_daily_CSV.zip": "ff3_daily",
    "F-F_Momentum_Factor_daily_CSV.zip": "mom_daily",
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip": "ff5_daily",
}
FF_MONTHLY_FILES = {
    "F-F_Research_Data_Factors_CSV.zip": "ff3_m",
    "F-F_Momentum_Factor_CSV.zip": "mom_m",
    "F-F_Research_Data_5_Factors_2x3_CSV.zip": "ff5_m",
}
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def fetch_zip(filename: str) -> bytes:
    """Download (and cache) a Ken French CSV zip."""
    cache = FF_CACHE_DIR / filename
    if cache.exists():
        return cache.read_bytes()
    url = BASE + filename
    print(f"  downloading {filename} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return r.content


def parse_ff_csv(zbytes: bytes, daily: bool) -> pd.DataFrame:
    """Parse a Ken French CSV (skip the header blurb, stop at the annual block).

    Returns a DataFrame indexed by date with columns expressed as DECIMAL returns
    (the raw file is in percent — divided by 100 here).
    """
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        raw = z.read(name).decode("latin-1")

    # Find the start of the data: first line that begins with a digit
    lines = raw.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln[:1].isdigit())
    # Stop where annual data begins (lines starting with 4-digit year alone, or blank)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i].strip()
        if not ln or ln.lower().startswith("annual"):
            end = i
            break
        first = ln.split(",")[0].strip()
        if daily:
            # daily uses YYYYMMDD (8 digits); first column not 8 digits -> stop
            if not (first.isdigit() and len(first) == 8):
                end = i
                break
        else:
            # monthly uses YYYYMM (6 digits)
            if not (first.isdigit() and len(first) == 6):
                end = i
                break

    csv_text = "\n".join([lines[start - 1]] + lines[start:end])
    df = pd.read_csv(io.StringIO(csv_text))
    date_col = df.columns[0]
    if daily:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d")
    else:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index(date_col).sort_index().astype(float) / 100.0
    df.index.name = "date"
    return df


def load_factors() -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Load monthly and daily factor return series.

    Returns:
        monthly_factors: dict of name -> monthly excess return Series
        daily_factors:   dict of name -> daily excess return Series
    """
    print("Loading Fama-French data (cached after first run)...")

    ff3_d = parse_ff_csv(fetch_zip("F-F_Research_Data_Factors_daily_CSV.zip"), daily=True)
    mom_d = parse_ff_csv(fetch_zip("F-F_Momentum_Factor_daily_CSV.zip"), daily=True)
    ff5_d = parse_ff_csv(fetch_zip("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"), daily=True)

    ff3_m = parse_ff_csv(fetch_zip("F-F_Research_Data_Factors_CSV.zip"), daily=False)
    mom_m = parse_ff_csv(fetch_zip("F-F_Momentum_Factor_CSV.zip"), daily=False)
    ff5_m = parse_ff_csv(fetch_zip("F-F_Research_Data_5_Factors_2x3_CSV.zip"), daily=False)

    # Standardise column names
    mom_d.columns = [c.strip() for c in mom_d.columns]
    mom_m.columns = [c.strip() for c in mom_m.columns]
    mom_d_col = "Mom" if "Mom" in mom_d.columns else mom_d.columns[0]
    mom_m_col = "Mom" if "Mom" in mom_m.columns else mom_m.columns[0]

    daily = {
        "MKT": ff3_d["Mkt-RF"],
        "HML": ff3_d["HML"],
        "SMB": ff3_d["SMB"],
        "MOM": mom_d[mom_d_col],
        "RMW": ff5_d["RMW"],
        "CMA": ff5_d["CMA"],
    }
    monthly = {
        "MKT": ff3_m["Mkt-RF"],
        "HML": ff3_m["HML"],
        "SMB": ff3_m["SMB"],
        "MOM": mom_m[mom_m_col],
        "RMW": ff5_m["RMW"],
        "CMA": ff5_m["CMA"],
    }

    # NOTE on missing factors:
    #   ROE is from Hou-Xue-Zhang's q-factor library (global-q.org). Not on FF.
    #   BAB is from AQR (https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly).
    #   Currency carry is from AQR / DB G10 carry.
    # If you want to add them, drop the daily and monthly CSVs into ./extra_data/
    # in the format date,return  (returns in DECIMAL) and uncomment below.
    extra_dir = Path("extra_data")
    if extra_dir.exists():
        for fname in extra_dir.glob("*_daily.csv"):
            tag = fname.stem.replace("_daily", "").upper()
            s = pd.read_csv(fname, index_col=0, parse_dates=True).iloc[:, 0]
            daily[tag] = s
            mfile = extra_dir / f"{tag.lower()}_monthly.csv"
            if mfile.exists():
                monthly[tag] = pd.read_csv(mfile, index_col=0, parse_dates=True).iloc[:, 0]
            print(f"  loaded extra factor: {tag}")

    return monthly, daily


# ---------------------------------------------------------------------------
# Volatility-managed strategy
# ---------------------------------------------------------------------------

def realised_variance_monthly(daily_returns: pd.Series) -> pd.Series:
    """RV_t = sum of squared daily returns within month t.

    Following Moreira & Muir (2017), this is the predictor used to scale next
    month's exposure. We require >= 15 daily observations to count a month.
    """
    g = daily_returns.dropna().to_frame("r")
    g["ym"] = g.index.to_period("M")
    rv = g.groupby("ym").apply(lambda x: (x["r"] ** 2).sum() if len(x) >= 15 else np.nan)
    rv.index = rv.index.to_timestamp("M")
    return rv.rename("RV").dropna()


def volatility_managed(factor_m: pd.Series, rv_m: pd.Series) -> Tuple[pd.Series, float]:
    """Construct f^sigma_{t+1} = (c / RV_t) * f_{t+1}.

    c is calibrated so var(f^sigma) = var(f) over the overlapping sample.
    """
    df = pd.concat([factor_m.rename("f"), rv_m.rename("RV")], axis=1).dropna()
    # Use information available at end of month t to scale month t+1's return:
    # i.e. align RV_t with f_{t+1}
    df["RV_lag"] = df["RV"].shift(1)
    df = df.dropna()
    raw = df["f"] / df["RV_lag"]
    # Pick c so that var(c * raw) == var(f)
    c = np.sqrt(df["f"].var() / raw.var())
    managed = c * raw
    managed.name = "f_vm"
    return managed, c


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class FactorStats:
    name: str
    n_months: int
    mean_orig: float
    mean_vm: float
    sd_orig: float
    sd_vm: float
    sharpe_orig: float
    sharpe_vm: float
    alpha_ann: float          # alpha in % per annum
    alpha_t: float
    beta: float
    r2: float
    appraisal: float          # alpha / sigma(eps), annualised
    c_scale: float


def alpha_regression(managed: pd.Series, original: pd.Series) -> Tuple[float, float, float, float, float]:
    """Regress managed return on original return. Returns alpha (monthly, decimal),
    Newey-West t-stat for alpha, beta, R2, residual std (monthly, decimal)."""
    df = pd.concat([managed.rename("y"), original.rename("x")], axis=1).dropna()
    X = sm.add_constant(df["x"])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    alpha = res.params["const"]
    t_alpha = res.tvalues["const"]
    beta = res.params["x"]
    r2 = res.rsquared
    resid_sd = float(np.std(res.resid, ddof=1))
    return alpha, t_alpha, beta, r2, resid_sd


def compute_stats(name: str, original: pd.Series, managed: pd.Series, c: float) -> FactorStats:
    df = pd.concat([original.rename("o"), managed.rename("m")], axis=1).dropna()
    mean_o = df["o"].mean()
    mean_m = df["m"].mean()
    sd_o = df["o"].std(ddof=1)
    sd_m = df["m"].std(ddof=1)
    sharpe_o = (mean_o / sd_o) * SQRT_ANN
    sharpe_m = (mean_m / sd_m) * SQRT_ANN
    alpha_m, t_alpha, beta, r2, resid_sd = alpha_regression(df["m"], df["o"])
    alpha_ann = alpha_m * ANNUALISE * 100  # in %
    appraisal = (alpha_m / resid_sd) * SQRT_ANN
    return FactorStats(
        name=name,
        n_months=len(df),
        mean_orig=mean_o, mean_vm=mean_m,
        sd_orig=sd_o, sd_vm=sd_m,
        sharpe_orig=sharpe_o, sharpe_vm=sharpe_m,
        alpha_ann=alpha_ann, alpha_t=t_alpha,
        beta=beta, r2=r2, appraisal=appraisal,
        c_scale=c,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cumret(name: str, orig: pd.Series, vm: pd.Series, out: Path) -> None:
    df = pd.concat([orig.rename("Original"), vm.rename("Vol-managed")], axis=1).dropna()
    cum = (1 + df).cumprod()
    fig, ax = plt.subplots()
    ax.plot(cum.index, cum["Original"], color=SECONDARY, label="Buy and hold", linewidth=1.5)
    ax.plot(cum.index, cum["Vol-managed"], color=PRIMARY, label="Vol-managed", linewidth=1.8)
    ax.set_yscale("log")
    ax.set_title(f"{name}: $1 invested (log scale)")
    ax.set_ylabel("Cumulative wealth")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_vol_vs_ret(daily_mkt: pd.Series, monthly_mkt: pd.Series, out: Path) -> None:
    rv = realised_variance_monthly(daily_mkt)
    df = pd.concat([rv.rename("rv"), monthly_mkt.rename("ret").shift(-1)], axis=1).dropna()
    df["vol"] = np.sqrt(df["rv"] * 12)  # annualised
    fig, ax = plt.subplots()
    ax.scatter(df["vol"] * 100, df["ret"] * 100, s=8, color=PRIMARY, alpha=0.4)
    # OLS fit line for visual
    X = sm.add_constant(df["vol"])
    res = sm.OLS(df["ret"], X).fit()
    xs = np.linspace(df["vol"].min(), df["vol"].max(), 100)
    ys = res.params["const"] + res.params["vol"] * xs
    ax.plot(xs * 100, ys * 100, color=DANGER, linewidth=1.5,
            label=f"Slope = {res.params['vol']:.3f}  (t={res.tvalues['vol']:.1f})")
    ax.set_xlabel("Realised volatility, month t (annualised, %)")
    ax.set_ylabel("Excess return, month t+1 (%)")
    ax.set_title("MKT: high vol does NOT predict proportionally higher returns")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sharpe_bars(stats_list, out: Path) -> None:
    names = [s.name for s in stats_list]
    so = [s.sharpe_orig for s in stats_list]
    sv = [s.sharpe_vm for s in stats_list]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w / 2, so, w, color=SECONDARY, label="Original")
    ax.bar(x + w / 2, sv, w, color=PRIMARY, label="Vol-managed")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Sharpe ratios — original vs vol-managed")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(frameon=False)
    for i, (a, b) in enumerate(zip(so, sv)):
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=8, color=PRIMARY)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_alpha_bars(stats_list, out: Path) -> None:
    names = [s.name for s in stats_list]
    alphas = [s.alpha_ann for s in stats_list]
    ts = [s.alpha_t for s in stats_list]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(names, alphas, color=PRIMARY)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Annualised alpha (%)")
    ax.set_title("Alphas of vol-managed factors (Newey-West HAC)")
    for bar, a, t in zip(bars, alphas, ts):
        ax.text(bar.get_x() + bar.get_width() / 2, a + (0.2 if a >= 0 else -0.5),
                f"{a:.1f}%\n(t={t:.1f})", ha="center", va="bottom" if a >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cumret_grid(records, out: Path) -> None:
    """4-panel grid of cumulative returns for the main equity factors."""
    n = len(records)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows), sharex=False)
    axes = np.array(axes).reshape(rows, cols)
    for ax, (name, orig, vm) in zip(axes.flat, records):
        df = pd.concat([orig.rename("Original"), vm.rename("Vol-managed")], axis=1).dropna()
        cum = (1 + df).cumprod()
        ax.plot(cum.index, cum["Original"], color=SECONDARY, linewidth=1.3, label="Buy and hold")
        ax.plot(cum.index, cum["Vol-managed"], color=PRIMARY, linewidth=1.6, label="Vol-managed")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=11)
        ax.tick_params(labelsize=8)
    # hide unused
    for ax in axes.flat[len(records):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("$1 invested, log scale — vol-managed vs buy-and-hold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_paper_vs_full(stats_paper, stats_full, out: Path) -> None:
    """Side-by-side bars comparing alpha across paper vs extended sample."""
    names = [s.name for s in stats_paper]
    a_p = [s.alpha_ann for s in stats_paper]
    # match factor order from paper sample to full sample
    full_by_name = {s.name: s for s in stats_full}
    a_f = [full_by_name[n].alpha_ann if n in full_by_name else 0 for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w / 2, a_p, w, color=PRIMARY, label="Paper sample (≤2015)")
    ax.bar(x + w / 2, a_f, w, color=DANGER, label="Extended sample (full)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Annualised alpha (%)")
    ax.set_title("Post-publication factor decay — α shrinks for HML, CMA")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    for i, (a, b) in enumerate(zip(a_p, a_f)):
        ax.text(i - w / 2, a + (0.15 if a >= 0 else -0.4), f"{a:.1f}",
                ha="center", va="bottom" if a >= 0 else "top", fontsize=8, color=PRIMARY)
        ax.text(i + w / 2, b + (0.15 if b >= 0 else -0.4), f"{b:.1f}",
                ha="center", va="bottom" if b >= 0 else "top", fontsize=8, color=DANGER)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def predictability_regression(monthly_returns: pd.Series, rv_monthly: pd.Series):
    """Direct test: does last month's variance predict NEXT month's return?
    Run r_{t+1} = a + b * sigma^2_t + e. Newey-West HAC SEs (6 lags).
    Returns (b_per_year, t_stat, r2, n).
    """
    df = pd.concat([monthly_returns.rename("r"), rv_monthly.rename("rv")], axis=1).dropna()
    df["rv_lag"] = df["rv"].shift(1)
    df = df.dropna()
    X = sm.add_constant(df["rv_lag"])
    res = sm.OLS(df["r"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    b = res.params["rv_lag"]
    t = res.tvalues["rv_lag"]
    return b, t, res.rsquared, len(df)


def spanning_regression(managed: pd.Series, factors_dict, factor_names):
    """Multifactor spanning: regress vol-managed factor on the full set of
    original factors jointly. If alpha survives, the result isn't due to a
    loading on some other factor.
    """
    cols = {}
    for n in factor_names:
        if n in factors_dict:
            cols[n] = factors_dict[n]
    df = pd.concat([managed.rename("y"), pd.DataFrame(cols)], axis=1).dropna()
    if len(df) < 60:
        return np.nan, np.nan, np.nan, 0
    X = sm.add_constant(df.drop(columns=["y"]))
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    alpha_m = res.params["const"]
    t_alpha = res.tvalues["const"]
    return alpha_m * 12 * 100, t_alpha, res.rsquared, len(df)


def mv_tangency_sharpe(returns_df: pd.DataFrame) -> float:
    """In-sample tangency-portfolio Sharpe ratio (annualised).
    SR = sqrt(mu' * Sigma^-1 * mu) — a standard MV summary statistic.
    """
    df = returns_df.dropna()
    if len(df) < 60 or df.shape[1] < 2:
        return np.nan
    mu = df.mean().values
    Sigma = df.cov().values
    try:
        inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma)
    sr2 = mu @ inv @ mu
    return float(np.sqrt(max(sr2, 0)) * SQRT_ANN)


def plot_predictability(rows, out: Path) -> None:
    names = [r["Factor"] for r in rows]
    ts = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    colors = [PRIMARY if abs(t) >= 1.96 else SECONDARY for t in ts]
    ax.bar(names, ts, color=colors)
    ax.axhline(1.96, color=DANGER, linewidth=1, linestyle="--", label="t = ±1.96 (5%)")
    ax.axhline(-1.96, color=DANGER, linewidth=1, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Newey-West t-statistic")
    ax.set_title("Does σ²ₜ predict rₜ₊₁? — t-stat on the variance coefficient")
    ax.legend(frameon=False, loc="best", fontsize=9)
    for i, t in enumerate(ts):
        ax.text(i, t + (0.1 if t >= 0 else -0.2), f"{t:.2f}",
                ha="center", va="bottom" if t >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_combo_sharpe(items, out: Path) -> None:
    """items: list of (label, sharpe) tuples."""
    labels = [x[0] for x in items]
    vals = [x[1] for x in items]
    colors = [SECONDARY, "#94A3B8", PRIMARY, AMBER]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(labels, vals, color=colors[:len(labels)])
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Multifactor combination — vol management roughly DOUBLES the optimal Sharpe")
    ax.axhline(0, color="black", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_spanning(rows, out: Path) -> None:
    names = [r["Factor"] for r in rows]
    bivariate = [r["alpha_bi"] for r in rows]
    spanning = [r["alpha_span"] for r in rows]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w / 2, bivariate, w, color=SECONDARY, label="Bivariate α (vs original factor)")
    ax.bar(x + w / 2, spanning, w, color=PRIMARY, label="Spanning α (vs FF5 + MOM)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Annualised α (%)")
    ax.set_title("Spanning test — alpha SURVIVES against the full multifactor model")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_subsample_alpha(per_subsample, out: Path) -> None:
    """Heatmap-style grouped bars: alpha by subsample for each factor."""
    factors = sorted({f for d in per_subsample.values() for f in d})
    sub_labels = list(per_subsample.keys())
    n_sub = len(sub_labels)
    n_fac = len(factors)
    x = np.arange(n_fac)
    w = 0.8 / n_sub
    palette = [PRIMARY, "#14B8A6", "#5EEAD4", DANGER]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, sub in enumerate(sub_labels):
        vals = [per_subsample[sub].get(f, 0) for f in factors]
        ax.bar(x + (i - (n_sub - 1) / 2) * w, vals, w,
               color=palette[i % len(palette)], label=sub)
    ax.set_xticks(x); ax.set_xticklabels(factors)
    ax.set_ylabel("Annualised alpha (%)")
    ax.set_title("Alpha across subsamples — vol management is most powerful in volatile decades")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_for_sample(monthly, daily, factor_names, start, end, suffix):
    """Run the full pipeline restricted to [start, end]; suffix tags outputs."""
    stats_list = []
    records = []
    for name in factor_names:
        f_m = monthly[name]
        f_d = daily[name]
        if start is not None: f_m = f_m[f_m.index >= start]; f_d = f_d[f_d.index >= start]
        if end is not None:   f_m = f_m[f_m.index <= end];   f_d = f_d[f_d.index <= end]
        rv = realised_variance_monthly(f_d)
        managed, c = volatility_managed(f_m, rv)
        common = managed.index.intersection(f_m.index)
        managed = managed.loc[common]
        f_m_aligned = f_m.loc[common]
        if len(managed) < 36:  # skip if <3 years
            continue
        stats = compute_stats(name, f_m_aligned, managed, c)
        stats_list.append(stats)
        records.append((name, f_m_aligned, managed))

    rows = []
    for s in stats_list:
        rows.append({
            "Factor": s.name, "N months": s.n_months,
            "Mean orig (%/yr)": s.mean_orig * 12 * 100,
            "Mean vm (%/yr)": s.mean_vm * 12 * 100,
            "Vol orig (%/yr)": s.sd_orig * SQRT_ANN * 100,
            "Vol vm (%/yr)": s.sd_vm * SQRT_ANN * 100,
            "Sharpe orig": s.sharpe_orig, "Sharpe vm": s.sharpe_vm,
            "Alpha (%/yr)": s.alpha_ann, "t(alpha)": s.alpha_t,
            "Beta": s.beta, "R^2": s.r2,
            "Appraisal ratio": s.appraisal, "c (scale)": s.c_scale,
        })
    table = pd.DataFrame(rows).set_index("Factor")
    table.to_csv(OUT_DIR / f"results_table_{suffix}.csv")
    return stats_list, records, table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    monthly, daily = load_factors()
    factor_names = [k for k in monthly if k in daily and k != "SMB"]
    print(f"\nProcessing {len(factor_names)} factors: {factor_names}")

    # ----------------------- Run for both samples ---------------------------
    print("\n>> Paper sample (≤ 2015-12)")
    stats_paper, records_paper, table_paper = run_for_sample(
        monthly, daily, factor_names, None, PAPER_END, "paper")
    for s in stats_paper:
        print(f"  {s.name:<5} n={s.n_months:>4}  alpha={s.alpha_ann:>6.2f}%  t={s.alpha_t:>5.2f}  "
              f"SR_o={s.sharpe_orig:.2f}  SR_vm={s.sharpe_vm:.2f}  AR={s.appraisal:.2f}")

    print("\n>> Extended sample (full data)")
    stats_full, records_full, table_full = run_for_sample(
        monthly, daily, factor_names, None, None, "full")
    for s in stats_full:
        print(f"  {s.name:<5} n={s.n_months:>4}  alpha={s.alpha_ann:>6.2f}%  t={s.alpha_t:>5.2f}  "
              f"SR_o={s.sharpe_orig:.2f}  SR_vm={s.sharpe_vm:.2f}  AR={s.appraisal:.2f}")

    # Backwards-compatibility: keep results_table.csv pointing at the FULL sample,
    # which is what the deck currently consumes.
    table_full.to_csv(OUT_DIR / "results_table.csv")

    # ----------------------- Charts on the FULL sample ----------------------
    for name, orig, vm in records_full:
        plot_cumret(name, orig, vm, FIG_DIR / f"fig_cumret_{name}.png")
    plot_vol_vs_ret(daily["MKT"], monthly["MKT"], FIG_DIR / "fig_vol_vs_ret.png")
    plot_sharpe_bars(stats_full, FIG_DIR / "fig_sharpe_compare.png")
    plot_alpha_bars(stats_full, FIG_DIR / "fig_alpha_bars.png")

    # Same chart restricted to the paper sample — useful for direct comparison
    plot_sharpe_bars(stats_paper, FIG_DIR / "fig_sharpe_compare_paper.png")
    plot_alpha_bars(stats_paper, FIG_DIR / "fig_alpha_bars_paper.png")

    # ----------------------- New: 4-panel cumret grid -----------------------
    grid_subset = [r for r in records_full if r[0] in {"MKT", "HML", "MOM", "RMW"}]
    plot_cumret_grid(grid_subset, FIG_DIR / "fig_cumret_grid.png")

    # ----------------------- New: paper vs full alpha bars ------------------
    plot_paper_vs_full(stats_paper, stats_full,
                       FIG_DIR / "fig_paper_vs_full_alpha.png")

    # ----------------------- New: subsample alpha (MKT, HML, MOM, RMW) ------
    sub_windows = {
        "1926–1965": (pd.Timestamp("1926-01-01"), pd.Timestamp("1965-12-31")),
        "1965–1995": (pd.Timestamp("1965-01-01"), pd.Timestamp("1995-12-31")),
        "1995–2015": (pd.Timestamp("1995-01-01"), pd.Timestamp("2015-12-31")),
        "2015–now":  (pd.Timestamp("2015-01-01"), None),
    }
    sub_targets = ["MKT", "HML", "MOM", "RMW", "CMA"]
    per_subsample = {label: {} for label in sub_windows}
    for label, (s, e) in sub_windows.items():
        sl, _, _ = run_for_sample(monthly, daily, sub_targets, s, e, f"sub_{label.replace('–', '_')}")
        for st in sl:
            per_subsample[label][st.name] = st.alpha_ann
    plot_subsample_alpha(per_subsample, FIG_DIR / "fig_subsample_alpha.png")

    # Subsample table for the deck / paper appendix
    sub_rows = []
    for label, (s, e) in sub_windows.items():
        for fac in sub_targets:
            v = per_subsample[label].get(fac, np.nan)
            sub_rows.append({"Subsample": label, "Factor": fac, "Alpha (%/yr)": v})
    pd.DataFrame(sub_rows).to_csv(OUT_DIR / "subsample_alpha.csv", index=False)

    # ----------------------- New: predictability regression -----------------
    # Direct test of the asymmetry — does last month's variance predict the
    # next month's return? If b is small/insignificant, conditional Sharpe
    # falls in high-vol regimes and vol-management is the right MV response.
    pred_rows = []
    for name in factor_names:
        f_m = monthly[name]
        f_d = daily[name]
        rv = realised_variance_monthly(f_d)
        b, t, r2, n = predictability_regression(f_m, rv)
        pred_rows.append({"Factor": name, "b (per-yr)": b * 12 * 100,
                          "t": t, "R^2": r2, "n": n})
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "predictability.csv", index=False)
    plot_predictability(pred_rows, FIG_DIR / "fig_predictability.png")

    # ----------------------- New: spanning regressions ----------------------
    # Regress each vol-managed factor on the FULL set of original equity
    # factors (FF5 + MOM) jointly. If alpha survives, the result is not
    # explained by exposure to a different factor.
    span_rows = []
    bivariate_lookup = {s.name: s.alpha_ann for s in stats_full}
    factors_dict = {n: monthly[n] for n in factor_names if n in monthly}
    for name, _, vm in records_full:
        a_span, t_span, r2, n = spanning_regression(vm, factors_dict, factor_names)
        span_rows.append({
            "Factor": name,
            "alpha_bi": bivariate_lookup.get(name, np.nan),
            "alpha_span": a_span,
            "t_span": t_span,
            "R^2": r2,
            "n": n,
        })
    pd.DataFrame(span_rows).to_csv(OUT_DIR / "spanning.csv", index=False)
    plot_spanning(span_rows, FIG_DIR / "fig_spanning.png")

    # ----------------------- New: multifactor combination -------------------
    # In-sample tangency Sharpe ratio for the basket of original factors,
    # then the basket of vol-managed factors. Apples-to-apples comparison.
    combo_orig = pd.concat({n: monthly[n] for n in factor_names}, axis=1).dropna(how="all")
    vm_lookup = {name: vm for name, _, vm in records_full}
    combo_vm = pd.concat(vm_lookup, axis=1).dropna(how="all")
    sr_combo_orig = mv_tangency_sharpe(combo_orig)
    sr_combo_vm = mv_tangency_sharpe(combo_vm)
    sr_market_only = stats_full[0].sharpe_orig if stats_full else np.nan
    sr_market_vm = stats_full[0].sharpe_vm if stats_full else np.nan

    combo_items = [
        ("Market only\n(buy & hold)", sr_market_only),
        ("Market only\n(vol-managed)", sr_market_vm),
        ("Multifactor MV\n(originals only)", sr_combo_orig),
        ("Multifactor MV\n(vol-managed)", sr_combo_vm),
    ]
    plot_combo_sharpe(combo_items, FIG_DIR / "fig_combo_sharpe.png")
    pd.DataFrame([
        {"Portfolio": label, "Annualised Sharpe": v} for label, v in combo_items
    ]).to_csv(OUT_DIR / "combination.csv", index=False)

    # ----------------------- Time series + summary --------------------------
    full_ts = pd.concat([
        pd.DataFrame({f"{n}_orig": o, f"{n}_vm": v}) for n, o, v in records_full
    ], axis=1)
    full_ts.to_csv(OUT_DIR / "timeseries.csv")

    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("Moreira & Muir (2017) — Volatility-Managed Portfolios — Replication\n")
        f.write("=" * 78 + "\n\n")
        f.write(">> PAPER SAMPLE (≤ 2015-12)\n")
        f.write(table_paper.round(3).to_string())
        f.write("\n\n>> FULL SAMPLE (extended through latest available)\n")
        f.write(table_full.round(3).to_string())
        f.write("\n\nNotes:\n")
        f.write("- All returns are EXCESS over the risk-free rate (FF convention).\n")
        f.write("- t-statistics use Newey-West HAC standard errors (6 lags).\n")
        f.write("- Sharpe ratios and appraisal ratios are annualised (sqrt(12) scaling).\n")
        f.write("- The scaling constant c is chosen so var(vm) = var(orig) over the sample.\n")
        f.write("- The 'paper sample' restriction lets you reproduce MM's Table II directly.\n")
        f.write("- The 'extended sample' shows post-publication factor decay (HML, CMA).\n")
        f.write("- ROE, BAB, Carry require external data — see extra_data/ note in code.\n")

    print(f"\nDone. See {OUT_DIR}/ and {FIG_DIR}/")


if __name__ == "__main__":
    main()
