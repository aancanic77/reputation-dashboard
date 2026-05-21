import requests
import pandas as pd
from typing import List, Dict
from utils import load_reddit_dataframe, ensure_mock_data

REDDIT_BASE_URL = "https://www.reddit.com"
HEADERS = {"User-Agent": "ReputationDashboard/0.1 by student"}

# ============================================================
# DEMO MODE automat dacă pică netul
# ============================================================
def internet_is_available() -> bool:
    try:
        requests.get("https://www.google.com", timeout=2)
        return True
    except:
        return False

DEMO_MODE = not internet_is_available()

# ============================================================
# DEMO MOCK — EXACT 60 comentarii
# ============================================================
def generate_demo_mock(n=60):
    companies = ["Apple", "Google", "Samsung"]
    rows = []
    for i in range(n):
        comp = companies[i % 3]
        rows.append({
            "comment_id": f"demo_{i}",
            "text": f"Demo comment {i} about {comp}",
            "subreddit": comp.lower(),
            "author": f"user_{i}",
            "created_utc": 1710000000 + i,
            "company": comp,
        })
    return pd.DataFrame(rows)

# ============================================================
# LIMITARE — DOAR ÎN DEMO MODE (60 rânduri)
# ============================================================
MAX_ROWS_DEMO = 60

def limit_df(df: pd.DataFrame, max_rows: int = None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    
    if DEMO_MODE:
        limit = MAX_ROWS_DEMO
    else:
        limit = max_rows if max_rows is not None else None
    
    if limit and len(df) > limit:
        return df.sample(n=limit, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)

# ============================================================
# FETCH REDDIT COMMENTS — EXACT CA ÎNAINTE
# ============================================================
def fetch_reddit_comments(subreddits: List[str], limit: int = 50, max_rows: int = None) -> pd.DataFrame:

    if DEMO_MODE:
        return generate_demo_mock(MAX_ROWS_DEMO)

    rows: List[Dict] = []

    for sub in subreddits:
        url = f"{REDDIT_BASE_URL}/r/{sub}/comments.json?limit={limit}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                raise Exception("Reddit returned non-200")

            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                d = child.get("data", {})
                rows.append(
                    {
                        "comment_id": d.get("id"),
                        "text": d.get("body") or d.get("selftext") or "",
                        "subreddit": d.get("subreddit", sub),
                        "author": d.get("author"),
                        "created_utc": d.get("created_utc"),
                    }
                )

        except Exception:
            return fallback_data()

    if not rows:
        return fallback_data(max_rows)

    return limit_df(pd.DataFrame(rows), max_rows)

# ============================================================
# FALLBACK DATA — EXACT CA ÎNAINTE
# ============================================================
def fallback_data(max_rows: int = None) -> pd.DataFrame:

    if DEMO_MODE:
        return generate_demo_mock(MAX_ROWS_DEMO)

    try:
        df = load_reddit_dataframe()
        if df is not None and len(df) > 0:
            df = df.rename(columns={"content": "text"})
            df = df[["comment_id", "text", "subreddit", "author", "created_utc"]]
            return limit_df(df, max_rows)
    except:
        pass

    try:
        csv_path = ensure_mock_data()
        csv_limit = max_rows if max_rows else 1000
        df = pd.read_csv(csv_path, nrows=csv_limit)
        return limit_df(df, max_rows)
    except:
        pass

    return generate_demo_mock(MAX_ROWS_DEMO)

# ============================================================
# EXTRACT FIELDS
# ============================================================
def extract_fields(df: pd.DataFrame, max_rows: int = None) -> pd.DataFrame:
    cols = ["comment_id", "text", "subreddit", "author", "created_utc"]
    existing = [c for c in cols if c in df.columns]
    return limit_df(df[existing].copy(), max_rows)

# ============================================================
# MAP TO COMPANY
# ============================================================
def map_to_company(df: pd.DataFrame, max_rows: int = None) -> pd.DataFrame:
    def _map(row):
        sub = str(row.get("subreddit", "")).lower()
        text = str(row.get("text", "")).lower()

        if "apple" in sub or "iphone" in text or "mac" in text:
            return "Apple"
        if "google" in sub or "android" in text or "pixel" in text:
            return "Google"
        if "samsung" in sub or "galaxy" in text:
            return "Samsung"
        return "Other"

    df = df.copy()
    df["company"] = df.apply(_map, axis=1)
    df = df[df["company"].isin(["Apple", "Google", "Samsung"])]
    return limit_df(df, max_rows)

# ============================================================
# PIPELINE COMPLET — EXACT CA ÎNAINTE
# ============================================================
def run_reddit_pipeline_live(limit_per_sub: int = 20) -> pd.DataFrame:

    subreddits = ["apple", "google", "samsung"]
    
    max_rows_limit = MAX_ROWS_DEMO if DEMO_MODE else (limit_per_sub * 3)

    try:
        raw = fetch_reddit_comments(subreddits, limit_per_sub, max_rows_limit)
    except Exception:
        raw = fallback_data(max_rows_limit)

    extracted = extract_fields(raw, max_rows_limit)
    mapped = map_to_company(extracted, max_rows_limit)

    if DEMO_MODE:
        mapped["dl_label"] = "disabled"

    return limit_df(mapped, max_rows_limit)
