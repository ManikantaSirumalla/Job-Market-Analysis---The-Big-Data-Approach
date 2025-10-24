import os
from pathlib import Path
import re
import pandas as pd


BRONZE_DIR = Path("data/bronze/job_postings")
SILVER_OUT = Path("data/silver/unified_postings.parquet")
SILVER_DIR = Path("data/silver/unified_postings")  # support dir-style parquet


TITLE_NOISE = {
    "religion","racial","color","gender","disability","veteran","eeo","equal",
    "vision","dental","medical","401k","hourly","yearly","weekly","benefit","benefits",
    "drug","background","check","sign","bonus","insurance","policy","policies","pto",
}

ALLOW_KEYWORDS = {
    "engineer","developer","scientist","analyst","manager","architect","lead","specialist",
    "consultant","product","program","project","qa","sdet","security","devops","cloud",
    "ml","ai","data","database","bi","business intelligence","backend","front end","frontend",
    "full stack","ios","android","mobile","platform","infra","site reliability","sre",
}


def is_real_title(title: str) -> bool:
    t = str(title).strip().lower()
    if len(t) < 3:
        return False
    if any(w in t for w in TITLE_NOISE):
        return False
    return any(k in t for k in ALLOW_KEYWORDS)


def normalize_skill_token(tok: str) -> str:
    s = str(tok).strip().lower()
    if not s:
        return ""
    # keep alnum and common tech symbols
    s = re.sub(r"[^a-z0-9+#\.\-]", "", s)
    if len(s) < 2:
        return ""
    return s


def load_bronze() -> pd.DataFrame:
    if not BRONZE_DIR.exists():
        return pd.DataFrame()
    parts = list(BRONZE_DIR.glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


def derive_salary_amount(row: pd.Series) -> float | None:
    for col in ("med_salary", "max_salary", "min_salary", "normalized_salary"):
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col])
            except Exception:
                try:
                    return float(str(row[col]).replace(",", ""))
                except Exception:
                    continue
    return None


def main() -> None:
    df = load_bronze()
    if df.empty:
        print("No bronze job_postings found; skipping.")
        return

    # Select and rename core columns if present
    col_map = {
        "job_id": "job_id",
        "title": "job_title",
        "company_name": "company",
        "location": "location",
        # posted_ts will be created by coalescing available source columns
        "pay_period": "pay_period",
        "currency": "currency",
        "min_salary": "min_salary",
        "med_salary": "med_salary",
        "max_salary": "max_salary",
        "normalized_salary": "normalized_salary",
        "skills": "skills_raw",
        "skills_desc": "skills_desc",
        "remote_allowed": "remote_allowed",
        "description": "description",
    }

    present = {k: v for k, v in col_map.items() if k in df.columns}
    sdf = df[list(present.keys())].rename(columns=present).copy()

    # Posted date from first available timestamp-like column
    ts = None
    if "listed_time" in df.columns:
        ts = pd.to_datetime(df["listed_time"], errors="coerce")
    if ts is None or ts.isna().all():
        if "original_listed_time" in df.columns:
            ts = pd.to_datetime(df["original_listed_time"], errors="coerce")
    if ts is None:
        sdf["posted_date"] = pd.NaT
    else:
        sdf["posted_date"] = ts.dt.date

    # Salary amount numeric
    sdf["salary_amount"] = sdf.apply(derive_salary_amount, axis=1)

    # Pay period/currency normalized to str
    if "pay_period" in sdf.columns:
        sdf["pay_period"] = sdf["pay_period"].astype(str).str.upper()
    if "currency" in sdf.columns:
        sdf["currency"] = sdf["currency"].astype(str).str.upper()

    # Remote flag to bool
    if "remote_allowed" in sdf.columns:
        col = sdf["remote_allowed"].astype(str).str.lower()
        sdf["remote_flag"] = col.isin(["true", "1", "yes", "y", "remote", "remotely"]) | col.str.contains("remote", na=False)
    else:
        sdf["remote_flag"] = False

    # Clean job titles
    if "job_title" in sdf.columns:
        sdf["job_title"] = sdf["job_title"].astype(str)
        sdf = sdf[sdf["job_title"].apply(is_real_title)]

    # Skills array
    skills_list = []
    for i, row in sdf.iterrows():
        raw = ""
        if "skills_raw" in sdf.columns and pd.notna(row.get("skills_raw")):
            raw = str(row.get("skills_raw"))
        elif "skills_desc" in sdf.columns and pd.notna(row.get("skills_desc")):
            raw = str(row.get("skills_desc"))
        tokens = (
            raw.lower()
            .replace(",", " ")
            .replace(";", " ")
            .replace("/", " ")
            .split()
        )
        cleaned = [normalize_skill_token(t) for t in tokens]
        cleaned = [t for t in cleaned if t]
        skills_list.append(cleaned)
    sdf["skills"] = skills_list

    # Final projection
    final_cols = [
        "job_id","job_title","company","location","posted_date","salary_amount",
        "currency","pay_period","remote_flag","skills",
    ]
    final_cols = [c for c in final_cols if c in sdf.columns]
    out = sdf[final_cols].copy()

    # Ensure silver dir exists
    SILVER_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SILVER_OUT, index=False)

    # Also write a file inside directory-style location for compatibility
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SILVER_DIR / "part-00000.parquet", index=False)

    print(f"Wrote unified postings: rows={len(out):,} -> {SILVER_OUT}")


if __name__ == "__main__":
    main()


