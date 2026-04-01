import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import csv
import numpy as np
import pandas as pd
import pdfplumber # type: ignore
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure


MONTH_FMT = "%Y-%m"


@dataclass(frozen=True)
class CleanResult:
    df: pd.DataFrame
    used_columns: Dict[str, str]


def _normalize_columns(cols: List[str]) -> List[str]:
    # Normalize column names for easier matching (keep readable labels).
    out = []
    for c in cols:
        s = str(c).strip()
        s = re.sub(r"\s+", " ", s)
        out.append(s)
    return out


def _find_first_matching_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    lowered = {c: str(c).lower() for c in columns}
    for c in columns:
        lc = lowered[c]
        for p in patterns:
            if p in lc:
                return c
    return None


def _to_numeric_series(s: pd.Series) -> pd.Series:
    # Strip currency symbols/commas; coerce errors to NaN.
    # Works for strings like "₹1,234.56" or "(123.45)".
    if s.dtype.kind in {"i", "u", "f"}:
        return pd.to_numeric(s, errors="coerce")

    cleaned = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    # Handle parentheses negatives: (123.45) => -123.45
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_dates(series: pd.Series) -> pd.Series:
    # Try explicit parsing for common statement formats to avoid slow/ambiguous inference.
    s = series.astype(str).str.strip()
    iso_ratio = s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False).mean()
    if iso_ratio >= 0.8:
        return pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")

    # dayfirst=True helps for common statement formats (DD/MM/YYYY).
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def extract_transactions_from_pdf(uploaded_file) -> pd.DataFrame:
    """
    Best-effort PDF extraction: try tables first. Bank statements are often table-like,
    so this focuses on extract_tables().
    """
    uploaded_file.seek(0)
    with pdfplumber.open(uploaded_file) as pdf:
        dfs = []
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                if not table or len(table) < 2:
                    continue

                header = table[0]
                rows = table[1:]
                if any(h is not None and str(h).strip() for h in header):
                    # Keep non-empty header names; coerce to strings
                    header_clean = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(header)]
                    df = pd.DataFrame(rows, columns=header_clean)
                else:
                    df = pd.DataFrame(rows)

                dfs.append(df)

        if not dfs:
            raise ValueError("Could not extract any tables from the PDF. Try CSV/XLSX export if possible.")

        return pd.concat(dfs, ignore_index=True)


def load_statement(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # Try common encodings; Streamlit provides a file-like object.
        raw = uploaded_file.read()
        for enc in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, skipinitialspace=True)
                return _fix_quoted_single_column_csv(df)
            except UnicodeDecodeError:
                continue
        # Last attempt
        df = pd.read_csv(io.BytesIO(raw), encoding="latin1", skipinitialspace=True)
        return _fix_quoted_single_column_csv(df)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".pdf"):
        return extract_transactions_from_pdf(uploaded_file)
    raise ValueError("Unsupported file type. Upload CSV, Excel, or PDF.")


def _fix_quoted_single_column_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some exported "CSVs" wrap each entire row in quotes, e.g.
      "Date,Description,Amount"
      "2025-01-01,Swiggy Order,-320"
    In that case pandas reads it as a single column whose header contains commas.
    Detect and split into real columns.
    """
    if df is None or df.empty:
        return df
    if df.shape[1] != 1:
        return df

    only_col = str(df.columns[0])
    if "," not in only_col:
        return df

    header_parts = [h.strip().strip('"').strip("'") for h in only_col.split(",")]
    if len(header_parts) < 2:
        return df

    s = df.iloc[:, 0].astype(str).str.strip()
    # Heuristic: most rows contain commas, indicating embedded CSV.
    if (s.str.contains(",", na=False).mean()) < 0.6:
        return df

    # Strip a single pair of wrapping quotes if present.
    s = s.str.replace(r'^"(.*)"$', r"\1", regex=True).str.replace(r"^'(.*)'$", r"\1", regex=True)
    split = s.str.split(",", n=len(header_parts) - 1, expand=True)
    # If split failed (still 1 col), bail out.
    if split.shape[1] < len(header_parts):
        return df

    split = split.iloc[:, : len(header_parts)]
    split.columns = header_parts
    return split


def clean_transactions(df_raw: pd.DataFrame) -> CleanResult:
    df = df_raw.copy()
    df.columns = _normalize_columns([str(c) for c in df.columns])

    # Identify date column
    date_col = _find_first_matching_column(list(df.columns), ["date", "txn date", "transaction date"])
    if not date_col:
        # Heuristic: first column that can be parsed as dates.
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.6:
                date_col = c
                break
    if not date_col:
        raise ValueError("Could not find a date column. Ensure your statement has a visible Date field.")

    df["__date"] = _parse_dates(df[date_col])

    # Identify amount representation: debit/credit columns or a generic amount column.
    debit_col = _find_first_matching_column(list(df.columns), ["debit", "withdrawal", "outflow", "paid"])
    credit_col = _find_first_matching_column(list(df.columns), ["credit", "deposit", "inflow"])
    amount_col = _find_first_matching_column(list(df.columns), ["amount", "amt", "value", "transaction amount"])

    if debit_col or credit_col:
        if not debit_col and credit_col:
            # Only credit exists; treat credit as "amount" (best-effort magnitude).
            df["__spend_amount"] = _to_numeric_series(df[credit_col]).abs()
        else:
            df["__spend_amount"] = np.nan
            if debit_col:
                df.loc[df[debit_col].notna(), "__spend_amount"] = _to_numeric_series(df.loc[df[debit_col].notna(), debit_col]).abs()
            # Ignore credit rows for spending purposes when both exist.
    else:
        if not amount_col:
            raise ValueError("Could not find an amount column. Ensure your statement includes an Amount field.")
        numeric_amount = _to_numeric_series(df[amount_col])
        # If statement includes negative numbers for spends, use magnitude of negatives.
        if (numeric_amount < 0).any():
            df["__spend_amount"] = numeric_amount.where(numeric_amount < 0, np.nan).abs()
            # If that produces too few rows, fall back to all non-null magnitude.
            if df["__spend_amount"].notna().mean() < 0.2:
                df["__spend_amount"] = numeric_amount.abs()
        else:
            df["__spend_amount"] = numeric_amount.abs()

    # Identify description/merchant column for categorization & top expenses.
    desc_col = _find_first_matching_column(
        list(df.columns),
        [
            "merchant",
            "description",
            "narration",
            "particulars",
            "details",
            "transaction description",
            "ref",
            "name",
        ],
    )
    if not desc_col:
        # fallback: first non-date, non-amount-ish column
        candidate_cols = [c for c in df.columns if c not in {date_col, debit_col, credit_col, amount_col}]
        desc_col = candidate_cols[0] if candidate_cols else date_col

    df["__description"] = df[desc_col].astype(str).fillna("")
    df["__category"] = np.nan

    # Drop missing critical fields (missing values cleaning).
    df["__amount_ok"] = df["__spend_amount"].notna()
    df["__date_ok"] = df["__date"].notna()
    df = df[df["__date_ok"] & df["__amount_ok"]]
    df["__spend_amount"] = pd.to_numeric(df["__spend_amount"], errors="coerce")
    df = df[df["__spend_amount"].notna() & (df["__spend_amount"] > 0)]

    # Final cleanup
    df["__month"] = df["__date"].dt.to_period("M").astype(str)

    used = {
        "date_col": str(date_col),
        "amount_source": str(debit_col or credit_col or amount_col),
        "description_col": str(desc_col),
    }
    return CleanResult(df=df.reset_index(drop=True), used_columns=used)


def auto_categorize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple keyword mapping only (as requested).
    """
    keyword_map = {
        "swiggy": "Food",
        "zomato": "Food",
        "restaurant": "Food",
        "cafe": "Food",
        "uber": "Transport",
        "ola": "Transport",
        "fuel": "Transport",
        "petrol": "Transport",
        "amazon": "Shopping",
        "myntra": "Shopping",
        "flipkart": "Shopping",
        "electricity": "Bills",
        "bill": "Bills",
        "recharge": "Bills",
        "internet": "Bills",
        "wifi": "Bills",
        "flight": "Travel",
        "hotel": "Travel",
        "airbnb": "Travel",
        "movie": "Entertainment",
        "netflix": "Entertainment",
        "spotify": "Entertainment",
        "prime": "Entertainment",
        "gym": "Health",
        "pharmacy": "Health",
        "hospital": "Health",
    }

    descriptions = df["__description"].str.lower().fillna("")
    category = pd.Series(["Other" for _ in range(len(df))], index=df.index, dtype=object)
    for kw, cat in keyword_map.items():
        mask = descriptions.str.contains(kw, na=False)
        category = category.where(~mask, cat)

    # If the statement already contains a category column, we can respect it later,
    # but for this scope we keep the keyword-based logic.
    df["category"] = category
    return df


def monthly_spending(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("__month", as_index=False)["__spend_amount"].sum()
    out = out.sort_values("__month")
    out.rename(columns={"__month": "month", "__spend_amount": "spend"}, inplace=True)
    return out


def zscore(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    mean = vals.mean()
    std = vals.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0] * len(vals), index=values.index)
    return (vals - mean) / std


def detect_anomalies(df: pd.DataFrame, target_month: str, min_ratio: float = 1.5) -> List[str]:
    """
    Simple but sensitive anomaly detection:
    - Transaction outliers in selected month using z-score threshold 1.5.
    - Category monthly spikes: current month vs previous-month average.
    """
    messages: List[str] = []

    # 1) Single-transaction outliers in selected month.
    month_df = df[df["__month"] == target_month].copy()
    if not month_df.empty and month_df["__spend_amount"].std(ddof=0) > 0:
        month_df["__z_txn"] = zscore(month_df["__spend_amount"])
        outliers = month_df[month_df["__z_txn"].abs() > 1.5].sort_values("__spend_amount", ascending=False).head(2)
        for _, row in outliers.iterrows():
            messages.append(
                f"Spike detected: {row['__description']} (INR {row['__spend_amount']:.0f}) exceeds normal pattern"
            )

    # 2) Category monthly spikes.
    categories = df["category"].unique().tolist()
    for cat in categories:
        cat_df = df[df["category"] == cat].copy()
        if cat_df.empty:
            continue

        monthly = cat_df.groupby("__month")["__spend_amount"].sum().sort_index()
        if target_month not in monthly.index:
            continue
        current = float(monthly.loc[target_month])
        prev = monthly.drop(index=target_month)
        prev_mean = float(prev.mean()) if len(prev) > 0 else 0.0
        if prev_mean <= 0:
            continue

        ratio = current / prev_mean
        zs = float(zscore(monthly).loc[target_month]) if len(monthly) >= 2 else 0.0

        if ratio >= min_ratio and zs >= 1.0:
            messages.append(f"{cat} expenses increased by {ratio:.1f}x this month")

    # Keep output short and recruiter-friendly
    messages.sort(key=lambda s: float(re.search(r"([0-9]+(\.[0-9]+)?)x", s).group(1)) if re.search(r"([0-9]+(\.[0-9]+)?)x", s) else 0, reverse=True)
    return messages[:4]


def detect_subscriptions(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Simple repeated-transaction detector:
    - Group by merchant/description.
    - Consider it a subscription if it appears in >= 3 months and the amount is stable.
    """
    if df.empty:
        return pd.DataFrame(), 0.0

    # Some statements have long descriptions; limit noise.
    work = df[["__description", "__month", "__spend_amount", "__date"]].copy()
    work["__merchant_key"] = work["__description"].astype(str).str.strip().str.lower().str.slice(0, 80)

    # Compute per-merchant monthly totals first (vectorized).
    monthly = (
        work.groupby(["__merchant_key", "__month"], as_index=False)
        .agg(
            monthly_cost=("__spend_amount", "sum"),
            transactions=("__spend_amount", "size"),
            last_transaction=("__date", "max"),
        )
    )

    # Then compute stability stats across months.
    stats = (
        monthly.groupby("__merchant_key", as_index=False)
        .agg(
            months_seen=("__month", "nunique"),
            monthly_cost_mean=("monthly_cost", "mean"),
            monthly_cost_std=("monthly_cost", "std"),
            transactions=("transactions", "sum"),
            last_transaction=("last_transaction", "max"),
        )
    )

    # Stability heuristic: std within 10% of mean.
    stats["monthly_cost_std"] = stats["monthly_cost_std"].fillna(0.0)
    stats = stats[(stats["months_seen"] >= 3) & (stats["monthly_cost_mean"] > 0)]
    stats = stats[(stats["monthly_cost_std"] / stats["monthly_cost_mean"]) <= 0.10]

    if stats.empty:
        return stats, 0.0

    stats["monthly_cost"] = stats["monthly_cost_mean"].round(2)
    stats["last_transaction"] = pd.to_datetime(stats["last_transaction"]).dt.date.astype(str)

    subs = stats[["__merchant_key", "monthly_cost", "months_seen", "transactions", "last_transaction"]].rename(
        columns={"__merchant_key": "merchant"}
    )

    subs = subs.sort_values("monthly_cost", ascending=False)
    yearly_total = float(subs["monthly_cost"].sum() * 12)
    return subs.reset_index(drop=True), yearly_total


def generate_insights(
    df: pd.DataFrame,
    budgets: Dict[str, float],
    selected_month: str,
    yearly_subscriptions: float,
) -> List[str]:
    insights: List[str] = []
    if df.empty:
        return insights

    # Highest expense category (overall)
    totals = df.groupby("category")["__spend_amount"].sum().sort_values(ascending=False)
    if len(totals) > 0:
        top_cat = totals.index[0]
        insights.append(f"{top_cat} is your highest expense")

    # Spending peak month (adds a concrete timeline signal).
    month_totals_series = df.groupby("__month")["__spend_amount"].sum().sort_index()
    if len(month_totals_series) > 1:
        peak_month = str(month_totals_series.idxmax())
        try:
            peak_month_label = pd.to_datetime(f"{peak_month}-01").strftime("%B")
        except Exception:
            peak_month_label = peak_month
        insights.append(f"Spending peaked in {peak_month_label}")

    # Weekday vs weekend behavior
    weekend_mask = df["__date"].dt.dayofweek.isin([5, 6])  # 5=Sat, 6=Sun
    weekend_spend = float(df.loc[weekend_mask, "__spend_amount"].sum())
    weekday_spend = float(df.loc[~weekend_mask, "__spend_amount"].sum())
    if weekday_spend > 0:
        ratio = weekend_spend / weekday_spend
        if ratio >= 1.1 and weekend_spend > 0:
            insights.append(f"Weekend spending is higher ({ratio:.2f}x your weekday spend)")

    # Category trend for selected month (specific and concrete)
    month_totals_all = df.groupby(["__month", "category"], as_index=False)["__spend_amount"].sum()
    prev_months = sorted(df["__month"].unique().tolist())
    prev_month = prev_months[-2] if len(prev_months) >= 2 else None
    if prev_month and selected_month:
        curr = month_totals_all[month_totals_all["__month"] == selected_month].set_index("category")["__spend_amount"]
        prev = month_totals_all[month_totals_all["__month"] == prev_month].set_index("category")["__spend_amount"]
        for cat in sorted(set(curr.index).intersection(set(prev.index))):
            if prev[cat] > 0:
                ratio = float(curr[cat] / prev[cat])
                if ratio >= 1.5:
                    insights.append(f"{cat} increased by {ratio:.1f}x in {selected_month}")
                    break

    # Budget exceedances (selected month only)
    month_df = df[df["__month"] == selected_month]
    month_totals = month_df.groupby("category")["__spend_amount"].sum()
    budget_label = {
        "Food": "dining",
        "Transport": "transport",
        "Shopping": "shopping",
        "Other": "category",
    }
    for cat, limit in budgets.items():
        if limit and limit > 0:
            used = float(month_totals.get(cat, 0.0))
            if used > limit:
                insights.append(f"You exceeded your {budget_label.get(cat, cat.lower())} budget in {selected_month}")

    if yearly_subscriptions > 0:
        insights.append(f"Subscriptions cost approximately {yearly_subscriptions:.0f} per year")

    # If nothing else triggered, add a simple suggestion.
    if len(insights) < 2:
        insights.append("Review your top expenses and adjust budgets for next month")

    # Keep it simple and short
    return insights[:4]


def pie_chart_by_category(df: pd.DataFrame) -> Figure:
    totals = df.groupby("category", as_index=False)["__spend_amount"].sum().sort_values("__spend_amount", ascending=False)
    if totals.empty:
        return px.pie(values=[1], names=["No data"], title="Spending by Category").update_traces(labels=["No data"])
    fig = px.pie(totals, names="category", values="__spend_amount", hole=0.45, title="Spending by Category")
    fig.update_layout(showlegend=True, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def line_chart_monthly_trend(df: pd.DataFrame) -> Figure:
    trend = monthly_spending(df)
    if trend.empty:
        return px.line(title="Monthly Spending Trend").update_traces(x=[], y=[])
    fig = px.line(trend, x="month", y="spend", markers=True, title="Monthly Spending Trend")
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def bar_top_expenses(df: pd.DataFrame, top_n: int = 10) -> Figure:
    by_merchant = (
        df.groupby("__description", as_index=False)["__spend_amount"].sum().sort_values("__spend_amount", ascending=False).head(top_n)
    )
    if by_merchant.empty:
        return px.bar(title="Top Expenses").update_traces(x=[], y=[])
    by_merchant.rename(columns={"__description": "expense", "__spend_amount": "spend"}, inplace=True)
    by_merchant["expense_short"] = by_merchant["expense"].astype(str).str.slice(0, 18)
    by_merchant.loc[by_merchant["expense"].str.len() > 18, "expense_short"] = (
        by_merchant.loc[by_merchant["expense"].str.len() > 18, "expense_short"] + "..."
    )
    fig = px.bar(by_merchant, x="expense_short", y="spend", title="Top Expenses", text_auto=".2s", hover_data={"expense": True})
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(tickangle=-35, title_text="")
    return fig


def main():
    st.set_page_config(page_title="Finance Dashboard", layout="wide")

    st.title("Finance Dashboard")
    st.caption("Upload a bank statement to get spending insights, budgets, anomalies, and subscriptions.")

    with st.sidebar:
        st.header("1) Upload Statement")
        uploaded = st.file_uploader(
            "Upload bank statement (PDF / Excel / CSV)",
            type=["pdf", "csv", "xlsx", "xls"],
            accept_multiple_files=False,
        )

        st.divider()
        st.header("2) Options")
        show_raw_preview = st.checkbox("Show cleaned transactions preview", value=False)
        show_debug = st.checkbox("Show debug info (columns & parsing)", value=False)

    if not uploaded:
        st.info("Upload a statement to begin.")
        return

    try:
        raw_df = load_statement(uploaded)
        if raw_df is None or raw_df.empty:
            st.error("The uploaded file did not contain readable data.")
            return

        if show_debug:
            with st.expander("Debug: raw file preview", expanded=True):
                st.write("**Columns**", list(raw_df.columns))
                st.dataframe(raw_df.head(20), use_container_width=True)

        clean_res = clean_transactions(raw_df)
        df = clean_res.df
        df = auto_categorize(df)

        if df.empty:
            st.error("No usable transactions found after cleaning.")
            if show_debug:
                with st.expander("Debug: cleaning decisions", expanded=True):
                    st.write("**Detected columns**", clean_res.used_columns)
                    tmp = raw_df.copy()
                    tmp.columns = _normalize_columns([str(c) for c in tmp.columns])
                    # Re-run minimal parsing on raw to show success rates.
                    date_col = clean_res.used_columns.get("date_col")
                    amount_col = clean_res.used_columns.get("amount_source")
                    if date_col in tmp.columns:
                        parsed_date = _parse_dates(tmp[date_col])
                        st.write("**Date parse success %**", float(parsed_date.notna().mean() * 100))
                    if amount_col in tmp.columns:
                        parsed_amt = _to_numeric_series(tmp[amount_col])
                        st.write("**Amount parse success %**", float(parsed_amt.notna().mean() * 100))
            st.caption("Turn on 'Show debug info' in the sidebar to see exactly what failed.")
            st.stop()

        if show_raw_preview:
            st.subheader("Cleaned Transactions Preview")
            st.dataframe(
                df[["__date", "__description", "__spend_amount", "category", "__month"]].head(50),
                use_container_width=True,
            )

        # Choose month for budget & comparisons.
        months_sorted = sorted(df["__month"].unique().tolist())
        latest_month = months_sorted[-1] if months_sorted else None
        selected_month = st.sidebar.selectbox("Select month for budget & charts", options=months_sorted, index=len(months_sorted) - 1)

        # Budget: user sets limits per category.
        st.sidebar.divider()
        st.sidebar.header("3) Monthly Budgets (Limits)")
        # Because categorization is keyword-based, the category universe is small.
        # We still expose budgets for each category present to match the scope.
        top_categories = sorted(df["category"].dropna().unique().tolist()) if not df.empty else []
        budgets: Dict[str, float] = {}
        for cat in top_categories:
            budgets[cat] = float(
                st.sidebar.number_input(
                    f"{cat} limit",
                    min_value=0.0,
                    value=0.0,
                    step=50.0,
                    format="%.2f",
                )
            )

    except Exception as e:
        st.exception(e)
        st.stop()

    # Main dashboard
    month_df = df[df["__month"] == selected_month].copy()

    st.subheader("Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pie_chart_by_category(month_df), use_container_width=True)
    with col2:
        st.plotly_chart(line_chart_monthly_trend(df), use_container_width=True)

    st.plotly_chart(bar_top_expenses(month_df), use_container_width=True)

    # Budget tracking
    st.subheader("Budget Tracking")
    if budgets:
        month_totals = month_df.groupby("category")["__spend_amount"].sum().to_dict()
        # Render a progress bar per category.
        progress_cols = st.columns(2 if len(budgets) > 1 else 1)
        for i, (cat, limit) in enumerate(budgets.items()):
            if limit <= 0:
                continue
            used = float(month_totals.get(cat, 0.0))
            pct = min(1.0, used / limit) if limit > 0 else 0.0

            container = progress_cols[i % len(progress_cols)]
            with container:
                st.markdown(f"**{cat}**: {used:.2f} / {limit:.2f}")
                st.progress(int(pct * 100))
                if used >= limit:
                    st.error(f"Exceeded {cat} limit!")
                elif used >= 0.9 * limit:
                    st.warning(f"Near limit for {cat} (>= 90%).")
    else:
        st.caption("Set budgets in the sidebar to enable progress tracking and near-limit alerts.")

    # Anomaly detection
    st.subheader("Anomaly Detection")
    anomalies = detect_anomalies(df, target_month=selected_month)
    if anomalies:
        for msg in anomalies:
            st.info(msg)
    else:
        st.success("No unusually high spikes detected for this month.")

    # Subscription detector
    st.subheader("Subscription Detector")
    with st.spinner("Detecting subscriptions..."):
        subs, yearly_total = detect_subscriptions(df)
    if subs.empty:
        st.caption("No clear recurring subscriptions detected (needs repeated similar transactions).")
    else:
        st.markdown(f"**Estimated yearly subscription cost:** `{yearly_total:.2f}`")
        st.dataframe(subs[["merchant", "monthly_cost", "months_seen", "transactions", "last_transaction"]].head(15), use_container_width=True)

    # Insights
    st.subheader("Insights & Suggestions")
    insights = generate_insights(df, budgets=budgets, selected_month=selected_month, yearly_subscriptions=yearly_total)
    for line in insights:
        st.write(f"- {line}")

    st.caption(
        "Note: Categorization uses transparent keyword rules (Food/Transport/Shopping/Bills/Travel/Entertainment/Health + Other)."
    )


if __name__ == "__main__":
    main()

