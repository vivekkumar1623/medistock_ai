"""Data profiling: column-level facts a reviewer can check against the raw files.

Outputs a JSON profile (machine-readable) and a Markdown summary (human-readable)
so the report can quote real numbers rather than impressions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _outlier_count(series: pd.Series) -> int:
    """IQR rule: outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0
    q1, q3 = numeric.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return int((numeric != q1).sum())
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((numeric < lower) | (numeric > upper)).sum())


def profile_column(series: pd.Series) -> dict[str, Any]:
    nulls = int(series.isna().sum())
    facts: dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": nulls,
        "null_fraction": round(nulls / len(series), 6) if len(series) else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        facts.update(
            {
                "min": float(numeric.min()) if numeric.notna().any() else None,
                "max": float(numeric.max()) if numeric.notna().any() else None,
                "mean": round(float(numeric.mean()), 6) if numeric.notna().any() else None,
                "std": round(float(numeric.std()), 6) if numeric.notna().any() else None,
                "p50": float(numeric.median()) if numeric.notna().any() else None,
                "p95": float(numeric.quantile(0.95)) if numeric.notna().any() else None,
                "zero_count": int((numeric == 0).sum()),
                "outlier_count_iqr": _outlier_count(series),
            }
        )
    elif pd.api.types.is_datetime64_any_dtype(series):
        facts.update(
            {
                "min": series.min().isoformat() if series.notna().any() else None,
                "max": series.max().isoformat() if series.notna().any() else None,
            }
        )
    else:
        top = series.astype("string").value_counts(dropna=True).head(5)
        facts["top_values"] = {str(k): int(v) for k, v in top.items()}
    return facts


def profile_frame(frame: pd.DataFrame, table: str) -> dict[str, Any]:
    return {
        "table": table,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "column_profile": {column: profile_column(frame[column]) for column in frame.columns},
    }


def profile_dataset(
    products: pd.DataFrame,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    batches: pd.DataFrame,
    panel: pd.DataFrame,
    dataset_meta: dict[str, Any],
) -> dict[str, Any]:
    panel_zero_share = float((panel["quantity"] == 0).mean()) if len(panel) else 0.0
    per_product_zero = (
        panel.assign(zero=(panel["quantity"] == 0).astype(int)).groupby("product_id")["zero"].mean()
    )
    return {
        "dataset": dataset_meta,
        "tables": [
            profile_frame(products, "products"),
            profile_frame(sales, "sales"),
            profile_frame(inventory, "inventory"),
            profile_frame(batches, "batches"),
            profile_frame(panel, "demand_panel"),
        ],
        "demand_characteristics": {
            "panel_zero_share": round(panel_zero_share, 6),
            "products_with_gt_50pct_zero_days": int((per_product_zero > 0.5).sum()),
            "products_with_zero_sales_all_period": int(
                panel.groupby("product_id")["quantity"].sum().eq(0).sum()
            ),
            "mean_daily_units": round(float(panel["quantity"].mean()), 4),
        },
        "category_summary": (
            panel.merge(products[["product_id", "category"]], on="product_id")
            .groupby("category")["quantity"]
            .agg(["sum", "mean", "std"])
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
    }


def render_markdown(profile: dict[str, Any]) -> str:
    lines: list[str] = ["# Data Profile", ""]
    dataset = profile.get("dataset", {})
    lines += [
        f"* Dataset version: `{dataset.get('dataset_version', 'n/a')}`",
        f"* Synthetic: **{dataset.get('is_synthetic')}**",
        f"* Seed: `{dataset.get('seed')}`",
        f"* Date range: {dataset.get('date_range', {}).get('start')} "
        f"to {dataset.get('date_range', {}).get('end')}",
        f"* Snapshot date: {dataset.get('snapshot_date')}",
        "",
        "## Tables",
        "",
        "| Table | Rows | Columns | Duplicate rows |",
        "|---|---:|---:|---:|",
    ]
    for table in profile["tables"]:
        lines.append(
            f"| {table['table']} | {table['rows']} | {table['columns']} "
            f"| {table['duplicate_rows']} |"
        )

    demand = profile["demand_characteristics"]
    lines += [
        "",
        "## Demand characteristics",
        "",
        f"* Zero-demand share across the whole panel: **{demand['panel_zero_share']:.2%}**",
        "* Products with more than half of days at zero: "
        f"**{demand['products_with_gt_50pct_zero_days']}**",
        f"* Mean daily units per product-day: **{demand['mean_daily_units']:.2f}**",
        "",
        "## Category summary",
        "",
        "| Category | Total units | Mean/day | Std |",
        "|---|---:|---:|---:|",
    ]
    for row in profile["category_summary"]:
        lines.append(
            f"| {row['category']} | {row['sum']:,.0f} | {row['mean']:.2f} | {row['std']:.2f} |"
        )

    lines += [
        "",
        "## Column-level outliers (IQR rule)",
        "",
        "| Table | Column | Outliers | Min | Max |",
        "|---|---|---:|---:|---:|",
    ]
    for table in profile["tables"]:
        for column, facts in table["column_profile"].items():
            outliers = facts.get("outlier_count_iqr")
            if outliers:
                lines.append(
                    f"| {table['table']} | {column} | {outliers} | "
                    f"{facts.get('min')} | {facts.get('max')} |"
                )
    lines.append("")
    return "\n".join(lines)
