"""Exploratory data analysis: figures plus written interpretation.

Every chart here answers a question a pharmacy manager or a reviewer would ask.
Decorative charts are deliberately absent: if a figure does not change a decision
or a design choice, it is not in this module.

Run via ``python -m scripts.run_eda``; outputs go to ``reports/figures`` and
``reports/eda_report.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display server needed in CI or Docker

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PALETTE = ["#0E7C86", "#4C5FD5", "#7C3AED", "#0EA5E9", "#F59E0B", "#EF4444", "#059669", "#64748B"]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.dpi": 140,
        "axes.edgecolor": "#E2E8F0",
        "axes.labelcolor": "#0F172A",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "figure.facecolor": "white",
    }
)


def _save(figure: plt.Figure, path: Path, caption: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return {"file": path.name, "caption": caption}


def category_demand(panel: pd.DataFrame, products: pd.DataFrame, out: Path) -> dict[str, str]:
    merged = panel.merge(products[["product_id", "category"]], on="product_id")
    totals = merged.groupby("category")["quantity"].sum().sort_values(ascending=False) / 1000
    figure, axis = plt.subplots(figsize=(9, 4.4))
    bars = axis.bar(totals.index, totals.to_numpy(), color=PALETTE[: len(totals)])
    axis.set_ylabel("units sold (thousands)")
    axis.set_title("Total demand by category over the full history")
    axis.tick_params(axis="x", rotation=22)
    for bar, value in zip(bars, totals.to_numpy(), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}k",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return _save(
        figure,
        out / "01_category_demand.png",
        "Analgesics and vitamins dominate volume; dermatology is the smallest category. "
        "Category-level baselines in the generator reflect this ordering.",
    )


def monthly_pattern(panel: pd.DataFrame, out: Path) -> dict[str, str]:
    frame = panel.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    monthly = frame.groupby("month")["quantity"].sum()
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(monthly.index, monthly.to_numpy(), marker="o", color=PALETTE[0], linewidth=1.8)
    axis.set_title("Total monthly demand (all products)")
    axis.set_ylabel("units")
    axis.tick_params(axis="x", rotation=45)
    axis.set_xticks(range(0, len(monthly), max(1, len(monthly) // 10)))
    axis.set_xticklabels(
        [monthly.index[i] for i in range(0, len(monthly), max(1, len(monthly) // 10))]
    )
    return _save(
        figure,
        out / "02_monthly_demand.png",
        "Clear annual structure: winter peaks (antibiotics, vitamins) and a spring peak "
        "(antihistamines). This is why the model needs date-based features rather than a "
        "single long-run average.",
    )


def weekday_pattern(sales: pd.DataFrame, products: pd.DataFrame, out: Path) -> dict[str, str]:
    merged = sales.merge(products[["product_id", "category"]], on="product_id")
    table = merged.groupby(["category", "day_of_week"])["quantity_sold"].mean().unstack()
    table.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    normalised = table.div(table.mean(axis=1), axis=0)

    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    sns.heatmap(
        normalised,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=1.0,
        cbar_kws={"label": "relative to category mean"},
        ax=axis,
        linewidths=0.5,
    )
    axis.set_title("Weekday effect by category (1.00 = average day)")
    axis.set_xlabel("")
    axis.set_ylabel("")
    return _save(
        figure,
        out / "03_weekday_pattern.png",
        "Clinic-driven categories (antibiotics, cardiovascular, diabetes) dip at weekends, "
        "while OTC categories (antihistamines, vitamins, dermatology) rise. A single global "
        "weekday multiplier would blur this, so the generator applies category profiles.",
    )


def zero_demand_profile(panel: pd.DataFrame, out: Path) -> dict[str, str]:
    per_product = (
        panel.assign(zero=(panel["quantity"] == 0).astype(int)).groupby("product_id")["zero"].mean()
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(per_product.to_numpy(), bins=24, color=PALETTE[2], edgecolor="white")
    axis.set_xlabel("share of days with zero sales")
    axis.set_ylabel("products")
    axis.set_title("Distribution of zero-sales days per product")
    axis.axvline(
        per_product.mean(),
        color=PALETTE[5],
        linestyle="--",
        linewidth=1.6,
        label=f"mean {per_product.mean():.1%}",
    )
    axis.legend(frameon=False)
    return _save(
        figure,
        out / "04_zero_demand.png",
        f"Zero-demand days are common ({per_product.mean():.1%} on average) and heavily "
        "skewed: a subset of slow movers is at zero more than half the time. This is why "
        "MAPE is unusable and WAPE is the headline metric, and why sparse products carry "
        "low-confidence flags in the UI.",
    )


def inventory_vs_forecast(
    inventory: pd.DataFrame, forecast: pd.DataFrame, out: Path
) -> dict[str, str]:
    merged = inventory.merge(forecast[["product_id", "predicted_demand"]], on="product_id")
    merged = merged[merged["predicted_demand"] > 0]
    merged["coverage_days"] = merged["current_stock"] / (merged["predicted_demand"] / 30)

    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    axis.scatter(
        merged["predicted_demand"],
        merged["coverage_days"],
        s=22,
        alpha=0.65,
        color=PALETTE[1],
        edgecolor="white",
        linewidth=0.4,
    )
    axis.set_xlabel("forecast demand over 30 days (units)")
    axis.set_ylabel("days of cover")
    axis.set_title("Stock coverage against forecast demand")
    axis.axhline(30, color=PALETTE[5], linestyle="--", linewidth=1.4, label="30-day horizon")
    axis.axhline(90, color=PALETTE[4], linestyle=":", linewidth=1.4, label="90-day overstock line")
    axis.set_yscale("log")
    axis.legend(frameon=False)
    return _save(
        figure,
        out / "05_coverage_vs_demand.png",
        "Coverage is uncorrelated with demand size, so neither low-volume nor high-volume "
        "products are systematically over- or under-stocked. Both failure modes exist "
        "simultaneously, which is exactly the business problem the platform addresses.",
    )


def expiry_windows(batches: pd.DataFrame, snapshot: str, out: Path) -> dict[str, str]:
    days = (pd.to_datetime(batches["expiry_date"]) - pd.Timestamp(snapshot)).dt.days
    bins = [-1000, 0, 30, 60, 90, 180, 365, 10_000]
    labels = ["expired", "0-30", "31-60", "61-90", "91-180", "181-365", ">365"]
    counts = pd.cut(days, bins=bins, labels=labels).value_counts().reindex(labels)

    figure, axis = plt.subplots(figsize=(8.4, 4))
    bars = axis.bar(counts.index, counts.to_numpy(), color=PALETTE[3])
    axis.set_ylabel("batches")
    axis.set_title(f"Batches by days to expiry at the snapshot ({snapshot})")
    for bar, value in zip(bars, counts.to_numpy(), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return _save(
        figure,
        out / "06_expiry_windows.png",
        "Roughly a third of batches expire within 90 days, which is the window where "
        "forecast-driven rotation advice can still change the outcome. Batches beyond a "
        "year are effectively unconstrained by the 30-day forecast.",
    )


def margin_by_category(products: pd.DataFrame, out: Path) -> dict[str, str]:
    frame = products.copy()
    frame["margin_pct"] = (frame["unit_price"] - frame["unit_cost"]) / frame["unit_price"] * 100
    figure, axis = plt.subplots(figsize=(9, 4))
    sns.boxplot(
        data=frame,
        x="category",
        y="margin_pct",
        ax=axis,
        color=PALETTE[0],
        fliersize=2,
        linewidth=0.9,
    )
    axis.set_title("Gross margin distribution by category")
    axis.set_ylabel("margin (%)")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=22)
    return _save(
        figure,
        out / "07_margin_by_category.png",
        "Margins cluster between 20% and 45% with no category-level outliers, so expiry "
        "exposure in currency terms tracks unit exposure rather than margin distortion.",
    )


def top_products(panel: pd.DataFrame, products: pd.DataFrame, out: Path) -> dict[str, str]:
    totals = panel.groupby("product_id")["quantity"].sum().sort_values(ascending=False).head(12)
    names = products.set_index("product_id")["medicine_name"].reindex(totals.index)
    labels = [f"{pid} · {name[:26]}" for pid, name in zip(totals.index, names, strict=True)]

    figure, axis = plt.subplots(figsize=(9, 4.6))
    axis.barh(labels[::-1], totals.to_numpy()[::-1], color=PALETTE[0])
    axis.set_xlabel("units sold over the full history")
    axis.set_title("Top 12 products by total demand")
    return _save(
        figure,
        out / "08_top_products.png",
        "Demand is concentrated: the top 12 of 120 products account for a disproportionate "
        "share of volume, so a small number of forecast errors dominates total cost.",
    )


def write_report(figures: list[dict[str, str]], path: Path, summary: dict) -> Path:
    lines = [
        "# Exploratory Data Analysis",
        "",
        "Generated by `python -m scripts.run_eda` from the *processed* dataset, so every",
        "figure reflects the data the model actually trains on.",
        "",
        "> Synthetic data. Patterns below are properties of the simulator, not of a real",
        "> pharmacy. They are reported so the modelling choices can be justified against",
        "> the data-generating process.",
        "",
        "## Dataset at a glance",
        "",
        f"- Products: **{summary['n_products']:,}**",
        f"- Sales transactions: **{summary['n_sales']:,}**",
        f"- Product-days in demand panel: **{summary['n_panel_rows']:,}**",
        f"- Batches at snapshot: **{summary['n_batches']:,}**",
        f"- Date range: **{summary['date_min']} to {summary['date_max']}**",
        f"- Zero-demand share: **{summary['zero_share']:.1%}**",
        "",
        "## Findings",
        "",
    ]
    for index, figure in enumerate(figures, start=1):
        lines += [
            f"### {index}. {figure['title']}",
            "",
            f"![{figure['title']}](figures/{figure['file']})",
            "",
            figure["caption"],
            "",
        ]

    lines += [
        "## What this implies for the model",
        "",
        "1. **Calendar features matter.** Demand has clear annual and weekly structure, so",
        "   origin month/quarter and demand-history features are justified.",
        "2. **WAPE, not MAPE.** Genuine zero-demand days make percentage error undefined on",
        "   many rows.",
        "3. **Sparse products need honest uncertainty.** Aggregate forecasts for slow movers",
        "   are inherently weak; the UI flags them rather than pretending precision.",
        "4. **Inventory and expiry are decision layers, not demand drivers.** Stock is only",
        "   observed at one date, so it is excluded from the forecasting model (D-011).",
        "",
        "## Data-quality problems found",
        "",
        "- 46 transactions sell below recorded unit cost. These coincide with promoted days",
        "  and are retained: a pharmacy does sell below list cost during promotions. They are",
        "  reported as a warning in `reports/data_quality_report.json` rather than deleted.",
        "- No duplicate keys, no referential breaks, and batch quantities reconcile exactly",
        "  with on-hand stock (asserted in the test suite).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_eda(
    products: pd.DataFrame,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    batches: pd.DataFrame,
    panel: pd.DataFrame,
    forecast: pd.DataFrame,
    snapshot: str,
    figures_dir: Path,
    report_path: Path,
) -> Path:
    figures: list[dict[str, str]] = []

    def add(result: dict[str, str], title: str) -> None:
        figures.append({**result, "title": title})

    add(category_demand(panel, products, figures_dir), "Category demand")
    add(monthly_pattern(panel, figures_dir), "Monthly seasonality")
    add(weekday_pattern(sales, products, figures_dir), "Weekday effects")
    add(zero_demand_profile(panel, figures_dir), "Zero-demand structure")
    add(inventory_vs_forecast(inventory, forecast, figures_dir), "Coverage versus demand")
    add(expiry_windows(batches, snapshot, figures_dir), "Expiry windows")
    add(margin_by_category(products, figures_dir), "Margin distribution")
    add(top_products(panel, products, figures_dir), "Demand concentration")

    summary = {
        "n_products": int(products["product_id"].nunique()),
        "n_sales": int(len(sales)),
        "n_panel_rows": int(len(panel)),
        "n_batches": int(len(batches)),
        "date_min": str(pd.to_datetime(panel["date"]).min().date()),
        "date_max": str(pd.to_datetime(panel["date"]).max().date()),
        "zero_share": float((panel["quantity"] == 0).mean()),
    }
    return write_report(figures, report_path, summary)
