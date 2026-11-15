"""Synthetic pharmacy simulator.

Pipeline: products -> daily demand matrix -> sales transactions -> inventory
snapshot -> batch ledger. Every stochastic draw comes from a named child stream
(see ``ml/common/seeding.py``), so the output is reproducible and independent of
call order.

Honest framing: this simulates *plausible* pharmacy demand, not observed demand.
The generative structure is documented in ``docs/data_generation.md``; the
limitations are in ``LIMITATIONS.md`` section 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.common.io import write_csv, write_json, write_parquet
from ml.common.seeding import (
    STREAM_BATCHES,
    STREAM_INVENTORY,
    STREAM_PRODUCTS,
    STREAM_SALES,
    rng_for,
)
from ml.data_generation.catalog import build_medicine_name
from ml.data_generation.config import (
    CATEGORY_PROFILES,
    MANUFACTURERS,
    SUPPLIERS,
    GeneratorConfig,
)

CATEGORY_ORDER: tuple[str, ...] = tuple(CATEGORY_PROFILES.keys())

PUBLIC_PRODUCT_COLUMNS: list[str] = [
    "product_id",
    "medicine_name",
    "category",
    "manufacturer",
    "unit_price",
    "unit_cost",
    "supplier_id",
    "minimum_stock_level",
    "lead_time_days",
]

SALES_COLUMNS: list[str] = [
    "transaction_id",
    "product_id",
    "transaction_date",
    "quantity_sold",
    "selling_price",
    "discount",
    "day_of_week",
    "month",
    "season",
    "is_weekend",
    "is_festival_window",
]

INVENTORY_COLUMNS: list[str] = [
    "product_id",
    "current_stock",
    "reorder_level",
    "last_restock_date",
    "supplier_lead_time",
]

BATCH_COLUMNS: list[str] = [
    "batch_id",
    "product_id",
    "batch_quantity",
    "remaining_quantity",
    "manufacturing_date",
    "expiry_date",
    "batch_arrival_date",
]

SEASON_BY_MONTH: dict[int, str] = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


@dataclass
class GeneratedDataset:
    products: pd.DataFrame
    sales: pd.DataFrame
    inventory: pd.DataFrame
    batches: pd.DataFrame
    demand_panel: pd.DataFrame
    config: GeneratorConfig
    dates: pd.DatetimeIndex


# ----------------------------------------------------------------------------
# Products
# ----------------------------------------------------------------------------
def generate_products(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Create the product master plus the latent parameters used by the simulator.

    Columns starting with ``_`` are *simulator internals* and are never written to
    ``products.csv``. Keeping them attached makes the demand model auditable while
    keeping the published schema clean.
    """
    rows: list[dict[str, object]] = []
    category_counts: dict[str, int] = {category: 0 for category in CATEGORY_ORDER}

    for index in range(cfg.n_products):
        category = CATEGORY_ORDER[index % len(CATEGORY_ORDER)]
        profile = CATEGORY_PROFILES[category]
        position_in_category = category_counts[category]
        category_counts[category] += 1

        supplier = SUPPLIERS[index % len(SUPPLIERS)]
        lead_time = int(
            supplier.base_lead_time_days + rng.integers(0, supplier.lead_time_jitter_days + 1)
        )
        price_low, price_high = profile.price_range
        unit_price = float(np.round(rng.uniform(price_low, price_high), 2))
        unit_cost = float(np.round(unit_price * rng.uniform(0.55, 0.80), 2))

        # Latent demand behaviour
        demand_scale = float(np.clip(rng.lognormal(mean=0.0, sigma=0.38), 0.25, 3.0))
        trend = float(rng.uniform(-0.25, 0.35))
        is_intermittent = bool(rng.random() < cfg.intermittent_share)
        if is_intermittent:
            active_prob = float(rng.uniform(*cfg.intermittent_active_prob_range))
        else:
            active_prob = 1.0

        cover_low, cover_high = _pick_band(rng, cfg.cover_days_mixture)
        cover_days = float(rng.uniform(cover_low, cover_high))

        shelf_low, shelf_high = profile.shelf_life_days
        shelf_life = int(rng.integers(shelf_low, shelf_high + 1))

        rows.append(
            {
                "product_id": f"MED-{index + 1:04d}",
                "medicine_name": build_medicine_name(category, position_in_category),
                "category": category,
                "manufacturer": MANUFACTURERS[index % len(MANUFACTURERS)],
                "unit_price": unit_price,
                "unit_cost": unit_cost,
                "supplier_id": supplier.supplier_id,
                "minimum_stock_level": int(
                    np.ceil(profile.base_daily_units * demand_scale * lead_time * 1.1)
                ),
                "lead_time_days": lead_time,
                "_demand_scale": demand_scale,
                "_trend": trend,
                "_is_intermittent": is_intermittent,
                "_active_prob": active_prob,
                "_cover_days": cover_days,
                "_shelf_life_days": shelf_life,
            }
        )

    return pd.DataFrame(rows)


def _pick_band(
    rng: np.random.Generator, bands: tuple[tuple[float, float, float], ...]
) -> tuple[float, float]:
    """Choose one (weight, low, high) band by its weight and return its bounds."""
    weights = np.array([band[0] for band in bands], dtype=float)
    weights = weights / weights.sum()
    chosen = int(rng.choice(len(bands), p=weights))
    return bands[chosen][1], bands[chosen][2]


# ----------------------------------------------------------------------------
# Demand
# ----------------------------------------------------------------------------
def simulate_daily_demand(
    cfg: GeneratorConfig, products: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Simulate a (n_products x n_days) integer demand matrix.

    Returns ``(counts, promo_mask, dates)``. Zero cells are genuine zero-demand
    days: the pharmacy sold nothing that day.
    """
    dates = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="D")
    n_products = len(products)
    n_days = len(dates)

    day_of_week = dates.dayofweek.to_numpy()
    day_of_year = dates.dayofyear.to_numpy()
    years_elapsed = np.arange(n_days, dtype=float) / 365.25

    categories = products["category"].to_numpy()
    scales = products["_demand_scale"].to_numpy()
    trends = products["_trend"].to_numpy()

    mu = np.zeros((n_products, n_days), dtype=float)
    for category in CATEGORY_ORDER:
        positions = np.flatnonzero(categories == category)
        if positions.size == 0:
            continue
        profile = CATEGORY_PROFILES[category]
        weekday = np.asarray(profile.weekday_factors, dtype=float)[day_of_week]
        seasonal = 1.0 + profile.seasonal_amplitude * np.sin(
            2.0 * np.pi * (day_of_year - profile.seasonal_peak_day) / 365.25
        )
        shape = profile.base_daily_units * weekday * seasonal

        mu[positions, :] = (
            scales[positions, None]
            * (1.0 + trends[positions, None] * years_elapsed[None, :])
            * shape[None, :]
        )

    # Occasional promotion spikes (uplift applies only on promotion days).
    promo_mask = rng.random((n_products, n_days)) < cfg.promo_day_share
    uplift = rng.uniform(*cfg.promo_uplift_range, size=(n_products, n_days))
    mu = np.where(promo_mask, mu * uplift, mu)

    counts = rng.poisson(np.clip(mu, 0.0, None))

    # Intermittent (slow-moving) products: many genuine zero-sales days.
    active_prob = products["_active_prob"].to_numpy()
    active = rng.random((n_products, n_days)) < active_prob[:, None]
    counts = np.where(active, counts, 0).astype(np.int64)

    return counts, promo_mask, dates


def build_sales(
    cfg: GeneratorConfig,
    products: pd.DataFrame,
    counts: np.ndarray,
    promo_mask: np.ndarray,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Convert the demand matrix into transaction rows (positive sales only).

    A pharmacy records sales, not absence of sales. The complete product x day
    grid (including zeros) is derived separately as the demand panel.
    """
    product_positions, day_positions = np.nonzero(counts)
    if product_positions.size == 0:
        return pd.DataFrame(columns=SALES_COLUMNS)

    order = np.lexsort((product_positions, day_positions))
    product_positions = product_positions[order]
    day_positions = day_positions[order]

    quantity = counts[product_positions, day_positions].astype(np.int64)
    product_ids = products["product_id"].to_numpy()[product_positions]
    unit_price = products["unit_price"].to_numpy()[product_positions]
    on_promo = promo_mask[product_positions, day_positions]

    discount = np.where(
        on_promo,
        rng.uniform(*cfg.discount_on_promo_range, size=quantity.size),
        np.where(
            rng.random(quantity.size) < cfg.discount_off_promo_prob,
            0.05,
            0.0,
        ),
    ).round(4)

    price_noise = 1.0 + rng.normal(0.0, cfg.price_noise_sigma, size=quantity.size)
    selling_price = np.clip(np.round(unit_price * (1.0 - discount) * price_noise, 2), 0.01, None)

    transaction_dates = dates.to_numpy()[day_positions]
    frame = pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i + 1:08d}" for i in range(quantity.size)],
            "product_id": product_ids,
            "transaction_date": pd.to_datetime(transaction_dates),
            "quantity_sold": quantity,
            "selling_price": selling_price,
            "discount": discount,
        }
    )
    frame["day_of_week"] = frame["transaction_date"].dt.dayofweek
    frame["month"] = frame["transaction_date"].dt.month
    frame["season"] = frame["month"].map(SEASON_BY_MONTH)
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    # Oct-Nov festive/seasonal trading window (Diwali + year-end).
    frame["is_festival_window"] = frame["month"].isin([10, 11]).astype(int)

    return frame[SALES_COLUMNS].reset_index(drop=True)


def build_demand_panel(
    products: pd.DataFrame, counts: np.ndarray, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Complete product x day grid, zeros included (derived artefact)."""
    n_products = len(products)
    n_days = len(dates)
    return pd.DataFrame(
        {
            "product_id": np.repeat(products["product_id"].to_numpy(), n_days),
            "date": np.tile(dates.to_numpy(), n_products),
            "quantity": counts.reshape(-1),
        }
    )


# ----------------------------------------------------------------------------
# Inventory and batches
# ----------------------------------------------------------------------------
def build_inventory(
    cfg: GeneratorConfig,
    products: pd.DataFrame,
    counts: np.ndarray,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Snapshot stock at ``cfg.snapshot_date`` from trailing 90-day demand."""
    snapshot_position = int(dates.searchsorted(pd.Timestamp(cfg.snapshot_date), side="right"))
    window_start = max(0, snapshot_position - 90)
    trailing = counts[:, window_start:snapshot_position]
    mean_daily = trailing.mean(axis=1) if trailing.size else np.zeros(len(products))

    lead_time = products["lead_time_days"].to_numpy()
    cover_days = products["_cover_days"].to_numpy()
    current_stock = np.maximum(np.round(mean_daily * cover_days), 0).astype(np.int64)
    # A few products are deliberately out of stock (edge case for inference).
    out_of_stock = rng.random(len(products)) < 0.03
    current_stock = np.where(out_of_stock, 0, current_stock)

    reorder_level = np.ceil(mean_daily * lead_time * cfg.safety_factor + 3).astype(np.int64)
    restock_offset = rng.integers(1, lead_time + 8)
    last_restock = pd.Timestamp(cfg.snapshot_date) - pd.to_timedelta(restock_offset, unit="D")

    return pd.DataFrame(
        {
            "product_id": products["product_id"].to_numpy(),
            "current_stock": current_stock,
            "reorder_level": reorder_level,
            "last_restock_date": last_restock,
            "supplier_lead_time": lead_time,
        }
    )


def build_batches(
    cfg: GeneratorConfig,
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Split each product's stock into 1-4 batches with varied expiry windows.

    Invariants enforced here (and asserted in tests):
      * sum(remaining_quantity) == current_stock, per product
      * manufacturing_date < expiry_date
      * batch_arrival_date <= snapshot_date
      * remaining_quantity <= batch_quantity
    """
    snapshot = pd.Timestamp(cfg.snapshot_date)
    dataset_start = pd.Timestamp(cfg.start_date)
    rows: list[dict[str, object]] = []

    stock_by_product = dict(zip(inventory["product_id"], inventory["current_stock"], strict=True))
    product_ids = products["product_id"].to_numpy()
    shelf_lives = products["_shelf_life_days"].to_numpy()

    # NOTE: iterate by position rather than ``itertuples`` -- pandas renames
    # columns whose names start with "_" (they are invalid namedtuple fields),
    # which would silently drop the simulator's latent parameters.
    for position in range(len(products)):
        product_id = str(product_ids[position])
        stock = int(stock_by_product[product_id])
        if stock <= 0:
            continue

        n_batches = int(rng.integers(1, cfg.max_batches_per_product + 1))
        n_batches = min(n_batches, stock)
        weights = rng.dirichlet(np.ones(n_batches))

        # Every batch gets at least one unit; the remainder is multinomially split.
        remainder = stock - n_batches
        extra = (
            rng.multinomial(remainder, weights)
            if remainder > 0
            else np.zeros(n_batches, dtype=np.int64)
        )
        remaining = np.ones(n_batches, dtype=np.int64) + extra

        band_lows = np.array([band[1] for band in cfg.expiry_bands_days], dtype=float)
        band_highs = np.array([band[2] for band in cfg.expiry_bands_days], dtype=float)
        band_weights = np.array([band[0] for band in cfg.expiry_bands_days], dtype=float)
        band_weights = band_weights / band_weights.sum()
        chosen_bands = rng.choice(len(cfg.expiry_bands_days), size=n_batches, p=band_weights)
        days_to_expiry = np.array(
            [rng.uniform(band_lows[band], band_highs[band]) for band in chosen_bands]
        )

        shelf_life = int(shelf_lives[position])
        batch_records: list[dict[str, object]] = []
        for batch_index in range(n_batches):
            expiry = snapshot + pd.Timedelta(days=float(days_to_expiry[batch_index]))
            manufacturing = expiry - pd.Timedelta(days=shelf_life)
            if manufacturing > snapshot - pd.Timedelta(days=20):
                # Not yet manufactured-and-received stock; pull it into the past.
                manufacturing = snapshot - pd.Timedelta(days=int(rng.integers(20, 121)))
            arrival = manufacturing + pd.Timedelta(days=int(rng.integers(5, 46)))
            arrival = min(max(arrival, dataset_start), snapshot)
            if arrival < manufacturing:
                arrival = manufacturing

            sold_fraction = float(rng.uniform(*cfg.sold_fraction_range))
            batch_quantity = int(np.ceil(remaining[batch_index] / (1.0 - sold_fraction)))
            batch_quantity = max(batch_quantity, int(remaining[batch_index]))

            batch_records.append(
                {
                    "product_id": product_id,
                    "batch_quantity": batch_quantity,
                    "remaining_quantity": int(remaining[batch_index]),
                    "manufacturing_date": manufacturing.normalize(),
                    "expiry_date": expiry.normalize(),
                    "batch_arrival_date": arrival.normalize(),
                }
            )

        # FEFO ordering: earliest expiry first. Batch ids encode that order.
        batch_records.sort(key=lambda record: record["expiry_date"])
        for batch_index, record in enumerate(batch_records, start=1):
            rows.append(
                {
                    "batch_id": f"BATCH-{position + 1:04d}-{batch_index:02d}",
                    **record,
                }
            )

    frame = pd.DataFrame(rows, columns=BATCH_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["product_id", "expiry_date", "batch_id"], kind="mergesort"
        ).reset_index(drop=True)
    return frame


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def generate_dataset(cfg: GeneratorConfig | None = None) -> GeneratedDataset:
    """Generate the whole dataset deterministically from one seed."""
    cfg = cfg or GeneratorConfig()
    products = generate_products(cfg, rng_for(cfg.seed, STREAM_PRODUCTS))
    counts, promo_mask, dates = simulate_daily_demand(
        cfg, products, rng_for(cfg.seed, STREAM_SALES)
    )
    sales = build_sales(cfg, products, counts, promo_mask, dates, rng_for(cfg.seed, STREAM_SALES))
    inventory = build_inventory(cfg, products, counts, dates, rng_for(cfg.seed, STREAM_INVENTORY))
    batches = build_batches(cfg, products, inventory, rng_for(cfg.seed, STREAM_BATCHES))
    panel = build_demand_panel(products, counts, dates)
    return GeneratedDataset(
        products=products,
        sales=sales,
        inventory=inventory,
        batches=batches,
        demand_panel=panel,
        config=cfg,
        dates=dates,
    )


def write_dataset(dataset: GeneratedDataset, raw_dir: Path, interim_dir: Path) -> dict:
    """Persist the dataset canonically and return a provenance manifest."""
    paths = [
        write_csv(dataset.products, raw_dir / "products.csv", PUBLIC_PRODUCT_COLUMNS),
        write_csv(dataset.sales, raw_dir / "sales.csv", SALES_COLUMNS),
        write_csv(dataset.inventory, raw_dir / "inventory.csv", INVENTORY_COLUMNS),
        write_csv(dataset.batches, raw_dir / "batches.csv", BATCH_COLUMNS),
        write_parquet(dataset.demand_panel, interim_dir / "demand_panel.parquet"),
    ]

    from ml.common.io import file_fingerprints

    meta = {
        "dataset_version": dataset.config.dataset_version,
        "seed": dataset.config.seed,
        "generator": "ml/data_generation/generate.py",
        "is_synthetic": True,
        "provenance_note": (
            "Synthetic pharmacy data generated by the MediStock AI simulator. "
            "Not real pharmacy transaction data."
        ),
        "date_range": {
            "start": dataset.config.start_date,
            "end": dataset.config.end_date,
            "days": len(dataset.dates),
        },
        "snapshot_date": dataset.config.snapshot_date,
        "horizon_days": dataset.config.horizon_days,
        "n_products": int(len(dataset.products)),
        "n_sales_transactions": int(len(dataset.sales)),
        "n_demand_panel_rows": int(len(dataset.demand_panel)),
        "n_inventory_rows": int(len(dataset.inventory)),
        "n_batches": int(len(dataset.batches)),
        "files": file_fingerprints(paths, raw_dir.parent),
        "config": dataset.config.to_dict(),
    }
    write_json(meta, raw_dir / "dataset_meta.json")
    return meta
