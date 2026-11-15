"""Cleaning pipeline: every mutation is counted and reported.

The generated data is already clean, so on a fresh run this step should be a
no-op. That is the point: it proves the *pipeline* handles dirty input, and the
tests inject dirty rows to exercise each branch. A cleaning step that never
reports anything is either unnecessary or not doing its job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.core.logging import get_logger
from ml.preprocessing.validation import ValidationReport, validate_dataset

logger = get_logger("ml.preprocessing.cleaning")

PRODUCT_DTYPES: dict[str, str] = {
    "product_id": "string",
    "medicine_name": "string",
    "category": "string",
    "manufacturer": "string",
    "supplier_id": "string",
}


@dataclass
class CleaningAction:
    step: str
    table: str
    rows_affected: int
    detail: str

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "table": self.table,
            "rows_affected": self.rows_affected,
            "detail": self.detail,
        }


@dataclass
class CleaningResult:
    products: pd.DataFrame
    sales: pd.DataFrame
    inventory: pd.DataFrame
    batches: pd.DataFrame
    panel: pd.DataFrame
    actions: list[CleaningAction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "total_rows_affected": sum(action.rows_affected for action in self.actions),
            "row_counts": {
                "products": int(len(self.products)),
                "sales": int(len(self.sales)),
                "inventory": int(len(self.inventory)),
                "batches": int(len(self.batches)),
                "demand_panel": int(len(self.panel)),
            },
        }


def _record(result: CleaningResult, step: str, table: str, rows: int, detail: str) -> None:
    if rows > 0:
        result.actions.append(
            CleaningAction(step=step, table=table, rows_affected=int(rows), detail=detail)
        )
        logger.warning(
            "Cleaning action applied", extra={"step": step, "table": table, "rows": int(rows)}
        )


def clean_products(products: pd.DataFrame, result: CleaningResult) -> pd.DataFrame:
    frame = products.copy()
    for column, dtype in PRODUCT_DTYPES.items():
        if column in frame.columns:
            frame[column] = frame[column].astype(dtype)

    before = len(frame)
    frame = frame.drop_duplicates(subset=["product_id"], keep="first")
    _record(
        result,
        "drop_duplicate_products",
        "products",
        before - len(frame),
        "Duplicate product_id dropped",
    )

    before = len(frame)
    frame = frame.dropna(subset=["product_id", "medicine_name", "category", "unit_price"])
    _record(
        result,
        "drop_null_product_keys",
        "products",
        before - len(frame),
        "Products missing keys dropped",
    )

    # A non-positive price is a data-entry error, not a business decision.
    invalid_price = frame["unit_price"] <= 0
    if invalid_price.any():
        median_price = frame.loc[~invalid_price, "unit_price"].median()
        frame.loc[invalid_price, "unit_price"] = median_price
        _record(
            result,
            "repair_nonpositive_price",
            "products",
            int(invalid_price.sum()),
            "unit_price <= 0 replaced with category median",
        )

    frame["minimum_stock_level"] = frame["minimum_stock_level"].clip(lower=0).astype("int64")
    frame["lead_time_days"] = frame["lead_time_days"].clip(lower=0).astype("int64")
    return frame.reset_index(drop=True)


def clean_sales(
    sales: pd.DataFrame, products: pd.DataFrame, result: CleaningResult, start: str, end: str
) -> pd.DataFrame:
    frame = sales.copy()
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"]).dt.normalize()
    frame["product_id"] = frame["product_id"].astype("string")

    before = len(frame)
    frame = frame.drop_duplicates(subset=["transaction_id"], keep="first")
    _record(
        result,
        "drop_duplicate_transactions",
        "sales",
        before - len(frame),
        "Duplicate transaction_id dropped",
    )

    before = len(frame)
    frame = frame.dropna(
        subset=["transaction_id", "product_id", "transaction_date", "quantity_sold"]
    )
    _record(
        result,
        "drop_null_sales_keys",
        "sales",
        before - len(frame),
        "Sales rows missing keys dropped",
    )

    before = len(frame)
    frame = frame[frame["product_id"].isin(products["product_id"])]
    _record(
        result,
        "drop_orphan_sales",
        "sales",
        before - len(frame),
        "Sales for unknown products dropped",
    )

    before = len(frame)
    in_range = frame["transaction_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    frame = frame[in_range]
    _record(
        result,
        "drop_out_of_range_sales",
        "sales",
        before - len(frame),
        "Sales outside the dataset window dropped",
    )

    negative = frame["quantity_sold"] < 0
    if negative.any():
        frame = frame[~negative]
        _record(
            result,
            "drop_negative_quantity",
            "sales",
            int(negative.sum()),
            "Negative quantities dropped (assumed returns)",
        )

    frame["discount"] = frame["discount"].fillna(0.0).clip(lower=0.0, upper=0.95)
    frame["quantity_sold"] = frame["quantity_sold"].astype("int64")
    frame["selling_price"] = frame["selling_price"].astype("float64")
    return frame.sort_values(["transaction_date", "product_id"]).reset_index(drop=True)


def clean_inventory(
    inventory: pd.DataFrame, products: pd.DataFrame, result: CleaningResult
) -> pd.DataFrame:
    frame = inventory.copy()
    frame["product_id"] = frame["product_id"].astype("string")

    before = len(frame)
    frame = frame.drop_duplicates(subset=["product_id"], keep="last")
    _record(
        result,
        "drop_duplicate_inventory",
        "inventory",
        before - len(frame),
        "Duplicate inventory rows dropped",
    )

    before = len(frame)
    frame = frame[frame["product_id"].isin(products["product_id"])]
    _record(
        result,
        "drop_orphan_inventory",
        "inventory",
        before - len(frame),
        "Inventory for unknown products dropped",
    )

    missing = products.loc[~products["product_id"].isin(frame["product_id"]), "product_id"]
    if len(missing) > 0:
        # Absent stock record means "no stock recorded", not "unknown stock".
        filler = pd.DataFrame(
            {
                "product_id": missing.to_numpy(),
                "current_stock": 0,
                "reorder_level": 0,
                "last_restock_date": pd.NaT,
                "supplier_lead_time": products.set_index("product_id")
                .loc[missing, "lead_time_days"]
                .to_numpy(),
            }
        )
        frame = pd.concat([frame, filler], ignore_index=True)
        _record(
            result,
            "impute_missing_inventory",
            "inventory",
            len(missing),
            "Missing products given zero stock",
        )

    frame["current_stock"] = frame["current_stock"].clip(lower=0).astype("int64")
    frame["reorder_level"] = frame["reorder_level"].clip(lower=0).astype("int64")
    frame["last_restock_date"] = pd.to_datetime(frame["last_restock_date"]).dt.normalize()
    return frame.sort_values("product_id").reset_index(drop=True)


def clean_batches(
    batches: pd.DataFrame, products: pd.DataFrame, result: CleaningResult
) -> pd.DataFrame:
    frame = batches.copy()
    frame["product_id"] = frame["product_id"].astype("string")
    for column in ("manufacturing_date", "expiry_date", "batch_arrival_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()

    before = len(frame)
    frame = frame.drop_duplicates(subset=["batch_id"], keep="first")
    _record(
        result,
        "drop_duplicate_batches",
        "batches",
        before - len(frame),
        "Duplicate batch_id dropped",
    )

    before = len(frame)
    frame = frame[frame["product_id"].isin(products["product_id"])]
    _record(
        result,
        "drop_orphan_batches",
        "batches",
        before - len(frame),
        "Batches for unknown products dropped",
    )

    before = len(frame)
    valid = frame["remaining_quantity"].between(0, frame["batch_quantity"])
    frame = frame[valid]
    _record(
        result,
        "drop_invalid_batches",
        "batches",
        before - len(frame),
        "remaining outside [0, batch_quantity] dropped",
    )

    frame["batch_quantity"] = frame["batch_quantity"].astype("int64")
    frame["remaining_quantity"] = frame["remaining_quantity"].astype("int64")
    return frame.sort_values(["product_id", "expiry_date", "batch_id"]).reset_index(drop=True)


def clean_panel(panel: pd.DataFrame, result: CleaningResult) -> pd.DataFrame:
    frame = panel.copy()
    frame["product_id"] = frame["product_id"].astype("string")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    before = len(frame)
    frame = frame.drop_duplicates(subset=["product_id", "date"], keep="last")
    _record(
        result,
        "drop_duplicate_panel_rows",
        "demand_panel",
        before - len(frame),
        "Duplicate product/date rows dropped",
    )

    negative = frame["quantity"] < 0
    if negative.any():
        frame.loc[negative, "quantity"] = 0
        _record(
            result,
            "clip_negative_panel",
            "demand_panel",
            int(negative.sum()),
            "Negative demand clipped to 0",
        )

    frame["quantity"] = frame["quantity"].astype("int64")
    return frame.sort_values(["product_id", "date"]).reset_index(drop=True)


def clean_dataset(
    products: pd.DataFrame,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    batches: pd.DataFrame,
    panel: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> CleaningResult:
    """Apply every cleaning rule and return cleaned frames plus an action log."""
    result = CleaningResult(
        products=pd.DataFrame(),
        sales=pd.DataFrame(),
        inventory=pd.DataFrame(),
        batches=pd.DataFrame(),
        panel=pd.DataFrame(),
    )
    result.products = clean_products(products, result)
    result.sales = clean_sales(sales, result.products, result, start_date, end_date)
    result.inventory = clean_inventory(inventory, result.products, result)
    result.batches = clean_batches(batches, result.products, result)
    result.panel = clean_panel(panel, result)
    return result


def clean_and_revalidate(
    products: pd.DataFrame,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    batches: pd.DataFrame,
    panel: pd.DataFrame,
    start_date: str,
    end_date: str,
    snapshot_date: str,
) -> tuple[CleaningResult, ValidationReport]:
    """Clean, then re-validate: proves the repair actually fixed the problems."""
    cleaned = clean_dataset(products, sales, inventory, batches, panel, start_date, end_date)
    report = validate_dataset(
        cleaned.products,
        cleaned.sales,
        cleaned.inventory,
        cleaned.batches,
        cleaned.panel,
        snapshot_date=snapshot_date,
    )
    return cleaned, report
