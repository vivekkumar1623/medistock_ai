"""Dataset validation: explicit rules, machine-readable findings.

Philosophy: validation never "fixes" anything. It reports. Repair is a separate,
auditable step (`cleaning.py`) so that every mutation can be counted and
explained. A pipeline that silently repairs data cannot tell you how broken the
data was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Severity = Literal["error", "warning"]


@dataclass
class ValidationIssue:
    check: str
    severity: Severity
    table: str
    message: str
    count: int
    sample: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "table": self.table,
            "message": self.message,
            "count": self.count,
            "sample": self.sample,
        }


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        check: str,
        severity: Severity,
        table: str,
        message: str,
        count: int,
        sample: list[str] | None = None,
    ) -> None:
        if count <= 0:
            return
        self.issues.append(
            ValidationIssue(
                check=check,
                severity=severity,
                table=table,
                message=message,
                count=int(count),
                sample=(sample or [])[:5],
            )
        )

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "row_counts": self.row_counts,
            "checks_run": len({issue.check for issue in self.issues}) or None,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _ids(frame: pd.DataFrame, column: str) -> list[str]:
    return frame[column].astype(str).head(5).tolist()


def validate_products(products: pd.DataFrame, report: ValidationReport) -> None:
    table = "products"
    report.row_counts[table] = int(len(products))

    report.add(
        "products.pk_unique",
        "error",
        table,
        "Duplicate product_id values.",
        int(products["product_id"].duplicated().sum()),
    )
    required = [
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
    report.add(
        "products.required_columns",
        "error",
        table,
        f"Missing required columns: {sorted(set(required) - set(products.columns))}",
        int(len(set(required) - set(products.columns))),
    )
    if not set(required) <= set(products.columns):
        return

    report.add(
        "products.null_required",
        "error",
        table,
        "Nulls in required product fields.",
        int(products[required].isna().any(axis=1).sum()),
        _ids(products[products[required].isna().any(axis=1)], "product_id"),
    )
    report.add(
        "products.nonpositive_price",
        "error",
        table,
        "unit_price must be > 0.",
        int((products["unit_price"] <= 0).sum()),
    )
    report.add(
        "products.nonpositive_cost",
        "error",
        table,
        "unit_cost must be > 0.",
        int((products["unit_cost"] <= 0).sum()),
    )
    below_cost = products["unit_cost"] >= products["unit_price"]
    report.add(
        "products.cost_ge_price",
        "warning",
        table,
        "unit_cost >= unit_price: negative margin product.",
        int(below_cost.sum()),
        _ids(products[below_cost], "product_id"),
    )
    report.add(
        "products.negative_lead_time",
        "error",
        table,
        "lead_time_days must be >= 0.",
        int((products["lead_time_days"] < 0).sum()),
    )
    report.add(
        "products.negative_min_stock",
        "error",
        table,
        "minimum_stock_level must be >= 0.",
        int((products["minimum_stock_level"] < 0).sum()),
    )


def validate_sales(sales: pd.DataFrame, products: pd.DataFrame, report: ValidationReport) -> None:
    table = "sales"
    report.row_counts[table] = int(len(sales))

    report.add(
        "sales.pk_unique",
        "error",
        table,
        "Duplicate transaction_id values.",
        int(sales["transaction_id"].duplicated().sum()),
    )
    report.add(
        "sales.fk_product",
        "error",
        table,
        "Sales reference unknown product_id.",
        int((~sales["product_id"].isin(products["product_id"])).sum()),
    )
    report.add(
        "sales.null_keys",
        "error",
        table,
        "Nulls in transaction_id / product_id / transaction_date.",
        int(sales[["transaction_id", "product_id", "transaction_date"]].isna().any(axis=1).sum()),
    )
    report.add(
        "sales.nonpositive_quantity",
        "error",
        table,
        "quantity_sold must be > 0 in the transaction table.",
        int((sales["quantity_sold"] <= 0).sum()),
    )
    report.add(
        "sales.nonpositive_price",
        "error",
        table,
        "selling_price must be > 0.",
        int((sales["selling_price"] <= 0).sum()),
    )
    report.add(
        "sales.discount_range",
        "error",
        table,
        "discount must be within [0, 1).",
        int(((sales["discount"] < 0) | (sales["discount"] >= 1)).sum()),
    )
    duplicated_day = sales.duplicated(subset=["product_id", "transaction_date"], keep=False)
    report.add(
        "sales.duplicate_product_day",
        "warning",
        table,
        "More than one sale row for the same product and date.",
        int(duplicated_day.sum()),
    )
    report.add(
        "sales.price_below_cost",
        "warning",
        table,
        "selling_price below unit_cost (margin erosion, plausible with promotions).",
        int(
            sales.merge(products[["product_id", "unit_cost"]], on="product_id", how="left")
            .eval("selling_price < unit_cost")
            .sum()
        ),
    )


def validate_inventory(
    inventory: pd.DataFrame, products: pd.DataFrame, report: ValidationReport
) -> None:
    table = "inventory"
    report.row_counts[table] = int(len(inventory))

    report.add(
        "inventory.pk_unique",
        "error",
        table,
        "Duplicate product_id rows in inventory snapshot.",
        int(inventory["product_id"].duplicated().sum()),
    )
    report.add(
        "inventory.fk_product",
        "error",
        table,
        "Inventory references unknown product_id.",
        int((~inventory["product_id"].isin(products["product_id"])).sum()),
    )
    missing_products = int(len(set(products["product_id"]) - set(inventory["product_id"])))
    report.add(
        "inventory.missing_product",
        "error",
        table,
        "Products without an inventory row.",
        missing_products,
    )
    report.add(
        "inventory.negative_stock",
        "error",
        table,
        "current_stock must be >= 0.",
        int((inventory["current_stock"] < 0).sum()),
    )
    report.add(
        "inventory.negative_reorder",
        "error",
        table,
        "reorder_level must be >= 0.",
        int((inventory["reorder_level"] < 0).sum()),
    )
    report.add(
        "inventory.lead_time_mismatch",
        "warning",
        table,
        "supplier_lead_time disagrees with products.lead_time_days.",
        int(
            inventory.merge(products[["product_id", "lead_time_days"]], on="product_id", how="left")
            .eval("supplier_lead_time != lead_time_days")
            .sum()
        ),
    )


def validate_batches(
    batches: pd.DataFrame,
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    report: ValidationReport,
    snapshot_date: str,
) -> None:
    table = "batches"
    report.row_counts[table] = int(len(batches))

    # Coerce defensively: validation may receive raw CSV text. Unparseable dates
    # become NaT and are reported below rather than raising a TypeError.
    manufacturing = pd.to_datetime(batches["manufacturing_date"], errors="coerce")
    expiry = pd.to_datetime(batches["expiry_date"], errors="coerce")
    arrival = pd.to_datetime(batches["batch_arrival_date"], errors="coerce")
    report.add(
        "batches.unparseable_dates",
        "error",
        table,
        "Batch date columns contain values that are not valid dates.",
        int(manufacturing.isna().sum() + expiry.isna().sum() + arrival.isna().sum()),
    )

    report.add(
        "batches.pk_unique",
        "error",
        table,
        "Duplicate batch_id values.",
        int(batches["batch_id"].duplicated().sum()),
    )
    report.add(
        "batches.fk_product",
        "error",
        table,
        "Batches reference unknown product_id.",
        int((~batches["product_id"].isin(products["product_id"])).sum()),
    )
    report.add(
        "batches.negative_remaining",
        "error",
        table,
        "remaining_quantity must be >= 0.",
        int((batches["remaining_quantity"] < 0).sum()),
    )
    report.add(
        "batches.remaining_gt_quantity",
        "error",
        table,
        "remaining_quantity exceeds batch_quantity.",
        int((batches["remaining_quantity"] > batches["batch_quantity"]).sum()),
    )
    report.add(
        "batches.expiry_before_manufacture",
        "error",
        table,
        "manufacturing_date must precede expiry_date.",
        int((manufacturing >= expiry).sum()),
    )
    report.add(
        "batches.arrival_after_snapshot",
        "error",
        table,
        "batch_arrival_date is after the snapshot date.",
        int((arrival > pd.Timestamp(snapshot_date)).sum()),
    )
    already_expired = int((expiry <= pd.Timestamp(snapshot_date)).sum())
    report.add(
        "batches.expired_at_snapshot",
        "warning",
        table,
        "Batches already expired at the snapshot date.",
        already_expired,
    )

    batch_totals = batches.groupby("product_id")["remaining_quantity"].sum()
    stock = inventory.set_index("product_id")["current_stock"]
    joined = (
        stock.to_frame(name="current_stock")
        .join(batch_totals.rename("batch_total"), how="left")
        .fillna({"batch_total": 0})
    )
    mismatched = joined["current_stock"] != joined["batch_total"]
    report.add(
        "batches.stock_reconciliation",
        "error",
        table,
        "Sum of remaining batch quantity does not equal current_stock.",
        int(mismatched.sum()),
        [str(i) for i in joined.index[mismatched][:5]],
    )


def validate_panel(panel: pd.DataFrame, products: pd.DataFrame, report: ValidationReport) -> None:
    table = "demand_panel"
    report.row_counts[table] = int(len(panel))

    report.add(
        "panel.fk_product",
        "error",
        table,
        "Panel references unknown product_id.",
        int((~panel["product_id"].isin(products["product_id"])).sum()),
    )
    report.add(
        "panel.negative_quantity",
        "error",
        table,
        "Panel quantity must be >= 0.",
        int((panel["quantity"] < 0).sum()),
    )
    duplicated = panel.duplicated(subset=["product_id", "date"], keep=False)
    report.add(
        "panel.duplicate_product_day",
        "error",
        table,
        "Duplicate product/date rows in the demand panel.",
        int(duplicated.sum()),
    )
    expected = int(products["product_id"].nunique() * panel["date"].nunique())
    report.add(
        "panel.grid_completeness",
        "warning",
        table,
        f"Panel is not a complete product x date grid (expected {expected} rows).",
        int(abs(expected - len(panel))),
    )


def validate_dataset(
    products: pd.DataFrame,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    batches: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    snapshot_date: str = "2025-10-01",
) -> ValidationReport:
    """Run every structural, referential and domain check. Returns a report."""
    report = ValidationReport()
    validate_products(products, report)
    validate_sales(sales, products, report)
    validate_inventory(inventory, products, report)
    validate_batches(batches, products, inventory, report, snapshot_date)
    if panel is not None:
        validate_panel(panel, products, report)
    return report
