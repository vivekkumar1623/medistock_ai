"""ORM entities.

Design notes:

* **Surrogate + natural keys.** Entity tables keep their natural key
  (``product_id``, ``batch_id``, ``transaction_id``) as the primary key because it
  is globally unique in this dataset and makes joins obvious. Generated output
  tables (forecasts, risk assessments, recommendations) use an autoincrement id
  plus a unique constraint, because the same product is scored at many origins.
* **Constraints where the database can enforce them.** Non-negative stock,
  ``remaining <= batch_quantity``, and positive sale quantities are CHECK
  constraints, not just Python assertions -- a direct SQL insert must not be able
  to create an impossible row.
* **Indexes match the query patterns**: sales by product/date, batches by
  product/expiry (FEFO ordering), risk by date/level, recommendations by date and
  action.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin


class Product(Base, TimestampMixin):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    medicine_name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    manufacturer: Mapped[str] = mapped_column(String(120), nullable=False)
    supplier_id: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    unit_cost: Mapped[float] = mapped_column(Float, nullable=False)
    minimum_stock_level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        CheckConstraint("unit_price >= 0", name="unit_price_non_negative"),
        CheckConstraint("unit_cost >= 0", name="unit_cost_non_negative"),
        CheckConstraint("lead_time_days >= 0", name="lead_time_non_negative"),
        CheckConstraint("minimum_stock_level >= 0", name="min_stock_non_negative"),
    )

    inventory: Mapped[InventoryRecord | None] = relationship(
        back_populates="product", uselist=False, cascade="all, delete-orphan"
    )
    batches: Mapped[list[Batch]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )


class InventoryRecord(Base, TimestampMixin):
    __tablename__ = "inventory"

    product_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("products.product_id", ondelete="CASCADE"), primary_key=True
    )
    current_stock: Mapped[int] = mapped_column(Integer, nullable=False)
    reorder_level: Mapped[int] = mapped_column(Integer, nullable=False)
    last_restock_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    supplier_lead_time: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("current_stock >= 0", name="current_stock_non_negative"),
        CheckConstraint("reorder_level >= 0", name="reorder_level_non_negative"),
        CheckConstraint("supplier_lead_time >= 0", name="supplier_lead_time_non_negative"),
    )

    product: Mapped[Product] = relationship(back_populates="inventory")


class Batch(Base, TimestampMixin):
    __tablename__ = "batches"

    batch_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False
    )
    batch_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    remaining_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    manufacturing_date: Mapped[date] = mapped_column(Date, nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    batch_arrival_date: Mapped[date] = mapped_column(Date, nullable=False)

    __table_args__ = (
        CheckConstraint("remaining_quantity >= 0", name="remaining_non_negative"),
        CheckConstraint("remaining_quantity <= batch_quantity", name="remaining_le_batch_quantity"),
        CheckConstraint("manufacturing_date < expiry_date", name="manufacture_before_expiry"),
        Index("ix_batches_product_expiry", "product_id", "expiry_date"),
    )

    product: Mapped[Product] = relationship(back_populates="batches")


class Sale(Base, TimestampMixin):
    __tablename__ = "sales"

    transaction_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False
    )
    transaction_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    quantity_sold: Mapped[int] = mapped_column(Integer, nullable=False)
    selling_price: Mapped[float] = mapped_column(Float, nullable=False)
    discount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[str] = mapped_column(String(16), nullable=False)
    is_weekend: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_festival_window: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        CheckConstraint("quantity_sold > 0", name="quantity_positive"),
        CheckConstraint("selling_price > 0", name="selling_price_positive"),
        CheckConstraint("discount >= 0 AND discount < 1", name="discount_range"),
        Index("ix_sales_product_date", "product_id", "transaction_date"),
    )


class ForecastResult(Base, TimestampMixin):
    """Persisted scoring output, for audit and history."""

    __tablename__ = "forecast_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False
    )
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_demand: Mapped[float] = mapped_column(Float, nullable=False)
    lower_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_version: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "as_of_date",
            "artifact_version",
            name="uq_forecast_product_origin_version",
        ),
    )


class RiskAssessment(Base, TimestampMixin):
    __tablename__ = "risk_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(
        String(24), ForeignKey("batches.batch_id", ondelete="CASCADE"), nullable=False
    )
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    days_to_expiry: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_unsold_units: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("batch_id", "as_of_date", name="uq_risk_batch_date"),
        CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="risk_score_range"),
    )


class Recommendation(Base, TimestampMixin):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False
    )
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    priority: Mapped[str] = mapped_column(String(16), nullable=False)
    suggested_replenishment_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    coverage_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasons_json: Mapped[str] = mapped_column(String(2000), nullable=False, default="[]")

    __table_args__ = (
        UniqueConstraint("product_id", "as_of_date", name="uq_recommendation_product_date"),
        CheckConstraint("suggested_replenishment_qty >= 0", name="suggested_qty_non_negative"),
    )


class DatasetVersion(Base):
    """Provenance row: which dataset produced the loaded operational tables."""

    __tablename__ = "dataset_versions"

    dataset_version: Mapped[str] = mapped_column(String(32), primary_key=True)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    is_synthetic: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    n_products: Mapped[int] = mapped_column(Integer, nullable=False)
    n_sales: Mapped[int] = mapped_column(Integer, nullable=False)
    n_batches: Mapped[int] = mapped_column(Integer, nullable=False)
    date_min: Mapped[date] = mapped_column(Date, nullable=False)
    date_max: Mapped[date] = mapped_column(Date, nullable=False)
    loaded_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
