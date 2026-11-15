"""Generator configuration: every assumption is a named, overridable parameter.

Nothing about the simulated pharmacy is hard-coded inside generation logic. If a
reviewer asks "where does the winter antibiotics peak come from?", the answer is
a value in this file (``CATEGORY_PROFILES``), not a magic number three call
frames deep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class CategoryProfile:
    """Demand shape and commercial characteristics of one medicine category."""

    base_daily_units: float
    """Mean units/day for a *typical* product in this category at scale 1.0."""

    weekday_factors: tuple[float, ...]
    """Multiplier for Mon..Sun. Clinic-driven categories dip at weekends."""

    seasonal_amplitude: float
    """Fractional amplitude of the annual cycle (0.0 = flat, 0.8 = strong)."""

    seasonal_peak_day: int
    """Day-of-year (1-365) at which the annual cycle peaks."""

    price_range: tuple[float, float]
    shelf_life_days: tuple[int, int]
    is_clinic_driven: bool = False


# Rationale for each profile is in docs/data_generation.md. Summary:
#  * Antibiotics peak in winter (respiratory season) and are clinic-driven.
#  * Antihistamines peak in spring (allergy season) and sell at weekends.
#  * Vitamins peak in January (new-year resolutions) and around festivals.
#  * Cardiovascular / diabetes medicines are chronic-use: flat, high volume.
CATEGORY_PROFILES: dict[str, CategoryProfile] = {
    "Antibiotics": CategoryProfile(
        base_daily_units=7.0,
        weekday_factors=(1.10, 1.12, 1.10, 1.08, 1.05, 0.85, 0.62),
        seasonal_amplitude=0.55,
        seasonal_peak_day=15,
        price_range=(45.0, 320.0),
        shelf_life_days=(365, 730),
        is_clinic_driven=True,
    ),
    "Analgesics": CategoryProfile(
        base_daily_units=13.0,
        weekday_factors=(1.02, 1.03, 1.02, 1.04, 1.08, 1.12, 1.10),
        seasonal_amplitude=0.18,
        seasonal_peak_day=200,
        price_range=(18.0, 140.0),
        shelf_life_days=(540, 900),
        is_clinic_driven=False,
    ),
    "Antihistamines": CategoryProfile(
        base_daily_units=6.0,
        weekday_factors=(0.95, 0.96, 0.98, 1.00, 1.08, 1.20, 1.18),
        seasonal_amplitude=0.72,
        seasonal_peak_day=105,
        price_range=(28.0, 210.0),
        shelf_life_days=(365, 730),
        is_clinic_driven=False,
    ),
    "Cardiovascular": CategoryProfile(
        base_daily_units=10.5,
        weekday_factors=(1.06, 1.06, 1.05, 1.05, 1.04, 0.92, 0.82),
        seasonal_amplitude=0.08,
        seasonal_peak_day=300,
        price_range=(60.0, 480.0),
        shelf_life_days=(540, 1080),
        is_clinic_driven=True,
    ),
    "Diabetes Care": CategoryProfile(
        base_daily_units=5.5,
        weekday_factors=(1.05, 1.05, 1.04, 1.04, 1.02, 0.94, 0.86),
        seasonal_amplitude=0.06,
        seasonal_peak_day=250,
        price_range=(120.0, 720.0),
        shelf_life_days=(365, 900),
        is_clinic_driven=True,
    ),
    "Dermatology": CategoryProfile(
        base_daily_units=4.2,
        weekday_factors=(0.98, 1.00, 1.00, 1.02, 1.06, 1.14, 1.12),
        seasonal_amplitude=0.45,
        seasonal_peak_day=190,
        price_range=(55.0, 380.0),
        shelf_life_days=(365, 730),
        is_clinic_driven=False,
    ),
    "Gastrointestinal": CategoryProfile(
        base_daily_units=7.5,
        weekday_factors=(1.02, 1.02, 1.01, 1.01, 1.03, 1.08, 1.06),
        seasonal_amplitude=0.22,
        seasonal_peak_day=170,
        price_range=(32.0, 260.0),
        shelf_life_days=(365, 900),
        is_clinic_driven=False,
    ),
    "Vitamins & Supplements": CategoryProfile(
        base_daily_units=12.0,
        weekday_factors=(0.96, 0.97, 0.98, 1.00, 1.06, 1.15, 1.13),
        seasonal_amplitude=0.38,
        seasonal_peak_day=25,
        price_range=(75.0, 520.0),
        shelf_life_days=(540, 1080),
        is_clinic_driven=False,
    ),
}


@dataclass(frozen=True)
class Supplier:
    supplier_id: str
    name: str
    base_lead_time_days: int
    lead_time_jitter_days: int
    reliability: float


SUPPLIERS: tuple[Supplier, ...] = (
    Supplier("SUP-01", "Apex Pharma Distribution", 4, 2, 0.95),
    Supplier("SUP-02", "Northwind Medical Supplies", 7, 3, 0.90),
    Supplier("SUP-03", "Helix Healthcare Logistics", 10, 4, 0.86),
    Supplier("SUP-04", "Orion Pharma Wholesale", 6, 2, 0.93),
    Supplier("SUP-05", "Vertex Life Sciences", 12, 5, 0.80),
    Supplier("SUP-06", "BlueLeaf Pharma Traders", 5, 3, 0.91),
)

MANUFACTURERS: tuple[str, ...] = (
    "Apex Pharmaceuticals",
    "Northwind Labs",
    "Cipla Generics",
    "Helix Biotech",
    "Orion Remedies",
    "Sunrise Formulations",
    "Vertex Lifesciences",
    "BlueLeaf Pharma",
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Full parameterisation of the synthetic pharmacy."""

    seed: int = 42
    dataset_version: str = "ds-v1"
    n_products: int = 120

    # Calendar: a long history so that 180-day lookbacks exist at the first origin.
    start_date: str = "2024-05-01"
    end_date: str = "2025-12-31"

    # Origin grid for the modelling dataset.
    origin_start: str = "2025-01-01"
    origin_end: str = "2025-10-01"
    origin_step_days: int = 14
    horizon_days: int = 30

    # Inventory/batch snapshot used for expiry-risk evaluation.
    # The realised window must be long enough to observe expiry outcomes.
    snapshot_date: str = "2025-10-01"

    # Demand behaviour
    intermittent_share: float = 0.18
    intermittent_active_prob_range: tuple[float, float] = (0.25, 0.60)
    promo_day_share: float = 0.030
    promo_uplift_range: tuple[float, float] = (1.5, 3.0)
    discount_on_promo_range: tuple[float, float] = (0.05, 0.25)
    discount_off_promo_prob: float = 0.04
    price_noise_sigma: float = 0.03

    # Inventory posture: short / normal / excess cover in days of demand
    cover_days_mixture: tuple[tuple[float, float, float], ...] = (
        (0.20, 5.0, 15.0),
        (0.50, 20.0, 45.0),
        (0.30, 60.0, 150.0),
    )
    safety_factor: float = 1.25

    # Batches
    max_batches_per_product: int = 4
    expiry_bands_days: tuple[tuple[float, float, float], ...] = (
        (0.30, 12.0, 90.0),
        (0.40, 90.0, 240.0),
        (0.30, 240.0, 520.0),
    )
    sold_fraction_range: tuple[float, float] = (0.0, 0.45)

    tags: dict[str, str] = field(default_factory=dict)

    @property
    def start(self) -> date:
        return date.fromisoformat(self.start_date)

    @property
    def end(self) -> date:
        return date.fromisoformat(self.end_date)

    @property
    def snapshot(self) -> date:
        return date.fromisoformat(self.snapshot_date)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tags"] = dict(self.tags)
        payload["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        return payload
