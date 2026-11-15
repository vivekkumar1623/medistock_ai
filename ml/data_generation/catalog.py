"""Product-name catalogue.

Names are composed from an ingredient, a strength and a dosage form so that 120
distinct, plausible products exist without shipping a wall of literals. They are
fictional: no name here refers to a real product licence or a real patient.
"""

from __future__ import annotations

INGREDIENTS: dict[str, tuple[str, ...]] = {
    "Antibiotics": (
        "Amoxicillin",
        "Azithromycin",
        "Cefixime",
        "Clarithromycin",
        "Doxycycline",
        "Levofloxacin",
        "Metronidazole",
        "Ceftriaxone",
    ),
    "Analgesics": (
        "Paracetamol",
        "Ibuprofen",
        "Diclofenac",
        "Aceclofenac",
        "Tramadol",
        "Naproxen",
        "Aspirin",
        "Ketorolac",
    ),
    "Antihistamines": (
        "Cetirizine",
        "Levocetirizine",
        "Loratadine",
        "Fexofenadine",
        "Chlorpheniramine",
        "Montelukast",
        "Hydroxyzine",
        "Bilastine",
    ),
    "Cardiovascular": (
        "Amlodipine",
        "Atorvastatin",
        "Metoprolol",
        "Telmisartan",
        "Ramipril",
        "Rosuvastatin",
        "Bisoprolol",
        "Losartan",
    ),
    "Diabetes Care": (
        "Metformin",
        "Glimepiride",
        "Sitagliptin",
        "Vildagliptin",
        "Insulin Glargine",
        "Dapagliflozin",
        "Pioglitazone",
        "Gliclazide",
    ),
    "Dermatology": (
        "Mupirocin",
        "Clotrimazole",
        "Betamethasone",
        "Calcipotriol",
        "Adapalene",
        "Ketoconazole",
        "Hydrocortisone",
        "Tacrolimus",
    ),
    "Gastrointestinal": (
        "Pantoprazole",
        "Omeprazole",
        "Domperidone",
        "Ondansetron",
        "Sucralfate",
        "Lactulose",
        "Mesalamine",
        "Rifaximin",
    ),
    "Vitamins & Supplements": (
        "Vitamin D3",
        "Vitamin B12",
        "Multivitamin",
        "Calcium Carbonate",
        "Ferrous Ascorbate",
        "Omega-3",
        "Zinc Sulphate",
        "Folic Acid",
    ),
}

STRENGTHS: tuple[str, ...] = (
    "5mg",
    "10mg",
    "20mg",
    "25mg",
    "40mg",
    "50mg",
    "75mg",
    "100mg",
    "125mg",
    "200mg",
    "250mg",
    "400mg",
    "500mg",
    "650mg",
    "850mg",
    "1000mg",
)

FORMS: tuple[str, ...] = ("Tablet", "Capsule", "Syrup", "Injection", "Cream", "Sachet")

# Forms that make clinical sense per category keep the catalogue believable.
FORMS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "Antibiotics": ("Capsule", "Tablet", "Syrup", "Injection"),
    "Analgesics": ("Tablet", "Capsule", "Syrup", "Injection"),
    "Antihistamines": ("Tablet", "Syrup", "Capsule"),
    "Cardiovascular": ("Tablet", "Capsule"),
    "Diabetes Care": ("Tablet", "Injection"),
    "Dermatology": ("Cream", "Tablet"),
    "Gastrointestinal": ("Tablet", "Capsule", "Sachet", "Syrup"),
    "Vitamins & Supplements": ("Tablet", "Capsule", "Syrup", "Sachet"),
}


def build_medicine_name(category: str, index_in_category: int) -> str:
    """Deterministically compose a unique product name.

    ``index_in_category`` is used with three different moduli so that repeated
    (ingredient, strength, form) triples cannot collide within a category for
    realistic category sizes (<= 64 products per category).
    """
    ingredients = INGREDIENTS[category]
    forms = FORMS_BY_CATEGORY[category]
    ingredient = ingredients[index_in_category % len(ingredients)]
    strength = STRENGTHS[(index_in_category * 3 + 1) % len(STRENGTHS)]
    form = forms[(index_in_category // len(ingredients)) % len(forms)]
    return f"{ingredient} {strength} {form}"
