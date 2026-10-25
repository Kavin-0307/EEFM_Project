"""
Seed-Stage Data Contract and Feature Schema
===========================================
Defines the point-in-time schema for early-stage (Seed / Angel) investment evaluation
as implemented on the public Crunchbase investments dataset.

Any feature requiring knowledge after the early financing close date is strictly
excluded to prevent look-ahead bias as identified in the research survey.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class CrunchbaseSeedFeatures:
    """
    Features extracted from public Crunchbase investments data (investments_VC.csv).
    All signals are strictly point-in-time (known at Seed/Angel funding date).
    """
    # 1. Company Identity & Timing (Point-in-Time)
    name: str
    first_funding_year: int
    followup_years: float
    is_mature: int                  # 1 if >= 3 years observation window before snapshot cutoff
    months_to_first_funding: float  # Months from founding to first financing check
    founding_date_missing: int      # 1 if founding date was unrecorded (informative missingness)

    # 2. Seed-Stage Financing (Point-in-Time)
    seed_usd: float                 # Capital raised in Seed round
    angel_usd: float                # Capital raised in Angel round
    early_funding_usd: float        # Total early capital (seed + angel)
    log_seed_usd: float             # log1p(seed_usd)
    log_angel_usd: float            # log1p(angel_usd)
    log_early_funding_usd: float    # log1p(early_funding_usd)
    has_seed: int                   # 1 if seed_usd > 0
    has_angel: int                  # 1 if angel_usd > 0
    angel_share: float              # angel_usd / early_funding_usd (bounded in [0, 1])

    # 3. Geography & Venture Ecosystem (Point-in-Time)
    is_top_hub: int                 # 1 if located in top venture hub (SF, NY, Boston, London, BLR, etc.)
    country_tier: str               # Top-tier venture country (USA, GBR, CAN, IND, DEU, etc.) or OTHER
    has_homepage: int               # 1 if verified web domain is documented

    # 4. Clean Multi-Label Category Taxonomy (Point-in-Time)
    macro_sector: str               # Primary sector (Enterprise SaaS, Consumer, Healthcare, Fintech, etc.)
    is_b2b_saas: int                # Binary domain flag
    is_ai_analytics: int            # Binary domain flag
    is_biotech_health: int          # Binary domain flag
    is_fintech: int                 # Binary domain flag
    is_consumer_mobile: int         # Binary domain flag
    is_ecommerce: int               # Binary domain flag
    num_categories: int             # Number of tags listed in Crunchbase category taxonomy

    # 5. Ground-Truth Target Outcomes (Used for Training & Evaluation Only)
    target_series_a: int            # 1 if company raised Series A; 0 otherwise
    target_followon: int            # 1 if company raised Series A or Venture follow-on; 0 otherwise
    is_exit: Optional[float] = None # 1 if Acquired/IPO; 0 if Closed (defined only on resolved terminal states)


# Features that MUST NOT be present during Seed-stage evaluation (Anti-Leakage Blacklist)
LEAKAGE_BLACKLIST: List[str] = [
    "total_funding_usd",        # Lifetime funding across all future rounds
    "funding_rounds",           # Total round count across lifetime leaks company longevity
    "last_funding_at",          # Date of final round occurs years into future
    "round_a_usd",              # Target outcome (future milestone)
    "round_b_usd",              # Future information
    "round_c_usd",              # Future information
    "venture_usd",              # Future venture checks
    "status",                   # Contains future operating/exit state
]
