"""
Seed-Stage Data Contract and Feature Schema
===========================================
Defines the point-in-time boundary for early-stage (Seed / Angel) investment evaluation.
Any feature requiring knowledge after the Seed financing close date is strictly excluded
to prevent look-ahead bias as identified in the survey paper.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class SeedStageFeatures:
    # 1. Company Identity & Demographics
    company_name: str
    founded_year: int
    country_code: str
    city: str
    market_category: str
    is_major_startup_hub: int  # 1 if SF/SV, NYC, Boston, London, Bangalore, etc.

    # 2. Timing & Velocity Signals (Point-in-Time)
    months_founding_to_seed: float  # Speed of getting initial institutional validation
    seed_year: int                  # For macroeconomic cohort splitting (e.g., pre-2015 vs 2017+)
    
    # 3. Seed Financing Signals (Point-in-Time)
    seed_funding_usd: float         # Capital raised strictly in Seed/Angel round
    num_seed_investors: int         # Investor syndicate size in seed round
    has_tier1_investor: int         # 1 if tier-1 accelerator/VC (e.g., YC, Techstars, Sequoia, a16z)
    
    # 4. Founding Team Human Capital (Corea 21-Checklist & Park et al.)
    num_founders: int               # Solo vs team of 2-3
    has_serial_founder: int         # 1 if founder had prior startup experience
    has_technical_founder: int      # 1 if founder has technical/engineering pedigree
    
    # 5. Multimodal Textual Description (Kaiser & Kuhn)
    short_pitch: str                # One-sentence or short business summary
    
    # 6. Ground-Truth Target Outcome
    # y = 1 if company achieved successful exit (IPO or acquired for valuation/growth)
    # y = 0 if company closed or remained stagnant without exit
    target_exit_success: Optional[int] = None

# Features that MUST NOT be present in Seed stage evaluation (Anti-Leakage Blacklist)
LEAKAGE_BLACKLIST: List[str] = [
    "total_funding_usd",        # Sums all future rounds (Series A, B, C...)
    "funding_rounds",           # Total count across lifetime leaks company longevity
    "last_funding_at",          # Occurs years into the future
    "series_a_funding_usd",     # Future information
    "series_b_funding_usd",     # Future information
    "post_money_valuation",     # Future valuation
    "current_employee_count",   # Future headcount
]
