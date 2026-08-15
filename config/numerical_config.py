"""
Numerical reasoning configuration (Phase 4).

All settings are read from the root ``.env`` at import time.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Master switch for the numerical reasoning subsystem.
NUMERICAL_ENABLED = os.getenv("NUMERICAL_ENABLED", "true").strip().lower() in ("1", "true", "yes")

# Maximum number of self-repair attempts before returning a structured failure.
NUMERICAL_MAX_REPAIR_ATTEMPTS = int(os.getenv("NUMERICAL_MAX_REPAIR_ATTEMPTS", "2"))

# Hard timeout (ms) for sandboxed numerical execution.
NUMERICAL_EXEC_TIMEOUT_MS = int(os.getenv("NUMERICAL_EXEC_TIMEOUT_MS", "3000"))

# Verification tolerance (percentage points / absolute) for reported-vs-computed.
NUMERICAL_VERIFY_TOLERANCE = float(os.getenv("NUMERICAL_VERIFY_TOLERANCE", "0.05"))

# Decimal rounding applied to the final result.
NUMERICAL_PRECISION = int(os.getenv("NUMERICAL_PRECISION", "4"))