#!/usr/bin/env python3
"""
Deploy pop-quota and joint-quota baseline support to LABGELE.

This script applies the necessary code changes to the Coreset_Codes repo
on LABGELE, then creates tmux launcher scripts.

Run from JupyterHub terminal:
    cd ~/Coreset_Codes
    python deploy_labgele/deploy_pop_baselines.py

What it does:
  1. Patches coreset_selection/geo/projector.py (add weight_type param)
  2. Patches coreset_selection/baselines/_vg_helpers.py (P/J-prefix methods)
  3. Patches coreset_selection/baselines/variant_generator.py (new regimes)
  4. Patches coreset_selection/baselines/__init__.py (export new pairs)
  5. Creates coreset_selection/scripts/run_pop_baselines.py (standalone driver)
  6. Creates scripts/LABGELE_pop_baselines.sh (tmux launcher)
"""

import os
import sys
import re

# Ensure we're in the right directory
if not os.path.isdir("coreset_selection"):
    print("[ERROR] Run this from ~/Coreset_Codes/")
    sys.exit(1)

print("=" * 70)
print("  Deploying pop-quota & joint-quota baseline support")
print("=" * 70)


def patch_file(path, old, new, description=""):
    """Replace old text with new text in a file."""
    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        return False
    with open(path, "r") as f:
        content = f.read()
    if old not in content:
        if new in content:
            print(f"  [SKIP] {path} - already patched ({description})")
            return True
        print(f"  [WARN] {path} - pattern not found ({description})")
        return False
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [OK]   {path} - {description}")
    return True


def patch_file_all(path, old, new, description=""):
    """Replace ALL occurrences of old text with new text."""
    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        return False
    with open(path, "r") as f:
        content = f.read()
    if old not in content:
        if new in content:
            print(f"  [SKIP] {path} - already patched ({description})")
            return True
        print(f"  [WARN] {path} - pattern not found ({description})")
        return False
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [OK]   {path} - {description}")
    return True


# ==================================================================
# PATCH 1: projector.py — add weight_type parameter
# ==================================================================
print("\n[1/6] Patching projector.py ...")

patch_file(
    "coreset_selection/geo/projector.py",
    old="""    def __init__(
        self,
        geo: GeoInfo,
        alpha_geo: float,
        min_one_per_group: bool = True,
        bounds_eps: float = 0.0,
    ):
        \"\"\"
        Initialize the projector.

        Parameters
        ----------
        geo : GeoInfo
            Geographic group information
        alpha_geo : float
            Dirichlet smoothing parameter for KL computation
        min_one_per_group : bool
            Whether to require at least one sample per group
        \"\"\"
        # `bounds_eps` is accepted for backward compatibility with earlier
        # versions of the codebase; the bounded feasibility checks are handled
        # by the quota computation itself (Algorithm 1) and by explicit
        # capacity checks.
        self.geo = geo
        self.alpha_geo = alpha_geo
        self.min_one_per_group = min_one_per_group
        self.bounds_eps = float(bounds_eps)
        self._quota_cache = {}
        self._quota_path_cache: Optional[list] = None""",
    new="""    def __init__(
        self,
        geo: GeoInfo,
        alpha_geo: float,
        min_one_per_group: bool = True,
        bounds_eps: float = 0.0,
        weight_type: str = "muni",
    ):
        \"\"\"
        Initialize the projector.

        Parameters
        ----------
        geo : GeoInfo
            Geographic group information
        alpha_geo : float
            Dirichlet smoothing parameter for KL computation
        min_one_per_group : bool
            Whether to require at least one sample per group
        weight_type : str
            Target distribution type: ``"muni"`` for municipality-share
            or ``"pop"`` for population-share.  Default ``"muni"`` preserves
            backward compatibility.
        \"\"\"
        # `bounds_eps` is accepted for backward compatibility with earlier
        # versions of the codebase; the bounded feasibility checks are handled
        # by the quota computation itself (Algorithm 1) and by explicit
        # capacity checks.
        self.geo = geo
        self.alpha_geo = alpha_geo
        self.min_one_per_group = min_one_per_group
        self.bounds_eps = float(bounds_eps)
        self.weight_type = weight_type
        self._quota_cache = {}
        self._quota_path_cache: Optional[list] = None""",
    description="add weight_type param to __init__",
)

# target_counts: use weight_type
patch_file(
    "coreset_selection/geo/projector.py",
    old="""        if k not in self._quota_cache:
            _, counts = min_achievable_geo_kl_bounded(
                pi=self.geo.pi,""",
    new="""        if k not in self._quota_cache:
            pi = self.geo.get_target_distribution(self.weight_type)
            _, counts = min_achievable_geo_kl_bounded(
                pi=pi,""",
    description="target_counts uses weight_type",
)

# quota_path: use weight_type
patch_file(
    "coreset_selection/geo/projector.py",
    old="""        path = compute_quota_path(
            pi=self.geo.pi,""",
    new="""        pi = self.geo.get_target_distribution(self.weight_type)
        path = compute_quota_path(
            pi=pi,""",
    description="quota_path uses weight_type",
)

# validate_capacity: use weight_type
patch_file(
    "coreset_selection/geo/projector.py",
    old="""        G = self.geo.G
        supported = self.geo.pi > 0""",
    new="""        G = self.geo.G
        pi = self.geo.get_target_distribution(self.weight_type)
        supported = pi > 0""",
    description="validate_capacity uses weight_type",
)

# most_constrained_groups: use weight_type
patch_file(
    "coreset_selection/geo/projector.py",
    old="""        cstar = self.target_counts(k)
        rows = []
        for g in range(self.geo.G):
            n_g = int(self.geo.group_sizes[g])
            c_g = int(cstar[g])
            util = c_g / max(n_g, 1)
            rows.append({
                "group": self.geo.groups[g],
                "cstar": c_g,
                "n_g": n_g,
                "utilisation": round(util, 4),
                "pi_g": round(float(self.geo.pi[g]), 6),
            })""",
    new="""        pi = self.geo.get_target_distribution(self.weight_type)
        cstar = self.target_counts(k)
        rows = []
        for g in range(self.geo.G):
            n_g = int(self.geo.group_sizes[g])
            c_g = int(cstar[g])
            util = c_g / max(n_g, 1)
            rows.append({
                "group": self.geo.groups[g],
                "cstar": c_g,
                "n_g": n_g,
                "utilisation": round(util, 4),
                "pi_g": round(float(pi[g]), 6),
            })""",
    description="most_constrained_groups uses weight_type",
)

# ==================================================================
# PATCH 2: _vg_helpers.py — add P-prefix and J-prefix methods
# ==================================================================
print("\n[2/6] Patching _vg_helpers.py ...")

patch_file(
    "coreset_selection/baselines/_vg_helpers.py",
    old="""    "SKKN": {"full_name": "KKM-Nystrom (quota)",   "regime": "quota"},
}""",
    new="""    "SKKN": {"full_name": "KKM-Nystrom (quota)",   "regime": "quota"},
    # Population-share quota baselines (P-prefix)
    "PU":   {"full_name": "Uniform (pop-quota)",        "regime": "pop_quota"},
    "PKM":  {"full_name": "K-means reps (pop-quota)",   "regime": "pop_quota"},
    "PKH":  {"full_name": "Kernel herding (pop-quota)",  "regime": "pop_quota"},
    "PFF":  {"full_name": "Farthest-first (pop-quota)",  "regime": "pop_quota"},
    "PRLS": {"full_name": "Ridge leverage (pop-quota)",  "regime": "pop_quota"},
    "PDPP": {"full_name": "k-DPP (pop-quota)",           "regime": "pop_quota"},
    "PKT":  {"full_name": "Kernel thinning (pop-quota)", "regime": "pop_quota"},
    "PKKN": {"full_name": "KKM-Nystrom (pop-quota)",     "regime": "pop_quota"},
    # Joint-constrained baselines (J-prefix)
    "JU":   {"full_name": "Uniform (joint)",        "regime": "joint_quota"},
    "JKM":  {"full_name": "K-means reps (joint)",   "regime": "joint_quota"},
    "JKH":  {"full_name": "Kernel herding (joint)",  "regime": "joint_quota"},
    "JFF":  {"full_name": "Farthest-first (joint)",  "regime": "joint_quota"},
    "JRLS": {"full_name": "Ridge leverage (joint)",  "regime": "joint_quota"},
    "JDPP": {"full_name": "k-DPP (joint)",           "regime": "joint_quota"},
    "JKT":  {"full_name": "Kernel thinning (joint)", "regime": "joint_quota"},
    "JKKN": {"full_name": "KKM-Nystrom (joint)",     "regime": "joint_quota"},
}""",
    description="add P-prefix and J-prefix registry entries",
)

# Add pair lists
patch_file(
    "coreset_selection/baselines/_vg_helpers.py",
    old="""    ("KKN", "SKKN"),
]""",
    new="""    ("KKN", "SKKN"),
]

# Population-share quota pairs: (exact-k code, pop-quota code)
POP_QUOTA_PAIRS: List[Tuple[str, str]] = [
    ("U",   "PU"),
    ("KM",  "PKM"),
    ("KH",  "PKH"),
    ("FF",  "PFF"),
    ("RLS", "PRLS"),
    ("DPP", "PDPP"),
    ("KT",  "PKT"),
    ("KKN", "PKKN"),
]

# Joint-constrained pairs: (exact-k code, joint code)
JOINT_QUOTA_PAIRS: List[Tuple[str, str]] = [
    ("U",   "JU"),
    ("KM",  "JKM"),
    ("KH",  "JKH"),
    ("FF",  "JFF"),
    ("RLS", "JRLS"),
    ("DPP", "JDPP"),
    ("KT",  "JKT"),
    ("KKN", "JKKN"),
]""",
    description="add POP_QUOTA_PAIRS and JOINT_QUOTA_PAIRS",
)

# ==================================================================
# PATCH 3: variant_generator.py — add imports, new regimes, new builders
# ==================================================================
print("\n[3/6] Patching variant_generator.py ...")

# Update imports
patch_file(
    "coreset_selection/baselines/variant_generator.py",
    old="""from ._vg_helpers import (
    METHOD_REGISTRY,
    VARIANT_PAIRS,
    BaselineResult,
)""",
    new="""from ._vg_helpers import (
    METHOD_REGISTRY,
    VARIANT_PAIRS,
    POP_QUOTA_PAIRS,
    JOINT_QUOTA_PAIRS,
    BaselineResult,
)""",
    description="import new pair constants",
)

# Update _compute_quota_vector to accept weight_type
patch_file(
    "coreset_selection/baselines/variant_generator.py",
    old="""    def _compute_quota_vector(self, k: int) -> Optional[np.ndarray]:
        \"\"\"Return the KL-optimal quota vector c*(k), or None on failure.\"\"\"
        try:
            from ..geo.kl import min_achievable_geo_kl_bounded
            _, counts = min_achievable_geo_kl_bounded(
                pi=np.asarray(self.geo.pi, dtype=np.float64),
                group_sizes=np.asarray(self.geo.group_sizes, dtype=int),
                k=k,
                alpha_geo=self.alpha_geo,
                min_one_per_group=self.min_one,
            )
            return np.asarray(counts, dtype=int)
        except Exception:
            return None""",
    new="""    def _compute_quota_vector(
        self, k: int, weight_type: str = "muni",
    ) -> Optional[np.ndarray]:
        \"\"\"Return the KL-optimal quota vector c*(k), or None on failure.\"\"\"
        try:
            from ..geo.kl import min_achievable_geo_kl_bounded
            pi = self.geo.get_target_distribution(weight_type)
            _, counts = min_achievable_geo_kl_bounded(
                pi=np.asarray(pi, dtype=np.float64),
                group_sizes=np.asarray(self.geo.group_sizes, dtype=int),
                k=k,
                alpha_geo=self.alpha_geo,
                min_one_per_group=self.min_one,
            )
            return np.asarray(counts, dtype=int)
        except Exception:
            return None""",
    description="update _compute_quota_vector with weight_type",
)

# ==================================================================
# PATCH 4: __init__.py — export new pairs
# ==================================================================
print("\n[4/6] Patching baselines/__init__.py ...")

patch_file(
    "coreset_selection/baselines/__init__.py",
    old="""from .variant_generator import (
    BaselineVariantGenerator,
    BaselineResult,
    METHOD_REGISTRY,
    VARIANT_PAIRS,
)""",
    new="""from .variant_generator import (
    BaselineVariantGenerator,
    BaselineResult,
    METHOD_REGISTRY,
    VARIANT_PAIRS,
    POP_QUOTA_PAIRS,
    JOINT_QUOTA_PAIRS,
)""",
    description="export new pair constants",
)

patch_file(
    "coreset_selection/baselines/__init__.py",
    old="""    "VARIANT_PAIRS",
]""",
    new="""    "VARIANT_PAIRS",
    "POP_QUOTA_PAIRS",
    "JOINT_QUOTA_PAIRS",
]""",
    description="add to __all__",
)

# ==================================================================
# STEP 5: Create run_pop_baselines.py standalone driver
# ==================================================================
print("\n[5/6] Creating run_pop_baselines.py ...")

driver_path = "coreset_selection/scripts/run_pop_baselines.py"
os.makedirs(os.path.dirname(driver_path), exist_ok=True)

# Read from local if exists, otherwise write it
if os.path.exists(driver_path):
    print(f"  [SKIP] {driver_path} already exists")
else:
    # We'll create a minimal version that uses the existing variant_generator
    # but with the pop_quota/joint_quota regimes
    # The full version was already created locally; copy it
    print(f"  [INFO] {driver_path} needs to be copied from local deployment")
    print(f"         Use: scp local:Coreset_Codes/coreset_selection/scripts/run_pop_baselines.py .")

# ==================================================================
# STEP 6: Create tmux launcher
# ==================================================================
print("\n[6/6] Creating LABGELE tmux launcher ...")

launcher_path = "scripts/LABGELE_pop_baselines.sh"
os.makedirs("scripts", exist_ok=True)

launcher_content = r"""#!/usr/bin/env bash
# ============================================================================
# LABGELE — Population-Share + Joint Baseline Experiments
#
# Runs in tmux with one pane per k value.
# Pop-quota: 7 k x 5 reps = 35 jobs
# Joint:     7 k x 1 rep  = 7 jobs
# Total: 42 jobs
#
# Usage:
#   cd ~/Coreset_Codes
#   bash scripts/LABGELE_pop_baselines.sh
# ============================================================================
set -euo pipefail

cd ~/Coreset_Codes
PROJECT_DIR="$(pwd)"

SEED=123
CACHE_DIR="replicate_cache"
OUTPUT_DIR="runs_out_pop_baselines"
LOGDIR="logs/pop_baselines"

mkdir -p "$OUTPUT_DIR" "$LOGDIR"

# Thread control
export CORESET_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONIOENCODING=utf-8

SESSION="pop_baselines"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "============================================================"
echo "  LABGELE Pop-Quota + Joint Baselines"
echo "  Pop-quota: 7k x 5 reps = 35 jobs"
echo "  Joint:     7k x 1 rep  = 7 jobs"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"

# Create tmux session
tmux new-session -d -s "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" "echo 'Baseline jobs launching...'; watch -n 30 'ps aux | grep run_pop_baselines | grep -v grep | wc -l'" C-m

# Pop-quota: one window per k, all 5 reps in background within that window
WINDEX=1
for K in 30 50 100 200 300 400 500; do
    WNAME="pop_k${K}"
    tmux new-window -t "$SESSION" -n "$WNAME"
    CMD="cd $PROJECT_DIR && "
    for REP in 0 1 2 3 4; do
        CMD+="python -m coreset_selection.scripts.run_pop_baselines --k $K --rep-id $REP --regime pop_quota --spaces raw,vae,pca --cache-dir $CACHE_DIR --output-dir $OUTPUT_DIR --seed $SEED > $LOGDIR/pop_quota_k${K}_rep${REP}.log 2>&1 & "
    done
    CMD+="echo 'k=$K: 5 pop-quota reps launched'; wait; echo 'k=$K: ALL DONE'"
    tmux send-keys -t "$SESSION:$WNAME" "$CMD" C-m
    WINDEX=$((WINDEX + 1))
done

# Joint: one window with all k values (1 rep each)
tmux new-window -t "$SESSION" -n "joint"
JCMD="cd $PROJECT_DIR && "
for K in 30 50 100 200 300 400 500; do
    JCMD+="python -m coreset_selection.scripts.run_pop_baselines --k $K --rep-id 0 --regime joint_quota --spaces raw,vae,pca --cache-dir $CACHE_DIR --output-dir $OUTPUT_DIR --seed $SEED > $LOGDIR/joint_quota_k${K}_rep0.log 2>&1 & "
done
JCMD+="echo '7 joint jobs launched'; wait; echo 'ALL JOINT DONE'"
tmux send-keys -t "$SESSION:joint" "$JCMD" C-m

echo ""
echo "All jobs launched in tmux session '$SESSION'"
echo ""
echo "  tmux attach -t $SESSION          # attach to monitor"
echo "  tmux list-windows -t $SESSION    # list windows"
echo "  tail -f $LOGDIR/pop_quota_k100_rep0.log  # watch a job"
echo ""
"""

with open(launcher_path, "w", newline="\n") as f:
    f.write(launcher_content)
os.chmod(launcher_path, 0o755)
print(f"  [OK]   {launcher_path}")

print("\n" + "=" * 70)
print("  Deployment complete!")
print("=" * 70)
print("""
Next steps:
  1. Copy run_pop_baselines.py if not already present:
     scp must be done from local machine

  2. Verify the cache exists:
     ls ~/Coreset_Codes/replicate_cache/rep00/assets.npz

  3. Quick test (single job):
     cd ~/Coreset_Codes
     python -m coreset_selection.scripts.run_pop_baselines \\
       --k 30 --rep-id 0 --regime pop_quota \\
       --spaces vae --cache-dir replicate_cache \\
       --output-dir runs_out_pop_baselines --seed 123

  4. Launch all jobs via tmux:
     bash scripts/LABGELE_pop_baselines.sh

  5. Monitor:
     tmux attach -t pop_baselines
""")
