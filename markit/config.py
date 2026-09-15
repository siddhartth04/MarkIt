"""Tunable settings for MarkIt. Edit these, then restart."""

# --- detection -------------------------------------------------------------
DETECTION_PROMPT = "person."
FLORENCE_MODEL   = "microsoft/Florence-2-base"   # 'base' light / 'large' better+slower
SHOW_CAPTION     = True
CAPTION_EVERY    = 3      # run the scene caption once per N detection cycles

# --- camera ----------------------------------------------------------------
CAM_INDEX        = 0      # try 1 or 2 if the wrong camera opens

# --- server ----------------------------------------------------------------
HOST             = "127.0.0.1"
PORT             = 5000

# --- tracking --------------------------------------------------------------
# Feature points per box, and the floor below which a box counts as lost.
# Lower MIN_POINTS keeps boxes alive longer at the cost of more drift.
MAX_POINTS       = 24
MIN_POINTS       = 4
MAX_SCALE        = 4.0    # reject implausible box growth/shrink from bad flow
