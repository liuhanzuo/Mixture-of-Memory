#!/usr/bin/env bash
# A01 gate-1 depth-curve GAP FILL: resolve the collapse threshold.
#
# The first pass found the transition is SHARP, not gradual, and located it only
# to within a wide bracket:
#   Qwen3-8B (36L):  keep 4..24 all pinned at letter ~0.2297 (modal ~100%, ties ~0)
#                    then keep 30 -> 0.7263 (+45.74pp). Transition somewhere in (24, 30].
#   Llama-3-8B (32L): keep 4..16 sub-floor, keep 20 -> 0.5758 (+30.69pp).
#                    Transition somewhere in (16, 20].
#   Llama-2-7B (32L): non-monotone hump -- keep16 +6.00, keep20 +3.65, keep24 +0.33,
#                    i.e. it partially recovers then DEGRADES again toward the floor
#                    as depth increases. Needs keep 28/30/31 to see whether it ever
#                    reaches the intact 0.4100.
#
# This driver fills the brackets so the threshold is pinned to a single layer.
# Reusing scripts/_a01_gate1_depth_curve.sh; only KEEPS differs.
set -u
echo "This file documents the gap-fill KEEPS per family; the launcher is inline in the heartbeat."
echo "Qwen3-8B  : KEEPS='25 26 27 28 29'   (bracket 24 -> 30)"
echo "Llama-3-8B: KEEPS='17 18 19 28 30'   (bracket 16 -> 20, plus upper end)"
echo "Llama-2-7B: KEEPS='28 30 31 10 14'   (upper end + fill the hump)"
