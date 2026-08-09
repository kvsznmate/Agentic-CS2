"""Agentic-CS2 — behavioural-cloning FPS agent for Counter-Strike 2.

Two-feed architecture: a radar/probability panel drives navigation, a
first-person vision model drives detection + aim, an arbiter hands the mouse
to exactly one feed per frame. See CLAUDE.md and DECISIONS.md at the repo root.
"""

__version__ = "0.0.1"
