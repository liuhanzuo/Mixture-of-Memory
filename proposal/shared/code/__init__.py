"""Shared analysis code for multiple proposals.

Kept importable (rather than exec'd from source text, which is how
`canonical_eval_loaders` used to be consumed) so that a proposal's directory can
be archived without breaking the analyses of other proposals that depend on it.
"""
