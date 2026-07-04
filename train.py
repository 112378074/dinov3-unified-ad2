"""E1+ trainer skeleton: per-cat unified training = proven Branch-B recipe + bank-distillation.
Not yet runnable end-to-end — E1 (walnuts, single cat) is the first gated experiment.
Plan: build teacher bank once (membank_derisk.build_bank), train trunk with the 5 losses,
eval head_D against the teacher (AUPRO/SegF1 >= 95% of teacher passes the E1 gate)."""
raise SystemExit('E1 not implemented yet — see docs/PROPOSAL.md experiment plan')
