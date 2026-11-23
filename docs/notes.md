# Research notes

## Parked

### 3. Dynamic number of experts per token (parked)
- Let the router pick k per token instead of a fixed top-6 (threshold routing on sigmoid scores, with min/max clamps).
- Options compared: threshold routing, top-p routing, null-expert / learned-k routing.
- Recommended first try: threshold routing, calibrated so average k on coding text ≈ 6 before training.
- Run only after Q1 (fixed top-6, one MoE layer) has a result, so the two effects don't confound.
