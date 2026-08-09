# STATUS: DEAD — Cross-Family Layer Stitching

The original Paper D method direction is closed.

## Why

- direct LLM stitching/recomposition prior art already occupies the method space;
- physically spliced models run but lose language ability;
- affine/oracle bridges still yield generation PPL in the hundreds or thousands;
- usable CKA magnitude was badly overstated when compared with random-init rather
  than the layer-order-shuffle null.

## Retracted claims

- depth mismatch matters more than family mismatch;
- OLMo-2 low CKA is caused by post-norm;
- random-init is the correct layer-correspondence null;
- affine pilot proves every nonlinear bridge is impossible.

## Surviving assets

Moved into A01/shared:

- 91-pair representation matrices;
- layer-order null;
- U-shape geometry measurement;
- affine readout-bandwidth pilot.

