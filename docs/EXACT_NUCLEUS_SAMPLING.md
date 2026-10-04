# Experimental adaptive top-p selection

`RBITNET_CPU_TOP_P_HEAP=1` selects an experimental sorted-prefix sampler.
The default sampler is unchanged. The candidate preserves vocabulary-order
F32 totals, descending-weight/ascending-ID ties, retained-prefix F32 additions
and the original RNG draw. Already sorted inputs skip heap construction;
wide nuclei use stable sorting; small candidate nuclei try 64 heap pops before
a complete deterministic sort fallback. Temperature and repetition penalties
are applied before this selection, as before.

This source is prepared, not yet validated. Earlier isolated GPT-OSS vector
measurements of the unguarded heap showed gains and uniform-input regressions;
they do not validate this revision or establish end-to-end tokens/s improvement.
Synthetic/actual-vector equality, RNG state, CPU ablations and four-model
HTTP/SSE comparisons are required before publishing or enabling this option.
