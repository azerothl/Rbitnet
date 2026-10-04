# Experimental adaptive top-p selection

`RBITNET_CPU_TOP_P_HEAP=1` selects an experimental sorted-prefix sampler.
The default sampler is unchanged. The candidate preserves vocabulary-order
F32 totals, descending-weight/ascending-ID ties, retained-prefix F32 additions
and the original RNG draw. Already sorted inputs skip heap construction;
wide nuclei use stable sorting; small candidate nuclei try 64 heap pops before
a complete deterministic sort fallback. Temperature and repetition penalties
are applied before this selection, as before.

Fresh validation passed: 298 workspace tests (one ignored), 2,028 synthetic
cases, 864 actual-vector token/RNG checks and 96 four-model CPU/CUDA HTTP/SSE
observations. The optional path improved measured GPU wall rates on the two
writing prompts per model. Wider-distribution selection can still regress;
the default remains unchanged. See the complete [report and original captures](benchmarks/2026-10-04-adaptive-nucleus-fresh/README.md).
