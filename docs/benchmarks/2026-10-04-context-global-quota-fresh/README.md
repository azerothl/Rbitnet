# Cooperative global context disk quota: fresh validation

Compiled source: `ce8e7d3b4ed153afe8c25cb3dde9a00e4f183327`.
`RBITNET_CONTEXT_DISK_GLOBAL_MB` opts into a cooperative quota across
compatibility namespaces. A root OS lock covers admission, checkpoint writing
and rename. Recognized state and temporary objects consume quota, including
objects held by active foreign models. Reclamation protects active foreign
owners and removes only derived objects from inactive namespaces or local
cold LRU entries. External RAM leases remain valid. Lock/I/O/admission refusal
keeps RAM capture and normal replay available and records a write failure.

346 workspace tests pass, one network test is ignored. Check, Clippy and release
CLI/proxy/runner pass. All 19 context-store tests pass, including real separate
process ownership: the parent preserves a live child's checkpoint, kills that
child, then reclaims its checkpoint with exact physical cap and RAM lease intact.

Two actual models, Llama and Qwen3.5, share a deliberately small global cap:
the active foreign checkpoint remains intact, the other model serves from RAM,
and after the holder stops the other model persists by reclaiming the inactive
checkpoint. Physical derived object bytes remain within cap; outputs match
fresh references exactly. Existing process restart, RAM eviction, corruption
replay, stop/SSE/disconnect and proxy two-model restart checks also pass.

The Native CUDA library is reused from the sealed combined-stack proof, after
checking every Native source identity. This is not a new Native compilation.
The initial directory-only fixture correction was made before any hardware run;
the original prepared attempt is retained locally, not relabelled a test failure.

## Limits

All writers sharing this root must enable the same cap. Legacy uncooperative
writers are outside this guarantee. Different registrations are refused;
change configuration offline or use a fresh root. Existing active data over a
new cap refuses admission, rather than deleting active foreign objects.
Unrelated user files are excluded from accounting and deletion.

Filesystem fixtures cover refused I/O and interrupted prepared temporary
objects. They do not reproduce physical ENOSPC or termination during each
individual write/rename phase. Abrupt process termination covers an idle holder
after a completed write. Non-Windows locking/reparse behavior remains untested.
This quota correctness run does not claim a tokens/s improvement.
