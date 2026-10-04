# Executed Windows context-cache namespace guards

Original compiled commit: `267006e35e77882414fca4728b0e4bd4627421ca`. The publication branch subsequently
merged the documentation-only updates from #128; compiled Rust source bytes
remain unchanged and are verified against the index before publication.

The original implementation creates namespace files through a redirected format
directory. A negative control with the original implementation and the new test
reproduces that failure. The patched implementation rejects redirected format
and namespace directories before creating managed children, rejects redirected
lock files, and retains Windows directory/lock handles without delete sharing.
The Windows junction and rename fixtures run on this machine.

Check and Clippy pass; workspace tests: 338 passed,
1 ignored. Eleven context-store tests pass.
Fresh CLI/proxy/runner binaries execute real Llama and Qwen RAM eviction,
process restart, corruption replay and two-model proxy session/alias/idle reload
and proxy restart checks. The Native DLL is reused from #128, with its identity
and normalized Native source equivalence verified; no new Native build is claimed.

35 original compressed captures and exact orchestration helpers
are retained. Original model/binary/library/source identities are in the manifest;
executables and model files are not committed.

The configured parent directory remains user-selected. This is scoped managed
namespace hardening on Windows, not a guarantee against all parent-directory
races on every OS. Global old-namespace quotas, disk-full injection and crash
recovery sweeps remain separate work. No SSD-streamed model-weight support.
