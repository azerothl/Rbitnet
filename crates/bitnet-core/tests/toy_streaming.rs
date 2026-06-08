//! Toy path emits live stream deltas (not a single post-hoc chunk).

use bitnet_core::stream::StreamEvent;
use bitnet_core::sampling::SamplingOptions;
use bitnet_core::Engine;

#[test]
fn toy_complete_streaming_emits_deltas() {
    let prev_toy = std::env::var("RBITNET_TOY").ok();
    std::env::set_var("RBITNET_TOY", "1");
    std::env::remove_var("RBITNET_MODEL");
    std::env::remove_var("RBITNET_STUB");

    let engine = Engine::from_env().expect("engine");
    let mut deltas = 0u32;
    engine
        .complete_streaming("hello stream", 8, SamplingOptions::from_temperature(0.0), &mut |ev| {
            if matches!(ev, StreamEvent::Delta { .. }) {
                deltas += 1;
            }
            Ok(())
        })
        .expect("stream");

    assert!(deltas >= 1, "expected at least one StreamEvent::Delta, got {deltas}");

    match prev_toy {
        Some(v) => std::env::set_var("RBITNET_TOY", v),
        None => std::env::remove_var("RBITNET_TOY"),
    }
}
