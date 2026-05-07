//! Transformer graph (future) and toy LM for tests.

mod executor;
mod toy;

pub use executor::{BitNetExecutor, LlamaExecutor, ModelExecutor};
pub use toy::ToyLlm;
