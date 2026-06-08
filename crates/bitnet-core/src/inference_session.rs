//! Stateful generation sessions (continuous batching prerequisite).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::timings::PhaseTimings;

#[derive(Debug, Clone)]
pub struct InferenceSessionState {
    pub id: u64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub last_pos: usize,
    pub done: bool,
}

#[derive(Debug, Default)]
pub struct InferenceSessionStore {
    next_id: u64,
    sessions: HashMap<u64, InferenceSessionState>,
}

impl InferenceSessionStore {
    pub fn open(&mut self, prompt_tokens: u32) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.sessions.insert(
            id,
            InferenceSessionState {
                id,
                prompt_tokens,
                completion_tokens: 0,
                last_pos: prompt_tokens as usize,
                done: false,
            },
        );
        id
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut InferenceSessionState> {
        self.sessions.get_mut(&id)
    }

    pub fn close(&mut self, id: u64) {
        self.sessions.remove(&id);
    }

    pub fn record_decode_token(&mut self, id: u64, phases: &PhaseTimings) {
        if let Some(s) = self.sessions.get_mut(&id) {
            s.completion_tokens = s.completion_tokens.saturating_add(1);
            s.last_pos = s.last_pos.saturating_add(1);
            if phases.completion_tokens == 0 {
                s.done = true;
            }
        }
    }
}

static GLOBAL_SESSIONS: OnceLock<Mutex<InferenceSessionStore>> = OnceLock::new();

pub fn global_sessions() -> &'static Mutex<InferenceSessionStore> {
    GLOBAL_SESSIONS.get_or_init(|| Mutex::new(InferenceSessionStore::default()))
}

pub fn sessions_enabled() -> bool {
    matches!(
        std::env::var("RBITNET_SESSIONS").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    ) || matches!(
        std::env::var("RBITNET_CONTINUOUS_BATCHING").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}
