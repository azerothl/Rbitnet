//! Expert-placement policies only: router scores and selected IDs never change.
#[derive(Clone, Copy, Debug)]
pub(super) enum Policy {
    Lru,
    Lfu,
    LeastStale,
}
#[derive(Clone, Copy, Default)]
pub(super) struct Access {
    pub touched: u64,
    pub frequency: u64,
    pub pass: u64,
}
impl Policy {
    pub fn from_env() -> Self {
        match std::env::var("RBITNET_MOE_CACHE_POLICY").as_deref() {
            Ok("lfu") => Self::Lfu,
            Ok("least-stale") => Self::LeastStale,
            _ => Self::Lru,
        }
    }
    pub fn rank(self, key: (usize, usize), access: Access, pass: u64) -> (u64, u64, u64, u64) {
        match self {
            Self::Lru => (access.touched, key.0 as u64, key.1 as u64, 0),
            Self::Lfu => (access.frequency, access.touched, key.0 as u64, key.1 as u64),
            // SpecMD section 4.2: stale queue first, then current queue;
            // order each queue by layer/expert position. Leases are filtered
            // independently before ranking, including prefetched active slots.
            Self::LeastStale => (
                u64::from(access.pass == pass),
                key.0 as u64,
                key.1 as u64,
                access.touched,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporal_and_layer_policies_disagree_on_future_and_current_slots() {
        let old_future = Access {
            touched: 1,
            frequency: 9,
            pass: 4,
        };
        let old_left = Access {
            touched: 2,
            frequency: 1,
            pass: 4,
        };
        let current_left = Access {
            touched: 3,
            frequency: 1,
            pass: 5,
        };
        assert!(Policy::Lru.rank((8, 0), old_future, 5) < Policy::Lru.rank((1, 0), old_left, 5));
        assert!(Policy::Lfu.rank((1, 0), old_left, 5) < Policy::Lfu.rank((8, 0), old_future, 5));
        assert!(
            Policy::LeastStale.rank((1, 0), old_left, 5)
                < Policy::LeastStale.rank((8, 0), old_future, 5)
        );
        assert!(
            Policy::LeastStale.rank((8, 0), old_future, 5)
                < Policy::LeastStale.rank((1, 0), current_left, 5)
        );
        // Starting a new pass makes previously current entries stale again.
        assert!(
            Policy::LeastStale.rank((1, 0), current_left, 6)
                < Policy::LeastStale.rank((8, 0), old_future, 6)
        );
    }
}
