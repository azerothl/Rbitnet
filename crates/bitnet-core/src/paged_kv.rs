//! Paged KV cache MVP.

/// Minimal paged KV representation for scheduler/runtime integration.
#[derive(Debug, Clone)]
pub struct PagedKvCache {
    pub page_size_tokens: usize,
    pub max_pages: usize,
}

impl Default for PagedKvCache {
    fn default() -> Self {
        Self {
            page_size_tokens: 16,
            max_pages: 4096,
        }
    }
}

impl PagedKvCache {
    pub fn from_env() -> Self {
        let page_size_tokens = std::env::var("RBITNET_PAGED_KV_PAGE_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(16);
        let max_pages = std::env::var("RBITNET_PAGED_KV_MAX_PAGES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(4096);
        Self {
            page_size_tokens,
            max_pages,
        }
    }
}
