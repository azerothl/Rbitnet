//! Allocated capacity is separate from the model's training context metadata.
use crate::error::{BitNetError, Result};

pub(crate) fn resolve_capacity(
    training: usize,
    default: usize,
    hard: Option<usize>,
    requested: Option<&str>,
) -> Result<usize> {
    if training == 0 || default == 0 || hard == Some(0) {
        return Err(BitNetError::InvalidGguf(
            "context capacity must be positive".into(),
        ));
    }
    let requested = match requested {
        Some(value) => value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                BitNetError::Inference("RBITNET_MAX_SEQ must be a positive integer".into())
            })?,
        None => default,
    };
    Ok(training.min(requested).min(hard.unwrap_or(usize::MAX)))
}

pub(crate) fn capacity_from_env(
    training: usize,
    default: usize,
    hard: Option<usize>,
) -> Result<usize> {
    let requested = std::env::var("RBITNET_MAX_SEQ")
        .map(Some)
        .or_else(|e| match e {
            std::env::VarError::NotPresent => Ok(None),
            std::env::VarError::NotUnicode(_) => Err(BitNetError::Inference(
                "RBITNET_MAX_SEQ is not valid Unicode".into(),
            )),
        })?;
    resolve_capacity(training, default, hard, requested.as_deref())
}

pub(crate) fn check_request(prompt: usize, output: u32, capacity: usize) -> Result<()> {
    // Reserve the requested upper bound, even if the model might emit EOS early.
    if prompt
        .checked_add(output as usize)
        .is_none_or(|n| n > capacity)
    {
        return Err(BitNetError::ContextCapacityExceeded {
            prompt_tokens: prompt,
            max_tokens: output,
            capacity,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_caps_distinguish_training_request_and_family_limit() {
        assert_eq!(
            resolve_capacity(202752, 8192, None, Some("2048")).unwrap(),
            2048
        );
        assert_eq!(
            resolve_capacity(32768, 2048, Some(8192), Some("16384")).unwrap(),
            8192
        );
        assert_eq!(
            resolve_capacity(512, 8192, None, Some("2048")).unwrap(),
            512
        );
        for bad in ["", "0", "-1", "no", "184467440737095516160"] {
            assert!(resolve_capacity(8192, 2048, None, Some(bad)).is_err());
        }
        assert!(resolve_capacity(0, 2048, None, None).is_err());
    }
    #[test]
    fn reserve_complete_request_without_overflow_and_use_client_error() {
        check_request(2047, 1, 2048).unwrap();
        for (p, n, c) in [
            (2048, 1, 2048),
            (2049, 0, 2048),
            (usize::MAX, 1, usize::MAX),
        ] {
            let e = check_request(p, n, c).unwrap_err();
            assert_eq!(e.http_status_for_chat_completion(), 400);
            assert!(
                matches!(e,BitNetError::ContextCapacityExceeded {prompt_tokens,..} if prompt_tokens==p)
            );
        }
    }
}
