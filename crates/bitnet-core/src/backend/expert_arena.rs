//! Fixed disjoint expert slots backed by one managed CUDA allocation.
//! Padding belongs to the physical budget; views retain the allocation owner.
use super::*;
use crate::error::{BitNetError, Result};

const ALIGNMENT: usize = 256;

#[derive(Debug)]
pub(crate) struct Layout {
    spans: [usize; 3],
    group_bytes: usize,
    groups: usize,
    total_bytes: usize,
}

impl Layout {
    pub(crate) fn new(spans: [usize; 3], budget: usize) -> Result<Self> {
        let fail = || BitNetError::Inference("invalid or overflowing expert arena layout".into());
        let mut group_bytes = 0usize;
        for span in spans {
            if span == 0 {
                return Err(fail());
            }
            let padded = span.checked_add(ALIGNMENT - 1).ok_or_else(fail)? & !(ALIGNMENT - 1);
            group_bytes = group_bytes.checked_add(padded).ok_or_else(fail)?;
        }
        let groups = budget / group_bytes;
        if groups == 0 {
            return Err(BitNetError::Inference(
                "expert arena budget cannot hold one group".into(),
            ));
        }
        let total_bytes = groups.checked_mul(group_bytes).ok_or_else(fail)?;
        Ok(Self {
            spans,
            group_bytes,
            groups,
            total_bytes,
        })
    }

    fn offsets(&self, group: usize) -> [usize; 3] {
        assert!(group < self.groups);
        let mut offset = group * self.group_bytes;
        self.spans.map(|span| {
            let current = offset;
            offset += (span + ALIGNMENT - 1) & !(ALIGNMENT - 1);
            current
        })
    }
    pub(crate) fn group_bytes(&self) -> usize {
        self.group_bytes
    }
    pub(crate) fn groups(&self) -> usize {
        self.groups
    }
}

/// Consume a validated layout once. No API can issue overlapping mutable views.
pub(crate) fn allocate(
    rt: &Arc<CudaRuntime>,
    layout: &Layout,
) -> Result<(Arc<CudaDeviceBuffer>, Vec<[CudaDeviceBuffer; 3]>)> {
    let ptr = rt
        .alloc_device_category(layout.total_bytes, device_memory::EXPERTS)
        .ok_or_else(|| {
            BitNetError::Inference("managed CUDA budget cannot allocate expert arena".into())
        })?;
    let owner = Arc::new(CudaDeviceBuffer {
        rt: Arc::clone(rt),
        ptr: ptr as usize,
        nbytes: layout.total_bytes,
        owner: None,
    });
    // Build only disjoint views, without issuing copies or publishing matrices.
    let views = (0..layout.groups)
        .map(|group| {
            let offsets = layout.offsets(group);
            std::array::from_fn(|projection| CudaDeviceBuffer {
                rt: Arc::clone(rt),
                ptr: owner.ptr + offsets[projection],
                nbytes: layout.spans[projection],
                owner: Some(Arc::clone(&owner)),
            })
        })
        .collect();
    Ok((owner, views))
}

pub(crate) fn configured(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err(BitNetError::Inference(
            "RBITNET_MOE_ARENA must be 0 or 1".into(),
        )),
    }
}
pub(crate) fn from_env() -> Result<bool> {
    match std::env::var("RBITNET_MOE_ARENA") {
        Ok(value) => configured(Some(&value)),
        Err(std::env::VarError::NotPresent) => configured(None),
        Err(_) => Err(BitNetError::Inference(
            "RBITNET_MOE_ARENA must be 0 or 1".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_and_disjoint_slots_are_charged_to_the_physical_budget() {
        let layout = Layout::new([257, 1023, 1024], 10000).unwrap();
        assert_eq!(
            (layout.group_bytes, layout.groups, layout.total_bytes),
            (2560, 3, 7680)
        );
        let mut previous_end = 0;
        for group in 0..layout.groups {
            for (offset, span) in layout.offsets(group).into_iter().zip(layout.spans) {
                assert_eq!(offset % ALIGNMENT, 0);
                assert!(offset >= previous_end);
                previous_end = offset + span;
                assert!(previous_end <= layout.total_bytes);
            }
        }
        assert!(layout.total_bytes <= 10000);
        assert!(Layout::new([257, 1023, 1024], 2559).is_err());
        assert!(Layout::new([0, 512, 1024], 10000).is_err());
        assert!(Layout::new([usize::MAX, 512, 1024], usize::MAX).is_err());
        assert!(Layout::new([usize::MAX / 2, usize::MAX / 2, 1024], usize::MAX).is_err());
        assert_eq!(configured(None).unwrap(), false);
        assert!(configured(Some("1")).unwrap());
        assert!(configured(Some("yes")).is_err());
    }

    #[test]
    fn optional_actual_expert_arena_views_keep_one_allocation_until_the_last_owner() {
        if std::env::var("RBITNET_EXPERT_ARENA_TEST").as_deref() != Ok("1") {
            return;
        }
        let rt = CudaRuntime::try_load().expect("real CUDA required");
        let before = rt
            .managed_memory_stats()
            .expect("managed memory API required");
        let layout = Layout::new([256, 512, 1024], 7168).unwrap();
        let (owner, mut views) = allocate(&rt, &layout).unwrap();
        drop(owner); // Each remaining disjoint view retains the base allocation.
        let live = rt.managed_memory_stats().unwrap();
        assert_eq!(live.allocations - before.allocations, 1);
        assert_eq!(
            live.categories[device_memory::EXPERTS as usize]
                - before.categories[device_memory::EXPERTS as usize],
            7168
        );
        for (group, matrices) in views.iter().enumerate() {
            for (projection, buffer) in matrices.iter().enumerate() {
                let payload = vec![((group * 3 + projection) + 1) as u8; buffer.nbytes];
                assert!(rt.copy_host_to_device(
                    buffer.as_device_ptr(),
                    payload.as_ptr().cast(),
                    payload.len()
                ));
            }
        }
        for (group, matrices) in views.iter().enumerate() {
            for (projection, buffer) in matrices.iter().enumerate() {
                let mut actual = vec![0u8; buffer.nbytes];
                assert!(rt.copy_device_to_host(
                    actual.as_mut_ptr().cast(),
                    buffer.as_device_ptr(),
                    actual.len()
                ));
                assert!(actual
                    .iter()
                    .all(|&x| x == ((group * 3 + projection) + 1) as u8));
            }
        }
        let last = views.pop().unwrap();
        drop(views);
        assert_eq!(
            rt.managed_memory_stats().unwrap().categories[device_memory::EXPERTS as usize],
            live.categories[device_memory::EXPERTS as usize]
        );
        drop(last);
        let after = rt.managed_memory_stats().unwrap();
        assert_eq!(after.live, before.live);
        assert_eq!(after.categories, before.categories);
        println!("EXPERT_ARENA_OWNER_DONE physical_allocations=1 groups=4 views=12");
    }
}
