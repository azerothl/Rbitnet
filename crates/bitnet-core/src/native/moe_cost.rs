//! Draft for #86: measured whole routed-FFN decisions, never changes routing.
//! One model/layer owns the estimate, so different shapes/quantizations are not mixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Execution {
    Cache,
    Cpu,
    Adaptive,
}
impl Execution {
    pub(super) fn from_env() -> Self {
        match std::env::var("RBITNET_MOE_EXECUTION").as_deref() {
            Ok("cpu") => Self::Cpu,
            Ok("adaptive") => Self::Adaptive,
            _ => Self::Cache,
        }
    }
}
#[derive(Default, Debug)]
struct Mean {
    value: f64,
    samples: u64,
}
impl Mean {
    fn observe(&mut self, value: f64) {
        if !value.is_finite() || value <= 0.0 {
            return;
        }
        self.value = if self.samples == 0 {
            value
        } else {
            self.value * 0.75 + value * 0.25
        };
        self.samples = self.samples.saturating_add(1);
    }
}
#[derive(Debug)]
pub(super) struct Cost {
    policy: Execution,
    cpu: Mean,
    gpu: Mean,
    copy_per_byte: Mean,
    decisions: u64,
}
impl Cost {
    pub(super) fn new(policy: Execution) -> Self {
        Self {
            policy,
            cpu: Mean::default(),
            gpu: Mean::default(),
            copy_per_byte: Mean::default(),
            decisions: 0,
        }
    }
    /// Cached bytes still in flight must contribute their expected remaining wait.
    /// Whole selected groups are admitted together; capacity refusal bypasses GPU.
    pub(super) fn choose_cpu(&mut self, can_fit: bool, missing_bytes: usize) -> bool {
        self.decisions = self.decisions.saturating_add(1);
        if !can_fit {
            return true;
        }
        match self.policy {
            Execution::Cpu => true,
            Execution::Cache => false,
            Execution::Adaptive => {
                // Bounded calibration and periodic probes avoid freezing on a
                // stale estimate when workload, thermal state or locality changes.
                if self.cpu.samples < 2 {
                    return true;
                }
                if self.gpu.samples < 2 || (missing_bytes > 0 && self.copy_per_byte.samples < 2) {
                    return false;
                }
                if self.decisions % 128 == 0 {
                    return true;
                }
                if self.decisions % 128 == 1 {
                    return false;
                }
                let estimated = self.gpu.value + missing_bytes as f64 * self.copy_per_byte.value;
                // Require a measurable margin before paying for weight movement.
                estimated > self.cpu.value * 0.9
            }
        }
    }
    pub(super) fn observe_cpu(&mut self, nanoseconds: u64) {
        self.cpu.observe(nanoseconds as f64);
    }
    /// Wall time includes cache lookup/input/table copies/kernels/output sync.
    /// Subtract only actual expert fill time, never an inferred theoretical rate.
    pub(super) fn observe_gpu(&mut self, total_ns: u64, upload_bytes: u64, upload_ns: u64) {
        // Inconsistent admission/total intervals cannot teach a fictional
        // one-nanosecond GPU base cost. Admission is measured locally.
        if total_ns == 0 || upload_ns > total_ns {
            return;
        }
        self.gpu
            .observe(total_ns.saturating_sub(upload_ns).max(1) as f64);
        if upload_bytes > 0 {
            self.copy_per_byte
                .observe(upload_ns as f64 / upload_bytes as f64);
        }
    }
    #[cfg(test)]
    pub(super) fn estimates(&self, missing_bytes: usize) -> (Option<f64>, Option<f64>) {
        (
            (self.cpu.samples > 0).then_some(self.cpu.value),
            (self.gpu.samples > 0 && (missing_bytes == 0 || self.copy_per_byte.samples > 0))
                .then_some(self.gpu.value + missing_bytes as f64 * self.copy_per_byte.value),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_copy_cost_can_reverse_a_gpu_preference_without_changing_ids() {
        let mut c = Cost::new(Execution::Adaptive);
        c.observe_cpu(1000);
        c.observe_cpu(1000);
        c.observe_gpu(2200, 1000, 2000);
        c.observe_gpu(2200, 1000, 2000);
        assert!(!c.choose_cpu(true, 0));
        assert!(c.choose_cpu(true, 1000));
        assert!(c.choose_cpu(false, 0));
        assert_eq!(c.estimates(1000), (Some(1000.0), Some(2200.0)));
    }
    #[test]
    fn fixed_policies_and_calibration_are_explicit_and_probes_are_bounded() {
        assert!(!Cost::new(Execution::Cache).choose_cpu(true, 999999));
        assert!(Cost::new(Execution::Cpu).choose_cpu(true, 0));
        let mut c = Cost::new(Execution::Adaptive);
        assert!(c.choose_cpu(true, 1000));
        c.observe_cpu(1000);
        assert!(c.choose_cpu(true, 1000));
        c.observe_cpu(1000);
        assert!(!c.choose_cpu(true, 1000));
        c.observe_gpu(2200, 1000, 2000);
        assert!(!c.choose_cpu(true, 1000));
        c.observe_gpu(2200, 1000, 2000);
        assert!(c.choose_cpu(true, 1000));
        c.decisions = 127;
        assert!(c.choose_cpu(true, 0));
        assert!(!c.choose_cpu(true, 1000));
    }
    #[test]
    fn invalid_samples_and_clock_noise_cannot_create_negative_costs() {
        let mut c = Cost::new(Execution::Adaptive);
        c.observe_cpu(0);
        assert_eq!(c.estimates(0), (None, None));
        c.observe_gpu(50, 100, 100);
        assert_eq!(c.estimates(100), (None, None));
    }
}
