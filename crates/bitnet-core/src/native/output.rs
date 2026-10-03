//! Turn markers and channel headers are protocol syntax, not answer text.
pub(crate) fn gpt_oss_content(raw: &str) -> &str {
    if let Some((_, final_part)) = raw.rsplit_once("<|channel|>final<|message|>") {
        return final_part;
    }
    if raw.starts_with("<|channel|>analysis") || raw.starts_with("<|channel|") {
        return "";
    }
    raw
}

#[cfg(test)]
mod tests {
    #[test]
    fn harmony_analysis_is_not_a_finished_answer() {
        assert_eq!(
            super::gpt_oss_content("<|channel|>analysis<|message|>Need answer: Paris.<|end|>"),
            ""
        );
        assert_eq!(super::gpt_oss_content("<|channel|>analysis<|message|>Need answer: Paris.<|end|><|start|>assistant<|channel|>final<|message|>Paris"), "Paris");
        assert_eq!(super::gpt_oss_content("Paris"), "Paris");
        assert_eq!(super::gpt_oss_content("<|channel|>final<|message|>"), "");
    }
}
