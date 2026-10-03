//! Draft #24: honor SentencePiece BOS/EOS and visible control decoding.
//! Ordinary text keeps the original encoder/normalizer. Explicit controls in
//! multi-turn templates use the same model with visible control token types.
use crate::error::{BitNetError, Result};
use crate::gguf::{GgufArchive, GgufValue};
use sentencepiece::SentencePieceProcessor;
use std::ops::Range;
use std::sync::OnceLock;

fn invalid(message: &str) -> BitNetError {
    BitNetError::Inference(format!("SentencePiece metadata/model: {message}"))
}
fn varint(data: &[u8], position: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *data
            .get(*position)
            .ok_or_else(|| invalid("truncated protobuf varint"))?;
        *position += 1;
        if shift == 63 && byte > 1 {
            return Err(invalid("protobuf varint overflow"));
        }
        value |= ((byte & 127) as u64) << shift;
        if byte & 128 == 0 {
            return Ok(value);
        }
    }
    Err(invalid("protobuf varint overflow"))
}
struct Field {
    number: u32,
    wire: u8,
    payload: Range<usize>,
}
fn field(data: &[u8], position: &mut usize, depth: usize) -> Result<Field> {
    if depth > 32 {
        return Err(invalid("protobuf nesting limit"));
    }
    let tag = varint(data, position)?;
    let number = u32::try_from(tag >> 3).map_err(|_| invalid("protobuf field overflow"))?;
    let wire = (tag & 7) as u8;
    if number == 0 || number > 0x1fffffff {
        return Err(invalid("invalid protobuf field"));
    }
    let start = *position;
    let payload = match wire {
        0 => {
            varint(data, position)?;
            start..*position
        }
        1 | 5 => {
            *position = position
                .checked_add(if wire == 1 { 8 } else { 4 })
                .ok_or_else(|| invalid("protobuf offset overflow"))?;
            start..*position
        }
        2 => {
            let n = usize::try_from(varint(data, position)?)
                .map_err(|_| invalid("protobuf length overflow"))?;
            let start = *position;
            *position = position
                .checked_add(n)
                .ok_or_else(|| invalid("protobuf offset overflow"))?;
            start..*position
        }
        3 => {
            loop {
                let begin = *position;
                let end = varint(data, position)?;
                if end & 7 == 4 {
                    if end >> 3 != number as u64 {
                        return Err(invalid("protobuf group mismatch"));
                    }
                    break;
                }
                *position = begin;
                field(data, position, depth + 1)?;
            }
            start..*position
        }
        _ => return Err(invalid("invalid protobuf wire type")),
    };
    if payload.end > data.len() {
        return Err(invalid("truncated protobuf field"));
    }
    Ok(Field {
        number,
        wire,
        payload,
    })
}
/// Preserve every original protobuf byte, including normalizer, denormalizer,
/// trainer settings and extensions. Only CONTROL enum values become visible
/// USER_DEFINED values for visible decoding and explicit control recognition.
fn visible_model(proto: &[u8]) -> Result<(Vec<u8>, Vec<String>, Vec<String>)> {
    if proto.len() > 128 * 1024 * 1024 {
        return Err(invalid("tokenizer model exceeds 128 MiB"));
    }
    let mut visible = proto.to_vec();
    let mut names = Vec::new();
    let mut controls = Vec::new();
    let mut position = 0;
    while position < proto.len() {
        let outer = field(proto, &mut position, 0)?;
        if outer.number != 1 || outer.wire != 2 {
            continue;
        }
        if names.len() >= 1_048_576 {
            return Err(invalid("tokenizer vocabulary exceeds limit"));
        }
        let piece = &proto[outer.payload.clone()];
        let mut offset = 0;
        let mut name = None;
        let mut control = false;
        while offset < piece.len() {
            let f = field(piece, &mut offset, 0)?;
            if f.number == 1 && f.wire == 2 {
                name = Some(
                    std::str::from_utf8(&piece[f.payload.clone()])
                        .map_err(|_| invalid("piece is not UTF-8"))?
                        .to_owned(),
                );
            }
            if f.number == 3 && f.wire == 0 {
                let mut at = f.payload.start;
                if varint(piece, &mut at)? == 3 {
                    control = true;
                    let at = outer.payload.start + f.payload.start;
                    visible[at] = (visible[at] & 128) | 4;
                }
            }
        }
        let name = name.ok_or_else(|| invalid("piece has no text"))?;
        if control && !name.is_empty() {
            controls.push(name.clone());
        }
        names.push(name);
    }
    if names.is_empty() {
        return Err(invalid("tokenizer has no pieces"));
    }
    Ok((visible, names, controls))
}
fn boolean(archive: Option<&GgufArchive>, key: &str, default: bool) -> Result<bool> {
    match archive.and_then(|a| a.metadata.get(key)) {
        None => Ok(default),
        Some(GgufValue::Bool(value)) => Ok(*value),
        Some(_) => Err(invalid(&format!("{key} must be a boolean"))),
    }
}
fn check_id(archive: Option<&GgufArchive>, key: &str, actual: Option<u32>) -> Result<()> {
    let Some(archive) = archive else {
        return Ok(());
    };
    if archive.metadata.contains_key(key) {
        let expected = archive
            .metadata_u64(key)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| invalid(&format!("{key} must be a token ID")))?;
        if actual != Some(expected) {
            return Err(invalid(&format!("{key} disagrees with tokenizer.model")));
        }
    }
    Ok(())
}
pub(crate) struct SentencePieceCodec {
    processor: SentencePieceProcessor,
    visible_proto: Vec<u8>,
    visible: OnceLock<std::result::Result<SentencePieceProcessor, String>>,
    controls: Vec<String>,
    bos: Option<(u32, String)>,
    eos: Option<(u32, String)>,
    add_bos: bool,
    add_eos: bool,
}
impl SentencePieceCodec {
    pub(super) fn load(path: &std::path::Path, archive: Option<&GgufArchive>) -> Result<Self> {
        let proto = std::fs::read(path)?;
        Self::from_proto(&proto, archive)
    }
    fn from_proto(proto: &[u8], archive: Option<&GgufArchive>) -> Result<Self> {
        if proto.len() > 128 * 1024 * 1024 {
            return Err(invalid("tokenizer model exceeds 128 MiB"));
        }
        let processor = SentencePieceProcessor::from_serialized_proto(proto)
            .map_err(|e| invalid(&e.to_string()))?;
        let (visible_proto, names, controls) = visible_model(proto)?;
        if let Some(archive) = archive {
            if let Some(tokens) = archive.metadata.get("tokenizer.ggml.tokens") {
                let GgufValue::Array(tokens) = tokens else {
                    return Err(invalid("tokenizer.ggml.tokens must be an array"));
                };
                if tokens.len() != names.len()
                    || tokens.iter().zip(&names).any(
                        |(token, name)| !matches!(token,GgufValue::String(value)if value==name),
                    )
                {
                    return Err(invalid(
                        "vocabulary IDs/pieces disagree with GGUF tokenizer.ggml.tokens",
                    ));
                }
            }
            if let Some(embedding) = archive.tensor_by_name("token_embd.weight") {
                if embedding.dimensions.get(1).copied() != Some(names.len() as u64) {
                    return Err(invalid(
                        "vocabulary size disagrees with GGUF embedding rows",
                    ));
                }
            }
        }
        let token = |id: Option<u32>| -> Result<Option<(u32, String)>> {
            id.map(|id| {
                names
                    .get(id as usize)
                    .map(|name| (id, name.clone()))
                    .ok_or_else(|| invalid("special token ID is out of range"))
            })
            .transpose()
        };
        check_id(archive, "tokenizer.ggml.bos_token_id", processor.bos_id())?;
        check_id(archive, "tokenizer.ggml.eos_token_id", processor.eos_id())?;
        check_id(
            archive,
            "tokenizer.ggml.unknown_token_id",
            Some(processor.unk_id()),
        )?;
        check_id(
            archive,
            "tokenizer.ggml.padding_token_id",
            processor.pad_id(),
        )?;
        let add_bos = boolean(
            archive,
            "tokenizer.ggml.add_bos_token",
            processor.bos_id().is_some(),
        )?;
        let add_eos = boolean(archive, "tokenizer.ggml.add_eos_token", false)?;
        if (add_bos && processor.bos_id().is_none()) || (add_eos && processor.eos_id().is_none()) {
            return Err(invalid(
                "requested special token is absent from tokenizer.model",
            ));
        }
        let bos = token(processor.bos_id())?;
        let eos = token(processor.eos_id())?;
        Ok(Self {
            processor,
            visible_proto,
            visible: OnceLock::new(),
            controls,
            bos,
            eos,
            add_bos,
            add_eos,
        })
    }
    pub(super) fn encode(&self, prompt: &str, add_special: bool) -> Result<Vec<u32>> {
        // Templates may include an explicit boundary token. Consume that one
        // boundary separately; inserting automatic BOS/EOS must not duplicate it.
        let mut text = prompt;
        let mut explicit_bos = false;
        let mut explicit_eos = false;
        if let Some((_, piece)) = &self.bos {
            if let Some(rest) = text.strip_prefix(piece) {
                text = rest;
                explicit_bos = true;
            }
        }
        if let Some((_, piece)) = &self.eos {
            if let Some(rest) = text.strip_suffix(piece) {
                text = rest;
                explicit_eos = true;
            }
        }
        // A multi-turn template can have EOS/BOS or other controls inside the
        // text. Normal SentencePiece excludes CONTROL pieces from encoding.
        // Keep one normalizer pass, rather than independently normalizing text
        // fragments and accidentally introducing extra dummy whitespace.
        let mut expected = vec![0usize; self.controls.len()];
        let mut remainder = text;
        loop {
            // Earliest symbol first, longest match for overlapping names.
            let next = self
                .controls
                .iter()
                .enumerate()
                .filter_map(|(index, piece)| {
                    remainder
                        .find(piece)
                        .map(|offset| (offset, piece.len(), index))
                })
                .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            let Some((offset, len, index)) = next else {
                break;
            };
            expected[index] += 1;
            remainder = &remainder[offset + len..];
        }
        let processor = if expected.iter().any(|&count| count > 0) {
            self.visible_processor()?
        } else {
            &self.processor
        };
        let mut ids: Vec<u32> = processor
            .encode(text)
            .map_err(|e| invalid(&e.to_string()))?
            .into_iter()
            .map(|p| p.id)
            .collect();
        for (index, &count) in expected.iter().enumerate().filter(|(_, count)| **count > 0) {
            let id = self
                .processor
                .piece_to_id(&self.controls[index])
                .map_err(|e| invalid(&e.to_string()))?
                .ok_or_else(|| invalid("declared control has no ID"))?;
            if ids.iter().filter(|&&token| token == id).count() != count {
                // WORD models do not segment user-defined interior symbols.
                // Refuse explicitly instead of silently encoding them as UNK.
                return Err(invalid("model encoder cannot preserve interior control tokens; use an LLM UNIGRAM/BPE tokenizer"));
            }
        }
        if explicit_bos || (add_special && self.add_bos) {
            if let Some((id, _)) = &self.bos {
                ids.insert(0, *id);
            }
        }
        if explicit_eos || (add_special && self.add_eos) {
            if let Some((id, _)) = &self.eos {
                ids.push(*id);
            }
        }
        Ok(ids)
    }
    fn visible_processor(&self) -> Result<&SentencePieceProcessor> {
        self.visible
            .get_or_init(|| {
                SentencePieceProcessor::from_serialized_proto(&self.visible_proto)
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| invalid(e))
    }
    pub(super) fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        let processor = if skip_special {
            &self.processor
        } else {
            self.visible_processor()?
        };
        processor
            .decode_piece_ids(ids)
            .map_err(|e| invalid(&e.to_string()))
    }
    pub(super) fn piece_to_id(
        &self,
        piece: &str,
    ) -> std::result::Result<Option<u32>, std::ffi::NulError> {
        self.processor.piece_to_id(piece)
    }
    pub(super) fn eos_id(&self) -> Option<u32> {
        self.processor.eos_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn vint(out: &mut Vec<u8>, mut value: u64) {
        while value >= 128 {
            out.push((value as u8) | 128);
            value >>= 7;
        }
        out.push(value as u8);
    }
    fn integer(out: &mut Vec<u8>, field: u64, value: u64) {
        vint(out, field << 3);
        vint(out, value);
    }
    fn bytes(out: &mut Vec<u8>, field: u64, value: &[u8]) {
        vint(out, (field << 3) | 2);
        vint(out, value.len() as u64);
        out.extend(value);
    }
    fn model() -> Vec<u8> {
        model_with_type(1)
    }
    fn model_with_type(kind: u64) -> Vec<u8> {
        let mut model = Vec::new();
        for (name, kind) in [
            ("<unk>", 2),
            ("<s>", 3),
            ("</s>", 3),
            ("▁Hello", 1),
            ("▁world", 1),
            ("▁", 1),
            ("<|user|>", 4),
            ("<pad>", 3),
        ] {
            let mut piece = Vec::new();
            bytes(&mut piece, 1, name.as_bytes());
            vint(&mut piece, (2 << 3) | 5);
            piece.extend(0f32.to_le_bytes());
            integer(&mut piece, 3, kind);
            bytes(&mut model, 1, &piece);
        }
        let mut trainer = Vec::new();
        integer(&mut trainer, 3, kind);
        integer(&mut trainer, 4, 8);
        integer(&mut trainer, 40, 0);
        integer(&mut trainer, 41, 1);
        integer(&mut trainer, 42, 2);
        integer(&mut trainer, 43, 7);
        bytes(&mut model, 2, &trainer);
        let mut normalizer = Vec::new();
        bytes(&mut normalizer, 1, b"identity");
        integer(&mut normalizer, 3, 1);
        integer(&mut normalizer, 4, 1);
        integer(&mut normalizer, 5, 1);
        bytes(&mut model, 3, &normalizer);
        // Unknown extensions survive byte-for-byte, including group wire types.
        bytes(&mut model, 200, b"normalizer-and-extension-provenance");
        vint(&mut model, (201 << 3) | 3);
        integer(&mut model, 1, 3);
        vint(&mut model, (201 << 3) | 4);
        model
    }
    fn archive(values: &[(&str, GgufValue)]) -> (tempfile::TempDir, GgufArchive) {
        let mut data = b"GGUF".to_vec();
        data.extend(3u32.to_le_bytes());
        data.extend(0u64.to_le_bytes());
        data.extend((values.len() as u64).to_le_bytes());
        for (key, value) in values {
            data.extend((key.len() as u64).to_le_bytes());
            data.extend(key.as_bytes());
            match value {
                GgufValue::Bool(v) => {
                    data.extend(7u32.to_le_bytes());
                    data.push(u8::from(*v));
                }
                GgufValue::U32(v) => {
                    data.extend(4u32.to_le_bytes());
                    data.extend(v.to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        data.resize(data.len().div_ceil(32) * 32, 0);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metadata.gguf");
        std::fs::write(&path, data).unwrap();
        let archive = GgufArchive::mmap_path(&path).unwrap();
        (dir, archive)
    }
    #[test]
    fn sentencepiece_bos_eos_flags_match_metadata_and_do_not_duplicate_templates() {
        let model = model();
        let codec = SentencePieceCodec::from_proto(&model, None).unwrap();
        assert_eq!(codec.encode("Hello world", false).unwrap(), [3, 4]);
        assert_eq!(codec.encode("Hello world", true).unwrap(), [1, 3, 4]);
        assert_eq!(codec.encode("<s>Hello world", true).unwrap(), [1, 3, 4]);
        assert_eq!(
            codec.encode("<s>Hello world</s>", false).unwrap(),
            [1, 3, 4, 2]
        );
        let (_dir, gguf) = archive(&[
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_eos_token", GgufValue::Bool(true)),
            ("tokenizer.ggml.bos_token_id", GgufValue::U32(1)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(2)),
        ]);
        let codec = SentencePieceCodec::from_proto(&model, Some(&gguf)).unwrap();
        assert_eq!(codec.encode("Hello world", true).unwrap(), [3, 4, 2]);
        assert_eq!(codec.encode("Hello world", false).unwrap(), [3, 4]);
        assert_eq!(codec.encode("Hello world</s>", true).unwrap(), [3, 4, 2]);
        let (_dir, bad) = archive(&[("tokenizer.ggml.bos_token_id", GgufValue::U32(2))]);
        assert!(SentencePieceCodec::from_proto(&model, Some(&bad)).is_err());
        let (_dir, bad) = archive(&[("tokenizer.ggml.add_bos_token", GgufValue::U32(1))]);
        assert!(SentencePieceCodec::from_proto(&model, Some(&bad)).is_err());
    }
    #[test]
    fn sentencepiece_skip_special_false_exposes_controls_without_replacing_normalizer() {
        let model = model();
        let codec = SentencePieceCodec::from_proto(&model, None).unwrap();
        assert!(codec.visible.get().is_none());
        assert_eq!(codec.decode(&[1, 3, 4, 2, 7], true).unwrap(), "Hello world");
        let visible = codec.decode(&[1, 3, 4, 2, 7], false).unwrap();
        for piece in ["<s>", "Hello", "world", "</s>", "<pad>"] {
            assert!(visible.contains(piece), "{visible:?}");
        }
        assert!(codec.visible.get().is_some());
        assert_eq!(
            codec.decode(&[3, 4], false).unwrap(),
            codec.decode(&[3, 4], true).unwrap()
        );
        assert!(codec.decode(&[999], false).is_err());
        let (patched, names, controls) = visible_model(&model).unwrap();
        assert!(controls.contains(&"</s>".to_owned()));
        assert_eq!(names.len(), 8);
        let changes: Vec<_> = model.iter().zip(&patched).filter(|(a, b)| a != b).collect();
        assert_eq!(changes.len(), 3);
        assert!(changes.iter().all(|(a, b)| **a == 3 && **b == 4));
        assert!(patched.ends_with(&model[model.len() - 7..]));
    }
    #[test]
    fn malformed_proto_is_refused_without_dropping_unknown_fields_or_wrapping_lengths() {
        for bad in [
            vec![0],
            vec![10, 127, 1],
            vec![10, 128],
            vec![0xff; 11],
            vec![13, 0, 0],
            vec![11, 20],
        ] {
            assert!(visible_model(&bad).is_err());
        }
    }
    #[test]
    fn internal_controls_preserve_multi_turn_ids_and_plain_text_normalization() {
        let codec = SentencePieceCodec::from_proto(&model(), None).unwrap();
        assert_eq!(codec.encode("Hello</s> world", false).unwrap(), [3, 2, 4]);
        assert_eq!(
            codec.encode("<s>Hello</s> world</s>", true).unwrap(),
            [1, 3, 2, 4, 2]
        );
        assert_eq!(codec.encode("Hello<pad> world", false).unwrap(), [3, 7, 4]);
        for text in [
            "Hello world",
            "Hello  world",
            "\tééHello\nworld",
            "  Hello ",
        ] {
            let original = codec
                .processor
                .encode(text)
                .unwrap()
                .into_iter()
                .map(|p| p.id)
                .collect::<Vec<_>>();
            assert_eq!(codec.encode(text, false).unwrap(), original);
        }
    }
    #[test]
    fn mismatched_vocabulary_ids_or_embedding_rows_are_refused() {
        let proto = model();
        let (_, names, _) = visible_model(&proto).unwrap();
        let (_dir, mut gguf) = archive(&[]);
        gguf.metadata.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(names.iter().cloned().map(GgufValue::String).collect()),
        );
        assert!(SentencePieceCodec::from_proto(&proto, Some(&gguf)).is_ok());
        if let Some(GgufValue::Array(tokens)) = gguf.metadata.get_mut("tokenizer.ggml.tokens") {
            tokens.swap(3, 4);
        }
        assert!(SentencePieceCodec::from_proto(&proto, Some(&gguf)).is_err());
        gguf.metadata.remove("tokenizer.ggml.tokens");
        gguf.tensors.push(crate::gguf::GgufTensorInfo {
            name: "token_embd.weight".into(),
            dimensions: vec![256, 7],
            ggml_type: 0,
            offset: 0,
        });
        assert!(SentencePieceCodec::from_proto(&proto, Some(&gguf)).is_err());
        gguf.tensors[0].dimensions[1] = 8;
        assert!(SentencePieceCodec::from_proto(&proto, Some(&gguf)).is_ok());
    }
    #[test]
    fn word_model_refuses_lost_interior_controls_instead_of_encoding_unk() {
        let codec = SentencePieceCodec::from_proto(&model_with_type(3), None).unwrap();
        assert_eq!(codec.encode("Hello world", false).unwrap(), [3, 4]);
        assert!(codec
            .encode("Hello</s> world", false)
            .unwrap_err()
            .to_string()
            .contains("interior control"));
    }
    #[test]
    fn optional_trained_sentencepiece_reference_ids_normalizer_and_metadata() {
        let Ok(path) = std::env::var("RBITNET_SENTENCEPIECE_TEST_MODEL") else {
            return;
        };
        let proto = std::fs::read(&path).expect("explicit reference model must exist");
        let reference = SentencePieceProcessor::from_serialized_proto(&proto).unwrap();
        let codec = SentencePieceCodec::from_proto(&proto, None).unwrap();
        let known = [8, 465, 10, 947, 41, 10, 170, 168, 110, 28, 20, 143, 4];
        assert_eq!(
            codec
                .encode("I saw a girl with a telescope.", false)
                .unwrap(),
            known,
            "sentencepiece 0.11.3 upstream trained toy model IDs"
        );
        assert_eq!(
            codec.decode(&known, true).unwrap(),
            "I saw a girl with a telescope."
        );
        for text in [
            "I saw a girl with a telescope.",
            "  Hello   world  ",
            "été, café, résumé, 🙂.",
            "你好\n世界",
            "a\tb\r\nc",
            "",
        ] {
            let ids = reference
                .encode(text)
                .unwrap()
                .into_iter()
                .map(|p| p.id)
                .collect::<Vec<_>>();
            assert_eq!(codec.encode(text, false).unwrap(), ids);
            assert_eq!(
                codec.decode(&ids, true).unwrap(),
                reference.decode_piece_ids(&ids).unwrap()
            );
            if !ids.contains(&0) {
                assert_eq!(
                    codec.decode(&ids, false).unwrap(),
                    reference.decode_piece_ids(&ids).unwrap()
                );
            }
            for (add_bos, add_eos) in [(false, false), (true, false), (false, true), (true, true)] {
                let (_dir, gguf) = archive(&[
                    ("tokenizer.ggml.add_bos_token", GgufValue::Bool(add_bos)),
                    ("tokenizer.ggml.add_eos_token", GgufValue::Bool(add_eos)),
                ]);
                let c = SentencePieceCodec::from_proto(&proto, Some(&gguf)).unwrap();
                let mut expected = ids.clone();
                if add_bos {
                    expected.insert(0, 1);
                }
                if add_eos {
                    expected.push(2);
                }
                assert_eq!(c.encode(text, true).unwrap(), expected);
            }
        }
        let ids = codec.encode("<s>I saw</s> a girl</s>", true).unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == 1).count(), 1);
        assert_eq!(ids.iter().filter(|&&id| id == 2).count(), 2);
        let visible = codec.decode(&ids, false).unwrap();
        assert!(visible.contains("<s>") && visible.matches("</s>").count() == 2);
        assert!(!codec.decode(&ids, true).unwrap().contains("</s>"));
        let (_, names, _) = visible_model(&proto).unwrap();
        let (_dir, mut gguf) = archive(&[]);
        gguf.metadata.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(names.into_iter().map(GgufValue::String).collect()),
        );
        assert!(SentencePieceCodec::from_proto(&proto, Some(&gguf)).is_ok());
        println!("Trained SentencePiece reference: vocab={}, ordinary IDs/normalizer/4 BOS-EOS policies/interior controls exact",reference.len());
    }
    #[test]
    fn optional_actual_mistral_sentencepiece_ids_and_generations_match_hf() {
        if std::env::var("RBITNET_SENTENCEPIECE_REAL_TEST").as_deref() != Ok("1") {
            return;
        }
        use crate::backend::BackendKind;
        use crate::llama::LlamaRuntime;
        use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
        use crate::sampling::SamplingOptions;
        use std::path::Path;
        use std::sync::Arc;
        let gguf = std::env::var("RBITNET_SENTENCEPIECE_REAL_GGUF").unwrap();
        let folder = Path::new(&gguf).parent().unwrap();
        let sp_path = folder.join("tokenizer.model");
        let hf_path = folder.join("tokenizer.json");
        let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
        let codec = SentencePieceCodec::load(&sp_path, Some(&archive)).unwrap();
        let hf = LoadedPromptTokenizer::from_path_for_gguf(&hf_path, &archive).unwrap();
        let original = SentencePieceProcessor::open(&sp_path).unwrap();
        let mut hf_policy_differences = 0;
        for text in [
            "Bonjour le monde.",
            "  été, café, résumé, 🙂.  ",
            "你好\n世界",
            "a\tb\r\nc",
            "",
        ] {
            let plain = original
                .encode(text)
                .unwrap()
                .into_iter()
                .map(|p| p.id)
                .collect::<Vec<_>>();
            for special in [false, true] {
                let mut expected = plain.clone();
                if special && codec.add_bos {
                    expected.insert(0, codec.bos.as_ref().unwrap().0);
                }
                if special && codec.add_eos {
                    expected.push(codec.eos.as_ref().unwrap().0);
                }
                let ids = codec.encode(text, special).unwrap();
                assert_eq!(
                    ids, expected,
                    "actual original SentencePiece normalizer/IDs {text:?} special={special}"
                );
                if ids != hf.encode_ids(text, special).unwrap() {
                    hf_policy_differences += 1;
                }
            }
        }
        let prompt = "<s>[INST]Quelle est la capitale de la France ? Réponds en un mot.[/INST]";
        let ids = codec.encode(prompt, true).unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == 1).count(), 1);
        assert_eq!(
            hf.encode_ids(prompt, true).unwrap(),
            hf.encode_ids(prompt, false).unwrap(),
            "explicit HF BOS must not be inserted twice"
        );
        let turns = "<s>[INST]Bonjour.[/INST]Salut.</s><s>[INST]Donne le résultat de 2 + 2.[/INST]";
        let ids = codec.encode(turns, true).unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == 1).count(), 2);
        assert_eq!(ids.iter().filter(|&&id| id == 2).count(), 1);
        assert!(codec.decode(&ids, false).unwrap().contains("</s><s>"));
        println!("Actual Mistral normalization: original SentencePiece exact; HF converted-policy differences={hf_policy_differences}; multi-turn controls preserved");
        std::env::set_var("RBITNET_MAX_SEQ", "256");
        std::env::set_var("RBITNET_PREFIX_KV", "1");
        let prompts = [
            "[INST]Quelle est la capitale de la France ? Réponds en un mot.[/INST]",
            "[INST]Donne le résultat de 2 + 2.[/INST]",
        ];
        for text in prompts {
            assert_eq!(
                codec.encode(text, true).unwrap(),
                hf.encode_ids(text, true).unwrap(),
                "generation comparison requires equal input IDs"
            );
        }
        let mut seeded = SamplingOptions::from_temperature(0.7);
        seeded.seed = Some(42);
        let mut penalty = SamplingOptions::from_temperature(0.0);
        penalty.frequency_penalty = 0.2;
        penalty.presence_penalty = 0.1;
        let samples = [SamplingOptions::from_temperature(0.0), seeded, penalty];
        for backend in [BackendKind::Cpu, BackendKind::Cuda] {
            let before = crate::perf::snapshot().cuda_graph_replays;
            let mut expected = Vec::new();
            {
                let mut runtime =
                    LlamaRuntime::load(Arc::clone(&archive), &hf_path, backend).unwrap();
                for text in prompts {
                    for sample in samples {
                        expected.push(runtime.generate_with_timings(text, 12, sample).unwrap().0);
                    }
                }
            }
            let mut runtime = LlamaRuntime::load(Arc::clone(&archive), &sp_path, backend).unwrap();
            let mut n = 0;
            for text in prompts {
                for sample in samples {
                    for _ in 0..2 {
                        let actual = runtime.generate_with_timings(text, 12, sample).unwrap().0;
                        assert_eq!(
                            actual, expected[n],
                            "actual Mistral HF/SP backend={backend:?} output={n}"
                        );
                    }
                    n += 1;
                }
            }
            assert!(expected[0].contains("Paris"));
            assert!(!expected.iter().any(|s| s.contains('\u{fffd}')));
            for sample in samples {
                let first = runtime.generate_with_timings(turns, 12, sample).unwrap().0;
                let replay = runtime.generate_with_timings(turns, 12, sample).unwrap().0;
                assert_eq!(
                    first, replay,
                    "SP multi-turn prefix replay backend={backend:?}"
                );
                assert!(!first.contains('\u{fffd}'));
            }
            if backend == BackendKind::Cuda {
                assert!(
                    crate::perf::snapshot().cuda_graph_replays > before,
                    "actual CUDA resident path required"
                );
            }
            println!("ACTUAL_SP_MISTRAL backend={backend:?}: 32000 pieces metadata, original normalizer, boundaries, 6 reference/12 replay generations match HF on equal input IDs; 6 SP multi-turn generations replay exactly; Paris present");
        }
    }
}
