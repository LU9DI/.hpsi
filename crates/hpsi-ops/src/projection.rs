use crate::SafeTensorInfo;

/// A rank-2 floating-point tensor that can be considered by the HPSI importer.
/// Discovery is intentionally conservative: it never assigns calibration or
/// assumes a model architecture. Callers must explicitly approve each candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionCandidate {
    pub name: String,
    pub rows: u64,
    pub cols: u64,
    pub dtype: String,
    pub conventional_projection_name: bool,
}

impl ProjectionCandidate {
    /// Calibration is associated only by an exact layer/tensor name match.
    pub fn matches_calibration(&self, calibration_layer_name: &str) -> bool {
        self.name == calibration_layer_name
    }
}

/// Errors reported while discovering projection candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionError(pub &'static str);

impl std::fmt::Display for ProjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for ProjectionError {}

fn clone_candidate_text(value: &str) -> Result<String, ProjectionError> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(value.len())
        .map_err(|_| ProjectionError("could not allocate projection candidate text"))?;
    cloned.push_str(value);
    Ok(cloned)
}

/// Discover matrix-shaped floating point tensors without loading their payloads.
/// The returned order follows the Safetensors reader's deterministic tensor order.
/// This fallible API reports allocation failures instead of panicking.
pub fn try_discover_projection_candidates(
    tensors: &[SafeTensorInfo],
) -> Result<Vec<ProjectionCandidate>, ProjectionError> {
    let mut candidates = Vec::new();
    for tensor in tensors {
        if tensor.shape.len() != 2
            || tensor.shape[0] == 0
            || tensor.shape[1] == 0
            || !matches!(tensor.dtype.as_str(), "F32" | "F16" | "BF16")
        {
            continue;
        }
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];
        // VQ groups eight adjacent input features. Whether packed selectors
        // are byte-aligned also depends on the selected codebook count,
        // which is not known during candidate discovery.
        if cols % 8 != 0 {
            continue;
        }
        candidates
            .try_reserve(1)
            .map_err(|_| ProjectionError("could not allocate projection candidate directory"))?;
        candidates.push(ProjectionCandidate {
            name: clone_candidate_text(&tensor.name)?,
            rows,
            cols,
            dtype: clone_candidate_text(&tensor.dtype)?,
            conventional_projection_name: has_projection_suffix(&tensor.name),
        });
    }
    Ok(candidates)
}

/// Infallible convenience wrapper. Use the fallible API when processing
/// untrusted or potentially large model headers.
pub fn discover_projection_candidates(tensors: &[SafeTensorInfo]) -> Vec<ProjectionCandidate> {
    try_discover_projection_candidates(tensors)
        .expect("projection candidate allocation failed")
}

fn has_projection_suffix(name: &str) -> bool {
    let mut components = name.rsplit('.');
    let leaf = components.next().unwrap_or(name);
    let projection = if leaf.eq_ignore_ascii_case("weight") {
        components.next().unwrap_or(leaf)
    } else {
        leaf
    };
    [
        "q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "query", "key", "value",
        "dense", "fc1", "fc2", "up_proj", "down_proj", "gate_proj", "w1", "w2", "w3",
    ]
    .iter()
    .any(|suffix| projection.eq_ignore_ascii_case(suffix))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, dtype: &str, shape: Vec<u64>) -> SafeTensorInfo {
        SafeTensorInfo {
            name: name.into(),
            dtype: dtype.into(),
            shape,
            data_start: 0,
            data_end: 0,
        }
    }

    #[test]
    fn discovers_group_aligned_float_matrices_without_loading_payloads() {
        let tensors = vec![
            info(
                "model.layers.0.self_attn.q_proj.weight",
                "BF16",
                vec![64, 128],
            ),
            info("model.layers.0.norm.weight", "F32", vec![128]),
            info("model.layers.0.mlp.custom.weight", "F16", vec![32, 32]),
            info("model.layers.0.mlp.bad.weight", "F32", vec![32, 7]),
            info("model.layers.0.mlp.group8.weight", "F32", vec![32, 24]),
            info("embedding.weight", "I8", vec![100, 128]),
        ];
        let candidates = discover_projection_candidates(&tensors);
        assert_eq!(candidates.len(), 3);
        assert!(candidates[0].conventional_projection_name);
        assert_eq!((candidates[0].rows, candidates[0].cols), (64, 128));
        assert!(!candidates[1].conventional_projection_name);
        assert_eq!((candidates[2].rows, candidates[2].cols), (32, 24));
    }

    #[test]
    fn fallible_discovery_preserves_order_and_filters_without_payload_reads() {
        let tensors = vec![
            info("first.q_proj.weight", "F32", vec![8, 16]),
            info("not.a.matrix", "F32", vec![16]),
            info("second.K_PROJ.weight", "F16", vec![4, 8]),
        ];
        let candidates = try_discover_projection_candidates(&tensors).unwrap();
        assert_eq!(
            candidates.iter().map(|candidate| candidate.name.as_str()).collect::<Vec<_>>(),
            vec!["first.q_proj.weight", "second.K_PROJ.weight"]
        );
        assert!(candidates[0].conventional_projection_name);
        assert!(candidates[1].conventional_projection_name);
    }

    #[test]
    fn calibration_match_is_exact_and_never_guessed() {
        let candidate = ProjectionCandidate {
            name: "model.layers.0.mlp.up_proj".into(),
            rows: 16,
            cols: 32,
            dtype: "F32".into(),
            conventional_projection_name: true,
        };
        assert!(candidate.matches_calibration("model.layers.0.mlp.up_proj"));
        assert!(!candidate.matches_calibration("model.layers.1.mlp.up_proj"));
        assert!(!candidate.matches_calibration("up_proj"));
    }
}
