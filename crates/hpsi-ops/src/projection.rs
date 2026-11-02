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

/// Discover matrix-shaped floating point tensors without loading their payloads.
/// The returned order follows the Safetensors reader's deterministic tensor order.
pub fn discover_projection_candidates(tensors: &[SafeTensorInfo]) -> Vec<ProjectionCandidate> {
    tensors
        .iter()
        .filter_map(|tensor| {
            if tensor.shape.len() != 2
                || tensor.shape[0] == 0
                || tensor.shape[1] == 0
                || !matches!(tensor.dtype.as_str(), "F32" | "F16" | "BF16")
            {
                return None;
            }
            let rows = tensor.shape[0];
            let cols = tensor.shape[1];
            // Current VQ groups eight adjacent input features. Other shapes
            // remain visible to inspection but are not yet VQ-import candidates.
            if cols % 8 != 0 {
                return None;
            }
            Some(ProjectionCandidate {
                name: tensor.name.clone(),
                rows,
                cols,
                dtype: tensor.dtype.clone(),
                conventional_projection_name: has_projection_suffix(&tensor.name),
            })
        })
        .collect()
}

fn has_projection_suffix(name: &str) -> bool {
    let leaf = name.rsplit('.').next().unwrap_or(name).to_ascii_lowercase();
    matches!(
        leaf.as_str(),
        "q_proj"
            | "k_proj"
            | "v_proj"
            | "o_proj"
            | "out_proj"
            | "query"
            | "key"
            | "value"
            | "dense"
            | "fc1"
            | "fc2"
            | "up_proj"
            | "down_proj"
            | "gate_proj"
            | "w1"
            | "w2"
            | "w3"
    )
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
            info("model.layers.0.mlp.custom.weight", "F16", vec![32, 24]),
            info("model.layers.0.mlp.bad.weight", "F32", vec![32, 7]),
            info("embedding.weight", "I8", vec![100, 128]),
        ];
        let candidates = discover_projection_candidates(&tensors);
        assert_eq!(candidates.len(), 2);
        assert!(candidates[0].conventional_projection_name);
        assert_eq!((candidates[0].rows, candidates[0].cols), (64, 128));
        assert!(!candidates[1].conventional_projection_name);
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
