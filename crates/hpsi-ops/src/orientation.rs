//! Orientation-aware resolution of VQ tensor pairs from container metadata.
//!
//! The importer records, per source tensor, `weight_orientation`,
//! `orientation_source` and `logical_shape`. This module is the single Rust
//! place that interprets those fields for a consumer, so the CLI, the reader
//! path and the executor cannot drift into three different interpretations.
//!
//! It deliberately separates four claims, which are not interchangeable:
//!
//! * container integrity — CRC32C and descriptor checks, done by `hpsi-format`;
//! * metadata self-consistency — the fields agree with each other and with the
//!   physical layout of the tensors they describe;
//! * architectural compatibility — the declared orientation equals the
//!   orientation an externally supplied model definition requires;
//! * numeric correctness of a reconstruction — only provable against the
//!   original weights, which a container does not contain.
//!
//! Self-consistency is the only thing this module can prove on its own. A
//! square `[n, n]` matrix is reported as not verifiable without external
//! architectural information, because both orientations produce the identical
//! physical layout; nothing here infers orientation from tensor names or from
//! matching dimensions.

use crate::vq::{CODEBOOK_SIZE, GROUP_SIZE, HpsiVqLayout};
use crate::{WeightOrientation, WeightOrientationSource};
use hpsi_format::{DType, ModelReader, TensorDescriptor};
use serde_json::Value;

/// Format identifier an architecture manifest must declare to be accepted.
pub const ARCHITECTURE_MANIFEST_FORMAT: &str = "hpsi-architecture-manifest";
pub const ARCHITECTURE_MANIFEST_VERSION: u64 = 1;

const VQ_CODEBOOK_SUFFIX: &str = ".hpsi_vq.codebook";
const VQ_INDICES_SUFFIX: &str = ".hpsi_vq.indices";
const QUANTIZE_DISPOSITION: &str = "quantize_vq";
const RAW_DISPOSITION: &str = "preserve_raw";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrientationError(pub String);

impl std::fmt::Display for OrientationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for OrientationError {}

/// How far the orientation of one tensor has actually been established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrientationAssurance {
    /// The declared orientation equals what external architectural information
    /// requires for this exact tensor name. The only confirmed level.
    ArchitecturallyConfirmed,
    /// Structurally self-consistent and calibration-provenanced, but no
    /// architectural information was supplied, so nothing confirms it.
    DeclaredNotVerified,
    /// Square `[n, n]`: both orientations yield the identical physical layout,
    /// so even structural consistency cannot discriminate them.
    SquareNotVerifiable,
    /// The container carries no orientation metadata for this tensor.
    LegacyUndeclared,
}

impl OrientationAssurance {
    pub fn as_str(self) -> &'static str {
        match self {
            OrientationAssurance::ArchitecturallyConfirmed => "architecturally_confirmed",
            OrientationAssurance::DeclaredNotVerified => "declared_not_verified",
            OrientationAssurance::SquareNotVerifiable => "square_not_verifiable",
            OrientationAssurance::LegacyUndeclared => "legacy_undeclared",
        }
    }

    /// True only for orientation confirmed against architectural information.
    pub fn is_confirmed(self) -> bool {
        self == OrientationAssurance::ArchitecturallyConfirmed
    }
}

/// What architectural function a source tensor performs.
///
/// This is a dimension independent of [`WeightOrientation`]: an orientation says
/// how a projection matrix is laid out, a role says what the tensor is for. A
/// tied embedding/output head is legitimately reached as a table *and* as a
/// projection, which is why the producer records the one role it had evidence
/// for while the audit keeps layout confirmation separate from role
/// confirmation. No role is ever inferred here: a value exists only because a
/// producer wrote it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    /// Weight of a `torch.nn.Linear` projection.
    LinearProjection,
    /// Weight of a `transformers` `Conv1D` projection.
    Conv1dProjection,
    /// Lookup table: an axis is indexed, nothing multiplies.
    EmbeddingTable,
    /// A role the producer identified, that is none of the above.
    Other,
    /// The producer states the role was never established.
    Unknown,
}

impl TensorRole {
    pub fn as_str(self) -> &'static str {
        match self {
            TensorRole::LinearProjection => "linear_projection",
            TensorRole::Conv1dProjection => "conv1d_projection",
            TensorRole::EmbeddingTable => "embedding_table",
            TensorRole::Other => "other",
            TensorRole::Unknown => "unknown",
        }
    }

    pub fn parse(value: &str) -> Result<Self, OrientationError> {
        match value {
            "linear_projection" => Ok(TensorRole::LinearProjection),
            "conv1d_projection" => Ok(TensorRole::Conv1dProjection),
            "embedding_table" => Ok(TensorRole::EmbeddingTable),
            "other" => Ok(TensorRole::Other),
            "unknown" => Ok(TensorRole::Unknown),
            _ => Err(OrientationError(format!(
                "unknown tensor_role {value:?}; expected linear_projection, \
                 conv1d_projection, embedding_table, other or unknown"
            ))),
        }
    }

    /// True when the matrix multiplies, i.e. when an orientation token is an
    /// architectural claim rather than a mere encoding fact.
    pub fn is_projection(self) -> bool {
        matches!(
            self,
            TensorRole::LinearProjection | TensorRole::Conv1dProjection
        )
    }

    /// Parse a `tensor_role` token, reporting an unknown value under the code the
    /// Python auditor reports too. An unrecognised role is never rounded onto a
    /// neighbouring one.
    pub fn parse_token(token: &str) -> Result<Self, (LayoutViolation, String)> {
        TensorRole::parse(token).map_err(|_| {
            (
                LayoutViolation::RoleTokenUnknown,
                format!("unknown tensor_role {token:?}"),
            )
        })
    }
}

/// How far the architectural role of one tensor has been established.
///
/// Deliberately separate from [`OrientationAssurance`]: a confirmed layout is no
/// evidence about the role, and a confirmed role is no evidence that a recorded
/// orientation is right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleAssurance {
    /// No `tensor_role` was recorded: a pre-role artifact, read as before.
    NotDeclared,
    /// A role is recorded and the supplied architecture states the same role.
    Confirmed,
    /// A role is recorded, but nothing outside the container agrees with it.
    DeclaredNotConfirmed,
}

impl RoleAssurance {
    pub fn as_str(self) -> &'static str {
        match self {
            RoleAssurance::NotDeclared => "role_not_declared",
            RoleAssurance::Confirmed => "role_confirmed",
            RoleAssurance::DeclaredNotConfirmed => "role_declared_not_confirmed",
        }
    }
}

impl ArchitectureExpectation {
    /// The layout the model definition requires, either stated outright or
    /// entailed by the roles and indexing axis it states.
    ///
    /// A projection role names its own layout, and an indexed table puts its
    /// features on the axis the quantizer groups along. `None` means the
    /// definition says nothing that could confirm or refute a layout — which is
    /// absence of evidence, never agreement.
    pub fn effective_orientation(&self) -> Option<WeightOrientation> {
        if let Some(orientation) = self.orientation {
            return Some(orientation);
        }
        self.roles.iter().find_map(|role| match role {
            TensorRole::LinearProjection => Some(WeightOrientation::LinearOutIn),
            TensorRole::Conv1dProjection => Some(WeightOrientation::Conv1DInOut),
            TensorRole::EmbeddingTable => match self.index_axis {
                Some(0) => Some(WeightOrientation::LinearOutIn),
                Some(1) => Some(WeightOrientation::Conv1DInOut),
                _ => None,
            },
            _ => None,
        })
    }
}

/// Orientation, roles and indexing layout one model definition requires for one
/// exact tensor name.
///
/// A manifest entry may state an orientation (a projection), one or more roles
/// (any tensor whose architectural function the definition knows), or both. It
/// must state at least one: an entry that names a tensor without saying anything
/// checkable is refused rather than guessed at.
///
/// `roles` is a set because a tied embedding/output head really is reached under
/// two functions at once. Recording only one would force an auditor to call the
/// other a contradiction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureExpectation {
    pub name: String,
    pub orientation: Option<WeightOrientation>,
    /// Rank-2 source shape as the model definition declares it, when known.
    pub source_shape: Option<[u64; 2]>,
    /// Roles the model definition evidences, sorted and deduplicated. Empty
    /// means the entry states no role, which is absence of evidence and never a
    /// contradiction.
    pub roles: Vec<TensorRole>,
    /// Which axis a lookup table indexes, when an indexed role is stated.
    pub index_axis: Option<usize>,
    /// Module-level names the model definition reports for this one parameter.
    ///
    /// A tied embedding/output head is stored once and reached under two names.
    /// Without the aliases an entry saying "table *and* projection" looks like a
    /// contradiction, so the names that carry each role are recorded as data from
    /// the model definition. Absent means the definition stated no aliases, which
    /// is absence of evidence and never a claim of exclusivity.
    pub aliases: Vec<String>,
    /// Which module evidenced which role, when the definition says so.
    ///
    /// The pairing is the part that cannot be recovered from outside: a role
    /// list alone cannot show that `embedding_table` came from the embedding and
    /// `linear_projection` from the output head rather than the other way round.
    pub role_evidence: Vec<RoleEvidence>,
}

/// One module's claim about one tensor, as the model definition reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleEvidence {
    /// The parameter name the module holds, e.g. `lm_head.weight`.
    pub alias: String,
    /// The module path in the model definition, e.g. `lm_head`.
    pub module: String,
    /// The module's class as recorded at capture time, e.g. `linear`.
    pub module_class: String,
    pub role: TensorRole,
    /// Layout claim this class makes, when the class makes one.
    pub orientation: Option<WeightOrientation>,
    /// Indexing axis this class implies, for an indexed role.
    pub index_axis: Option<usize>,
}

impl RoleEvidence {
    fn from_json(name: &str, value: &Value) -> Result<Self, OrientationError> {
        let object = value.as_object().ok_or_else(|| {
            OrientationError(format!(
                "{name}: manifest `role_evidence` entries must be objects"
            ))
        })?;
        let text = |key: &str| -> Result<String, OrientationError> {
            object
                .get(key)
                .and_then(Value::as_str)
                .filter(|item| !item.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    OrientationError(format!(
                        "{name}: manifest role_evidence lacks a non-empty `{key}`"
                    ))
                })
        };
        let alias = text("alias")?;
        let module = text("module")?;
        let module_class = text("module_class")?;
        let role_token = text("tensor_role")?;
        let role = TensorRole::parse_token(&role_token)
            .map_err(|(violation, detail)| OrientationError(violation.render(name, &detail)))?;
        let orientation = match object.get("weight_orientation") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let token = value.as_str().ok_or_else(|| {
                    OrientationError(format!(
                        "{name}: manifest role_evidence `weight_orientation` must be a string"
                    ))
                })?;
                Some(WeightOrientation::parse(token).map_err(|_| {
                    OrientationError(format!(
                        "{name}: unknown role_evidence orientation {token:?}"
                    ))
                })?)
            }
        };
        let index_axis = match object.get("index_axis") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let axis = value.as_u64().ok_or_else(|| {
                    OrientationError(format!(
                        "{name}: manifest role_evidence `index_axis` must be 0 or 1"
                    ))
                })?;
                if axis > 1 {
                    return Err(OrientationError(format!(
                        "{name}: manifest role_evidence `index_axis` must be 0 or 1, found {axis}"
                    )));
                }
                Some(axis as usize)
            }
        };
        // Each class names exactly one layout, so an evidence row that pairs a
        // class with a layout the class does not have is refused here rather than
        // reaching the container as a false expectation.
        if let Some(stated) = orientation {
            let required = match role {
                TensorRole::LinearProjection => Some(WeightOrientation::LinearOutIn),
                TensorRole::Conv1dProjection => Some(WeightOrientation::Conv1DInOut),
                TensorRole::EmbeddingTable => match index_axis {
                    Some(0) => Some(WeightOrientation::LinearOutIn),
                    Some(1) => Some(WeightOrientation::Conv1DInOut),
                    _ => None,
                },
                _ => None,
            };
            if required != Some(stated) {
                return Err(OrientationError(format!(
                    "{name}: role_evidence for alias {alias:?} pairs role {} with orientation {:?}, which that role does not imply",
                    role.as_str(),
                    stated.as_str()
                )));
            }
        }
        if index_axis.is_some() && role != TensorRole::EmbeddingTable {
            return Err(OrientationError(format!(
                "{name}: role_evidence `index_axis` is only meaningful for embedding_table, not {}",
                role.as_str()
            )));
        }
        if role == TensorRole::EmbeddingTable && index_axis.is_none() {
            return Err(OrientationError(format!(
                "{name}: role_evidence for alias {alias:?} names embedding_table without an index_axis"
            )));
        }
        Ok(Self {
            alias,
            module,
            module_class,
            role,
            orientation,
            index_axis,
        })
    }
}

/// Architectural facts supplied from outside the container.
///
/// The manifest is produced from the model definition (module classes and
/// their parameter shapes), never from the container being checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureFacts {
    pub model_type: Option<String>,
    pub expectations: Vec<ArchitectureExpectation>,
}

impl ArchitectureFacts {
    /// Parse and validate an architecture manifest.
    ///
    /// Every expectation must name a tensor and state its orientation
    /// explicitly. Orientation is never derived from a name pattern here.
    pub fn from_json(input: &str) -> Result<Self, OrientationError> {
        let root: Value = serde_json::from_str(input)
            .map_err(|_| OrientationError("architecture manifest is not valid JSON".into()))?;
        let manifest = root
            .as_object()
            .ok_or_else(|| OrientationError("architecture manifest must be an object".into()))?;
        let format = manifest
            .get("format")
            .and_then(Value::as_str)
            .ok_or_else(|| OrientationError("architecture manifest lacks `format`".into()))?;
        if format != ARCHITECTURE_MANIFEST_FORMAT {
            return Err(OrientationError(format!(
                "unsupported architecture manifest format {format:?}, expected {ARCHITECTURE_MANIFEST_FORMAT:?}"
            )));
        }
        if manifest.get("version").and_then(Value::as_u64) != Some(ARCHITECTURE_MANIFEST_VERSION) {
            return Err(OrientationError(format!(
                "unsupported architecture manifest version, expected {ARCHITECTURE_MANIFEST_VERSION}"
            )));
        }
        let model_type = match manifest.get("model_type") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_str()
                    .filter(|text| !text.trim().is_empty())
                    .ok_or_else(|| {
                        OrientationError(
                            "architecture manifest `model_type` must be a string".into(),
                        )
                    })?
                    .to_owned(),
            ),
        };
        let entries = manifest
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                OrientationError("architecture manifest lacks a `tensors` array".into())
            })?;
        let mut expectations: Vec<ArchitectureExpectation> = Vec::new();
        for entry in entries {
            let object = entry.as_object().ok_or_else(|| {
                OrientationError("architecture manifest tensor entry must be an object".into())
            })?;
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| {
                    OrientationError("architecture manifest tensor entry lacks a `name`".into())
                })?
                .to_owned();
            let orientation = match object.get("weight_orientation") {
                None | Some(Value::Null) => None,
                Some(value) => {
                    let token = value.as_str().ok_or_else(|| {
                        OrientationError(format!(
                            "{name}: manifest `weight_orientation` must be a string"
                        ))
                    })?;
                    Some(WeightOrientation::parse(token).map_err(|_| {
                        OrientationError(format!("{name}: unknown manifest orientation {token:?}"))
                    })?)
                }
            };
            let mut roles = match object.get("tensor_role") {
                None | Some(Value::Null) => Vec::new(),
                Some(Value::Array(values)) => {
                    let mut parsed = Vec::new();
                    for value in values {
                        let token = value.as_str().ok_or_else(|| {
                            OrientationError(format!(
                                "{name}: manifest `tensor_role` entries must be strings"
                            ))
                        })?;
                        parsed.push(TensorRole::parse_token(token).map_err(
                            |(violation, detail)| {
                                OrientationError(violation.render(&name, &detail))
                            },
                        )?);
                    }
                    parsed
                }
                Some(value) => {
                    let token = value.as_str().ok_or_else(|| {
                        OrientationError(format!(
                            "{name}: manifest `tensor_role` must be a string or an array of strings"
                        ))
                    })?;
                    vec![
                        TensorRole::parse_token(token).map_err(|(violation, detail)| {
                            OrientationError(violation.render(&name, &detail))
                        })?,
                    ]
                }
            };
            roles.sort_by_key(|role| role.as_str());
            roles.dedup();
            let index_axis = match object.get("index_axis") {
                None | Some(Value::Null) => None,
                Some(value) => {
                    let axis = value.as_u64().ok_or_else(|| {
                        OrientationError(format!("{name}: manifest `index_axis` must be 0 or 1"))
                    })?;
                    if axis > 1 {
                        return Err(OrientationError(format!(
                            "{name}: manifest `index_axis` must be 0 or 1, found {axis}"
                        )));
                    }
                    Some(axis as usize)
                }
            };
            let mut aliases = match object.get("aliases") {
                None | Some(Value::Null) => Vec::new(),
                Some(value) => {
                    let items = value.as_array().ok_or_else(|| {
                        OrientationError(format!("{name}: manifest `aliases` must be an array"))
                    })?;
                    let mut parsed = Vec::new();
                    for item in items {
                        let text = item.as_str().ok_or_else(|| {
                            OrientationError(format!(
                                "{name}: manifest `aliases` entries must be strings"
                            ))
                        })?;
                        if text.trim().is_empty() {
                            return Err(OrientationError(format!(
                                "{name}: manifest `aliases` entries must be non-empty"
                            )));
                        }
                        parsed.push(text.to_owned());
                    }
                    parsed.sort();
                    parsed.dedup();
                    parsed
                }
            };
            let role_evidence = match object.get("role_evidence") {
                None | Some(Value::Null) => Vec::new(),
                Some(value) => {
                    let items = value.as_array().ok_or_else(|| {
                        OrientationError(format!(
                            "{name}: manifest `role_evidence` must be an array"
                        ))
                    })?;
                    let mut parsed = Vec::new();
                    for item in items {
                        let evidence = RoleEvidence::from_json(&name, item)?;
                        if parsed
                            .iter()
                            .any(|existing: &RoleEvidence| existing.alias == evidence.alias)
                        {
                            return Err(OrientationError(format!(
                                "{name}: alias {:?} appears twice in `role_evidence`",
                                evidence.alias
                            )));
                        }
                        parsed.push(evidence);
                    }
                    parsed.sort_by(|left, right| left.alias.cmp(&right.alias));
                    parsed
                }
            };
            if !role_evidence.is_empty() {
                // The pairing and the role list are two views of one claim, so
                // they must say the same thing. A manifest that names a role no
                // module evidences, or drops an evidenced one, is refused instead
                // of being read under whichever half happens to be convenient.
                for evidence in &role_evidence {
                    if !roles.contains(&evidence.role) {
                        return Err(OrientationError(format!(
                            "{name}: role_evidence names {} for alias {:?}, which `tensor_role` does not declare",
                            evidence.role.as_str(),
                            evidence.alias
                        )));
                    }
                    if !aliases.is_empty() && !aliases.contains(&evidence.alias) {
                        return Err(OrientationError(format!(
                            "{name}: role_evidence alias {:?} is not among `aliases`",
                            evidence.alias
                        )));
                    }
                }
                for role in &roles {
                    if !role_evidence.iter().any(|item| item.role == *role) {
                        return Err(OrientationError(format!(
                            "{name}: `tensor_role` names {} with no role_evidence naming the module that evidences it",
                            role.as_str()
                        )));
                    }
                }
                // A row may be less specific than the entry, never different from
                // it: an axis or a layout that disagrees with the entry's would be
                // a second, unresolved claim about the same parameter.
                for evidence in &role_evidence {
                    if evidence.index_axis != index_axis && evidence.index_axis.is_some() {
                        return Err(OrientationError(format!(
                            "{name}: role_evidence for alias {:?} claims index_axis {:?} while the entry declares {index_axis:?}",
                            evidence.alias, evidence.index_axis
                        )));
                    }
                    if evidence.orientation.is_some() && evidence.orientation != orientation {
                        return Err(OrientationError(format!(
                            "{name}: role_evidence for alias {:?} declares orientation {:?} while the entry declares {orientation:?}",
                            evidence.alias,
                            evidence.orientation.map(WeightOrientation::as_str)
                        )));
                    }
                }
            }
            if aliases.is_empty() {
                // When the entry omits `aliases`, the pairing is authoritative:
                // the names it states are exactly the names its evidence uses.
                aliases.extend(role_evidence.iter().map(|item| item.alias.clone()));
                aliases.sort();
                aliases.dedup();
            }
            if orientation.is_none() && roles.is_empty() {
                return Err(OrientationError(format!(
                    "{name}: architecture manifest entry states neither `weight_orientation` nor `tensor_role`; orientation is never inferred from a name and neither is a role"
                )));
            }
            if index_axis.is_some() && !roles.contains(&TensorRole::EmbeddingTable) {
                return Err(OrientationError(format!(
                    "{name}: index_axis is only meaningful alongside the embedding_table role"
                )));
            }
            // A definition may state only a role: its layout is entailed by
            // the class. The container cannot do that, because it must record the
            // layout it actually encoded. So the same matrix is checked with
            // the entailed layout filled in, which still refuses a manifest that
            // states a layout its roles contradict.
            let entailed = {
                let probe = ArchitectureExpectation {
                    name: String::new(),
                    orientation,
                    source_shape: None,
                    roles: roles.clone(),
                    index_axis,
                    aliases: Vec::new(),
                    role_evidence: Vec::new(),
                };
                probe.effective_orientation()
            };
            for role in &roles {
                check_role_layout(
                    &name,
                    Some(*role),
                    orientation.or(entailed),
                    if *role == TensorRole::EmbeddingTable {
                        index_axis
                    } else {
                        None
                    },
                )
                .map_err(OrientationError)?;
            }
            let source_shape = match object.get("source_shape") {
                None | Some(Value::Null) => None,
                Some(value) => {
                    let dims = value.as_array().ok_or_else(|| {
                        OrientationError(format!(
                            "{name}: manifest `source_shape` must be an array"
                        ))
                    })?;
                    if dims.len() != 2 {
                        return Err(OrientationError(format!(
                            "{name}: manifest `source_shape` must have exactly two dimensions"
                        )));
                    }
                    let parsed = dims
                        .iter()
                        .map(|item| {
                            item.as_u64().filter(|value| *value > 0).ok_or_else(|| {
                                OrientationError(format!(
                                    "{name}: manifest `source_shape` entries must be positive integers"
                                ))
                            })
                        })
                        .collect::<Result<Vec<u64>, OrientationError>>()?;
                    Some([parsed[0], parsed[1]])
                }
            };
            if expectations.iter().any(|existing| existing.name == name) {
                return Err(OrientationError(format!(
                    "{name}: architecture manifest declares the same tensor twice"
                )));
            }
            // An alias names one parameter, so two entries claiming it claim the
            // same array twice. This is the manifest-side counterpart of grouping
            // by identity: nothing may merge or split parameters by name.
            for alias in &aliases {
                if let Some(other) = expectations
                    .iter()
                    .find(|existing| existing.aliases.iter().any(|claimed| claimed == alias))
                {
                    return Err(OrientationError(format!(
                        "{name}: alias {alias:?} is declared twice, by {name:?} and {:?}; one name belongs to one parameter",
                        other.name
                    )));
                }
            }
            expectations.push(ArchitectureExpectation {
                name,
                orientation,
                source_shape,
                roles,
                index_axis,
                aliases,
                role_evidence,
            });
        }
        Ok(Self {
            model_type,
            expectations,
        })
    }

    /// Exact-name lookup. No prefix, suffix or fuzzy matching.
    pub fn expectation(&self, name: &str) -> Option<&ArchitectureExpectation> {
        self.expectations.iter().find(|item| item.name == name)
    }
}

/// Orientation facts established for one source tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorOrientationFacts {
    pub name: String,
    pub disposition: String,
    /// Shape of the source matrix as the container declares it.
    pub source_shape: Vec<u64>,
    pub declared: Option<WeightOrientation>,
    pub source: Option<WeightOrientationSource>,
    /// Logical `[out, in]` shape the container declares.
    pub logical_shape: Option<[u64; 2]>,
    pub square: bool,
    pub assurance: OrientationAssurance,
    /// What the supplied architecture requires, when supplied.
    pub architecture: Option<WeightOrientation>,
    pub physical_layout_consistent: bool,
    /// The architectural role the container records, if any. Independent of
    /// `declared`: an orientation is a layout, a role is a function.
    pub role: Option<TensorRole>,
    /// Which axis a lookup table indexes, for indexed roles.
    pub index_axis: Option<usize>,
    /// How far the recorded role has been established.
    pub role_assurance: RoleAssurance,
    /// Roles the supplied architecture evidences for this tensor.
    pub architecture_roles: Vec<TensorRole>,
    /// Parameter names the supplied architecture reports for this one tensor.
    pub architecture_aliases: Vec<String>,
    /// Which module evidenced which of those roles, when the architecture says.
    pub role_evidence: Vec<RoleEvidence>,
}

impl TensorOrientationFacts {
    pub fn orientation_token(&self) -> &'static str {
        match self.declared {
            Some(orientation) => orientation.as_str(),
            None => "<missing>",
        }
    }

    pub fn source_token(&self) -> &'static str {
        match self.source {
            Some(source) => source.as_str(),
            None => "<missing>",
        }
    }

    pub fn role_token(&self) -> &'static str {
        match self.role {
            Some(role) => role.as_str(),
            None => "<missing>",
        }
    }

    /// True when nothing forbids treating this tensor as a multiplied matrix.
    ///
    /// A tensor that records no role at all keeps the pre-role behaviour, which
    /// is what makes the role dimension additive: only an explicitly
    /// non-projecting role (`embedding_table`, `other`, `unknown`) closes the
    /// projection path.
    pub fn is_projection(&self) -> bool {
        self.role.is_none_or(|role| role.is_projection())
    }
}

/// Result of auditing every rank-2 source tensor declared by a container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrientationAudit {
    pub tensors: Vec<TensorOrientationFacts>,
    /// Metadata that is present but structurally invalid, or that contradicts
    /// the supplied architecture. Non-empty means no consumer may trust the
    /// orientation of the affected tensors.
    pub invalid: Vec<String>,
    /// Rank-2 matrices this audit reached a status for: every declared rank-2
    /// source tensor plus every undeclared rank-2 tensor. A matrix whose
    /// declaration is a contract violation appears in `invalid` instead of
    /// here, so it carries no status.
    pub rank2_tensors: usize,
    /// Rank-2 tensors the container does not describe at all.
    pub undeclared_rank2_tensors: usize,
    /// Rank-2 tensors that record an architectural role at all. A container
    /// written before the role contract reports zero here, and is read exactly
    /// as it always was.
    pub roles_declared: usize,
    /// Roles the supplied model definition confirms.
    pub roles_confirmed: usize,
    /// Tensors recorded as indexed tables rather than as projections.
    pub indexed_tables: usize,
    /// Whether an architecture manifest was supplied at all, independently of
    /// whether any of its names matched a declared tensor.
    pub architecture_supplied: bool,
}

impl OrientationAudit {
    pub fn count(&self, assurance: OrientationAssurance) -> usize {
        self.tensors
            .iter()
            .filter(|facts| facts.assurance == assurance)
            .count()
    }

    pub fn count_role(&self, assurance: RoleAssurance) -> usize {
        self.tensors
            .iter()
            .filter(|facts| facts.role_assurance == assurance)
            .count()
    }

    pub fn square_unverified(&self) -> usize {
        self.count(OrientationAssurance::SquareNotVerifiable)
    }

    pub fn confirmed(&self) -> usize {
        self.count(OrientationAssurance::ArchitecturallyConfirmed)
    }

    pub fn has_architecture(&self) -> bool {
        self.tensors
            .iter()
            .any(|facts| facts.architecture.is_some())
    }

    /// Findings for the audited tensor, if it was described at all.
    pub fn facts(&self, name: &str) -> Option<&TensorOrientationFacts> {
        self.tensors.iter().find(|facts| facts.name == name)
    }
}

fn tensor_by_name<'a>(
    reader: &'a ModelReader,
    name: &str,
) -> Option<(&'a TensorDescriptor, usize)> {
    reader
        .tensors()
        .iter()
        .enumerate()
        .find(|(_, descriptor)| {
            reader
                .metadata
                .tensor_names
                .get(descriptor.name_ref as usize)
                .map(String::as_str)
                == Some(name)
        })
        .map(|(position, descriptor)| (descriptor, position))
}

fn provenance_entries(reader: &ModelReader) -> Option<Vec<Value>> {
    reader
        .metadata
        .value
        .get("user")
        .and_then(|user| user.get("source_tensors"))
        .and_then(Value::as_array)
        .cloned()
}

fn as_u64_pair(value: &Value, field: &str, name: &str) -> Result<[u64; 2], String> {
    let array = value
        .as_array()
        .ok_or_else(|| format!("{name}: {field} must be an array of two unsigned integers"))?;
    if array.len() != 2 {
        return Err(format!("{name}: {field} must have exactly two entries"));
    }
    let mut dims = [0_u64; 2];
    for (index, entry) in array.iter().enumerate() {
        dims[index] = entry
            .as_u64()
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("{name}: {field} entries must be positive integers"))?;
    }
    Ok(dims)
}

fn physical_shape(descriptor: &TensorDescriptor) -> Vec<u64> {
    let rank = usize::from(descriptor.rank);
    descriptor.dimensions[..rank].to_vec()
}

/// Parse the two orientation tokens of an entry, when it carries them.
///
/// The importer emits the tokens for every source tensor, including rank-1 and
/// rank-3+ tensors whose layout the tokens cannot describe. Their *values*
/// still belong to the contract, so this validates them without pretending a
/// non-rank-2 tensor has a derivable logical shape.
fn parse_orientation_tokens(
    name: &str,
    orientation_token: Option<&str>,
    source_token: Option<&str>,
) -> Result<Option<(WeightOrientation, WeightOrientationSource)>, String> {
    let (Some(orientation_token), Some(source_token)) = (orientation_token, source_token) else {
        if orientation_token.is_some() || source_token.is_some() {
            return Err(format!(
                "{name}: weight_orientation and orientation_source must appear together"
            ));
        }
        return Ok(None);
    };
    let orientation = WeightOrientation::parse(orientation_token).map_err(|_| {
        format!("{name}: unknown weight_orientation {orientation_token:?}; expected 'linear_out_in' or 'conv1d_in_out'")
    })?;
    let source = WeightOrientationSource::parse(source_token).map_err(|_| {
        format!("{name}: unknown orientation_source {source_token:?}; expected 'calibration' or 'legacy_default'")
    })?;
    if source == WeightOrientationSource::LegacyDefault
        && orientation != WeightOrientation::LinearOutIn
    {
        return Err(format!(
            "{name}: orientation_source 'legacy_default' is only consistent with 'linear_out_in'"
        ));
    }
    Ok(Some((orientation, source)))
}

/// Whether a role, an orientation and an index axis form a legal combination.
///
/// A boolean wrapper for [`check_role_layout`], so a producer can refuse to
/// emit a self-contradictory artifact without carrying the message text.
pub(crate) fn role_layout_is_consistent(
    role: Option<TensorRole>,
    orientation: Option<WeightOrientation>,
    index_axis: Option<usize>,
) -> bool {
    check_role_layout("", role, orientation, index_axis).is_ok()
}

/// A named reason for rejecting a provenance or manifest entry.
///
/// The codes exist so the two auditors can be compared case by case: Rust and
/// Python implement this table independently (no shared code, no calling one
/// from the other), and a case one of them rejects, the other rejects **for the
/// same code**. `tests/fixtures/embedding_layout_cases.json` is the shared
/// table both test suites read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutViolation {
    /// A rank-2 source entry states no `weight_orientation`. v0.1 requires the
    /// grouping token for every matrix because the encoder must record which
    /// axis it grouped; it is never evidence that the tensor is a projection.
    OrientationRequired,
    /// A projection role is declared without any orientation token.
    RoleWithoutOrientation,
    /// A projection role is declared with the other projection's orientation.
    RoleOrientationMismatch,
    /// `index_axis` appears with no `tensor_role` to make it meaningful.
    AxisWithoutRole,
    /// `index_axis` appears on a role that is not an indexed table.
    AxisForNonIndexedRole,
    /// `embedding_table` without `index_axis`: which axis is indexed is unknown.
    TableWithoutAxis,
    /// `index_axis` is not 0 or 1.
    AxisValueUnsupported,
    /// `index_axis=1` has no producer path in v0.1: it needs the transposed
    /// grouping, whose orientation token only pairs with calibration
    /// provenance, and an indexed table must not carry activation statistics.
    AxisOneUnsupported,
    /// The recorded orientation groups a different axis than the declared index
    /// axis implies.
    TableOrientationMismatch,
    /// `logical_shape` is not the source array arranged by the declared axis.
    LogicalShapeConflictsAxis,
    /// `logical_shape` contradicts the recorded orientation token.
    LogicalShapeConflictsOrientation,
    /// The stored selector payload is not the size the logical shape requires.
    SelectorBytesMismatch,
    /// The logical shape cannot be encoded at the VQ group size at all.
    SelectorLayoutUnencodable,
    /// A quantized entry states no `logical_shape`, so the stored selector
    /// payload has nothing to be checked against.
    LogicalShapeRequired,
    /// Activation statistics attached to an indexed table.
    TableMustNotBeCalibrated,
    /// `tensor_role` is not one of the five documented values.
    RoleTokenUnknown,
    /// `logical_shape` appears on a tensor that is not a matrix.
    LogicalShapeOnNonMatrix,
    /// An indexed-table or projection role claims a layout for a non-matrix.
    MatrixRoleOnNonMatrix,
}

impl LayoutViolation {
    pub fn code(self) -> &'static str {
        match self {
            LayoutViolation::OrientationRequired => "orientation_required",
            LayoutViolation::RoleWithoutOrientation => "role_without_orientation",
            LayoutViolation::RoleOrientationMismatch => "role_orientation_mismatch",
            LayoutViolation::AxisWithoutRole => "axis_without_role",
            LayoutViolation::AxisForNonIndexedRole => "axis_for_non_indexed_role",
            LayoutViolation::TableWithoutAxis => "table_without_axis",
            LayoutViolation::AxisValueUnsupported => "axis_value_unsupported",
            LayoutViolation::AxisOneUnsupported => "axis_one_unsupported",
            LayoutViolation::TableOrientationMismatch => "table_orientation_mismatch",
            LayoutViolation::LogicalShapeConflictsAxis => "logical_shape_conflicts_axis",
            LayoutViolation::LogicalShapeConflictsOrientation => {
                "logical_shape_conflicts_orientation"
            }
            LayoutViolation::SelectorBytesMismatch => "selector_bytes_mismatch",
            LayoutViolation::SelectorLayoutUnencodable => "selector_layout_unencodable",
            LayoutViolation::LogicalShapeRequired => "logical_shape_required",
            LayoutViolation::TableMustNotBeCalibrated => "table_must_not_be_calibrated",
            LayoutViolation::RoleTokenUnknown => "role_token_unknown",
            LayoutViolation::LogicalShapeOnNonMatrix => "logical_shape_on_non_matrix",
            LayoutViolation::MatrixRoleOnNonMatrix => "matrix_role_on_non_matrix",
        }
    }

    /// The reason, spelled out. `subject` names the tensor so a report line stays
    /// readable; the code comes first so two implementations can be diffed.
    pub fn render(self, subject: &str, detail: &str) -> String {
        format!("{subject}: {}: {detail}", self.code())
    }
}

/// Where a layout check applies. A manifest entry states what a model
/// definition requires; a source entry states what a container encoded. Only
/// the second must record a grouping token, because only the second is a fact
/// about written bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutContext {
    Manifest,
    SourceEntry,
}

/// Everything one entry claims about one tensor, as data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryLayout {
    pub source_shape: Vec<u64>,
    pub orientation: Option<WeightOrientation>,
    pub logical_shape: Option<[u64; 2]>,
    pub role: Option<TensorRole>,
    pub index_axis: Option<usize>,
    pub context: LayoutContext,
    /// The tensor was quantized, so selector payload facts apply.
    pub quantized: bool,
    /// Number of codebooks the VQ pair stores, when quantized.
    pub codebook_count: u64,
    /// Bytes of packed selectors actually stored, when quantized.
    pub stored_index_bytes: u64,
    /// The entry carries an activation-calibration object.
    pub calibrated: bool,
}

impl EntryLayout {
    /// A manifest entry, which states no physical facts of its own.
    pub fn manifest(
        source_shape: Vec<u64>,
        orientation: Option<WeightOrientation>,
        role: Option<TensorRole>,
        index_axis: Option<usize>,
    ) -> Self {
        Self {
            source_shape,
            orientation,
            logical_shape: None,
            role,
            index_axis,
            context: LayoutContext::Manifest,
            quantized: false,
            codebook_count: 0,
            stored_index_bytes: 0,
            calibrated: false,
        }
    }
}

/// The normative layout matrix, in one place.
///
/// Rules, in the order they are applied. Every rejection names itself through a
/// [`LayoutViolation`] code so the Python auditor can be diffed against this
/// implementation case by case:
///
/// * a tensor that is not a matrix carries no logical shape and no matrix role;
/// * a `SourceEntry` of rank 2 must state `weight_orientation`, because the
///   encoder has to record which axis it grouped. `other` and `unknown` are not
///   exempt: the token describes the encoding, not the architectural function;
/// * a projection role requires its own orientation token and rejects the other
///   one;
/// * `embedding_table` requires `index_axis`; only axis 0 is supported, the
///   orientation must group the feature axis, `logical_shape` must be the source
///   array with the indexed axis first, and the stored selector payload must be
///   exactly the size that shape requires;
/// * `index_axis` on any other role, or with no role, is a violation;
/// * an indexed table must not carry activation statistics;
/// * a role absent from the vocabulary is reported, never mapped onto a
///   neighboring role.
pub fn validate_entry_layout(entry: &EntryLayout) -> Result<(), (LayoutViolation, String)> {
    let fail = |violation: LayoutViolation, detail: String| Err((violation, detail));
    let rank = entry.source_shape.len();
    // A manifest entry usually states a requirement for a name without restating
    // the shape, so its rank is unknown rather than non-matrix.
    let rank_known = !entry.source_shape.is_empty();
    let matrix = rank == 2;

    if rank_known && !matrix {
        if entry.logical_shape.is_some() {
            return fail(
                LayoutViolation::LogicalShapeOnNonMatrix,
                format!("logical_shape is only derivable for rank-2 tensors, found rank {rank}"),
            );
        }
        if entry.index_axis.is_some() {
            return fail(
                LayoutViolation::AxisForNonIndexedRole,
                format!(
                    "index_axis is only meaningful for embedding_table over a rank-2 tensor, found rank {rank}"
                ),
            );
        }
        if entry
            .role
            .is_some_and(|role| role != TensorRole::Other && role != TensorRole::Unknown)
        {
            return fail(
                LayoutViolation::MatrixRoleOnNonMatrix,
                format!(
                    "role {} describes a matrix layout; only 'other' or 'unknown' can describe a rank-{rank} tensor",
                    entry.role.expect("checked above").as_str()
                ),
            );
        }
        return Ok(());
    }

    // Zero when the rank was not stated (a manifest entry), and every rule that
    // reads the shape is gated on `matrix` below.
    let (rows, cols) = (
        entry.source_shape.first().copied().unwrap_or(0),
        entry.source_shape.get(1).copied().unwrap_or(0),
    );

    // v0.1 still allows an entry to state nothing about a layout at all: that is
    // a pre-contract artifact, reported as `legacy_undeclared` and interpreted as
    // it always was. What it may not do is declare *half* a layout — a logical
    // shape, an index axis or an indexed table all presuppose the token that
    // records which axis was actually grouped.
    if entry.context == LayoutContext::SourceEntry
        && entry.orientation.is_none()
        && (entry.logical_shape.is_some()
            || entry.index_axis.is_some()
            || entry.role == Some(TensorRole::EmbeddingTable))
    {
        return fail(
            LayoutViolation::OrientationRequired,
            "a rank-2 source entry declares logical_shape, index_axis or embedding_table without \
             weight_orientation: v0.1 requires the grouping token whenever any part of a layout is \
             stated, for any role. The token is a fact about the encoding, never evidence that the \
             tensor is a projection"
                .to_owned(),
        );
    }

    if entry.context == LayoutContext::SourceEntry
        && entry.orientation.is_some()
        && entry.logical_shape.is_none()
    {
        return fail(
            LayoutViolation::LogicalShapeRequired,
            "logical_shape is required alongside weight_orientation: the declared layout has \
             to state the shape it arranges the source array into"
                .to_owned(),
        );
    }

    if let Some(role) = entry.role {
        let required = match role {
            TensorRole::LinearProjection => Some(WeightOrientation::LinearOutIn),
            TensorRole::Conv1dProjection => Some(WeightOrientation::Conv1DInOut),
            TensorRole::EmbeddingTable | TensorRole::Other | TensorRole::Unknown => None,
        };
        if let Some(required) = required {
            match entry.orientation {
                None => {
                    return fail(
                        LayoutViolation::RoleWithoutOrientation,
                        format!(
                            "role {} requires weight_orientation {:?}",
                            role.as_str(),
                            required.as_str()
                        ),
                    );
                }
                Some(found) if found != required => {
                    return fail(
                        LayoutViolation::RoleOrientationMismatch,
                        format!(
                            "role {} requires weight_orientation {:?}, found {:?}",
                            role.as_str(),
                            required.as_str(),
                            found.as_str()
                        ),
                    );
                }
                Some(_) => {}
            }
        }
        if role != TensorRole::EmbeddingTable && entry.index_axis.is_some() {
            return fail(
                LayoutViolation::AxisForNonIndexedRole,
                format!(
                    "index_axis is only meaningful for embedding_table, not {}",
                    role.as_str()
                ),
            );
        }
    } else if entry.index_axis.is_some() {
        return fail(
            LayoutViolation::AxisWithoutRole,
            "index_axis is only meaningful alongside a tensor_role".to_owned(),
        );
    }

    if entry.role == Some(TensorRole::EmbeddingTable) {
        if entry.calibrated {
            return fail(
                LayoutViolation::TableMustNotBeCalibrated,
                "an embedding table is indexed, so it has no input activations; \
                 diagonal_second_moment does not describe it and must not be attached to it"
                    .to_owned(),
            );
        }
        let axis = entry.index_axis.ok_or_else(|| {
            (
                LayoutViolation::TableWithoutAxis,
                "role embedding_table requires index_axis 0 or 1".to_owned(),
            )
        })?;
        if axis > 1 {
            return fail(
                LayoutViolation::AxisValueUnsupported,
                format!("index_axis must be 0 or 1, found {axis}"),
            );
        }
        if axis == 1 {
            return fail(
                LayoutViolation::AxisOneUnsupported,
                "an embedding table indexed on axis 1 groups its features along the source \
                 rows, which v0.1 can only record as conv1d_in_out with calibration \
                 provenance; and an indexed table must not carry calibration. No producer \
                 path can write this combination, so it is rejected rather than guessed"
                    .to_owned(),
            );
        }
        if let Some(orientation) = entry.orientation {
            if orientation != WeightOrientation::LinearOutIn {
                return fail(
                    LayoutViolation::TableOrientationMismatch,
                    format!(
                        "an embedding table indexed on axis 0 keeps features on axis 1, so the grouped axis is the source columns and the recorded orientation must be {:?}, found {:?}",
                        WeightOrientation::LinearOutIn.as_str(),
                        orientation.as_str()
                    ),
                );
            }
        }
        if matrix {
            if let Some(logical) = entry.logical_shape {
                if logical != [rows, cols] {
                    return fail(
                        LayoutViolation::LogicalShapeConflictsAxis,
                        format!(
                            "logical_shape {logical:?} is not the source array {rows}x{cols} with index axis 0 first"
                        ),
                    );
                }
            }
        }
        if entry.quantized {
            let expected = index_payload_bytes(rows, cols, entry.codebook_count).ok_or_else(
                || {
                    (
                        LayoutViolation::SelectorLayoutUnencodable,
                        format!(
                            "logical shape {rows}x{cols} is not encodable at group size {GROUP_SIZE} with {} codebook(s)",
                            entry.codebook_count
                        ),
                    )
                },
            )?;
            if expected != entry.stored_index_bytes {
                return fail(
                    LayoutViolation::SelectorBytesMismatch,
                    format!(
                        "an indexed table with logical shape {rows}x{cols} stores {expected} selector bytes, the container stores {}",
                        entry.stored_index_bytes
                    ),
                );
            }
        }
        return Ok(());
    }

    // A source entry that states an orientation must state the shape that
    // orientation arranges, and the two must agree with the source array. An
    // entry that states no orientation at all is a pre-contract declaration and
    // stays legal here: it is reported as `legacy_undeclared`, never rewritten.
    if entry.context == LayoutContext::SourceEntry {
        if let Some(orientation) = entry.orientation {
            if let Some(logical) = entry.logical_shape {
                let expected = orientation
                    .logical_shape(&entry.source_shape)
                    .ok_or_else(|| {
                        (
                            LayoutViolation::LogicalShapeConflictsOrientation,
                            format!(
                                "cannot derive a logical shape from {:?}",
                                entry.source_shape
                            ),
                        )
                    })?;
                if logical != expected {
                    return fail(
                        LayoutViolation::LogicalShapeConflictsOrientation,
                        format!(
                            "logical_shape {logical:?} contradicts {} over source_shape {expected:?}",
                            orientation.as_str()
                        ),
                    );
                }
            }
        }
    }
    Ok(())
}

/// Enforce the role / orientation / index-axis matrix.
///
/// One function governs both a manifest entry and a container's provenance
/// entry, so a producer, an auditor and a model definition cannot drift into
/// three different ideas of what a role is allowed to claim. A manifest states
/// no physical facts, so it is checked in [`LayoutContext::Manifest`].
fn check_role_layout(
    name: &str,
    role: Option<TensorRole>,
    orientation: Option<WeightOrientation>,
    index_axis: Option<usize>,
) -> Result<(), String> {
    let entry = EntryLayout::manifest(Vec::new(), orientation, role, index_axis);
    validate_entry_layout(&entry).map_err(|(violation, detail)| violation.render(name, &detail))
}

/// Validate the calibration object an entry carries.
///
/// A calibration object does *not* imply `orientation_source: calibration`:
/// activation moments can be captured without the module type, and in that case
/// the importer still has to assume the legacy orientation. What must hold is
/// that the object and the entry describe the same orientation and the same
/// provenance, and that the moment count equals the input features of the
/// orientation that was actually encoded.
fn validate_calibration(
    name: &str,
    disposition: &str,
    calibration: &Value,
    orientation: Option<WeightOrientation>,
    source: Option<WeightOrientationSource>,
    role: Option<TensorRole>,
    source_shape: &[u64],
) -> Result<(), String> {
    if !calibration.is_object() {
        return Err(format!("{name}: calibration must be an object"));
    }
    if disposition == RAW_DISPOSITION {
        return Err(format!(
            "{name}: a RAW-preserved tensor must not carry calibration"
        ));
    }
    if role == Some(TensorRole::EmbeddingTable) {
        return Err(format!(
            "{name}: an embedding table is indexed, so it has no input activations; \
             the diagonal_second_moment statistic does not describe it and must not be \
             attached to it"
        ));
    }
    let calibration_orientation = calibration
        .get("weight_orientation")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    if let Some(orientation) = orientation {
        if calibration_orientation != orientation.as_str() {
            return Err(format!(
                "{name}: calibration weight_orientation {calibration_orientation:?} disagrees with the declared {:?}",
                orientation.as_str()
            ));
        }
    }
    let calibration_source = calibration
        .get("orientation_source")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    if let Some(source) = source {
        if calibration_source != source.as_str() {
            return Err(format!(
                "{name}: calibration orientation_source {calibration_source:?} disagrees with the declared {:?}",
                source.as_str()
            ));
        }
    }
    if let (Some(orientation), [_, _]) = (orientation, source_shape) {
        let features = calibration
            .get("features")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{name}: calibration lacks integer `features`"))?;
        let expected = orientation
            .in_features(source_shape)
            .ok_or_else(|| format!("{name}: cannot derive input features from {source_shape:?}"))?;
        if features != expected {
            return Err(format!(
                "{name}: calibration features {features} do not match the {expected} input features of {} over {source_shape:?}",
                orientation.as_str()
            ));
        }
    }
    Ok(())
}

/// Expected index-payload size for a logical `[rows, cols]` VQ pair.
fn index_payload_bytes(rows: u64, cols: u64, codebook_count: u64) -> Option<u64> {
    let groups = cols.checked_div(GROUP_SIZE as u64)?;
    if groups == 0 || cols % GROUP_SIZE as u64 != 0 {
        return None;
    }
    let selectors = groups.checked_mul(codebook_count)?;
    if selectors % 2 != 0 {
        return None;
    }
    rows.checked_mul(selectors.checked_div(2)?)
}

/// Tie the declared logical shape to the physical tensors in the container.
///
/// A `[rows, cols]` VQ pair whose packed selector payload does not match the
/// stored bytes cannot have been produced by that pair, so this check makes
/// some rectangular orientation claims falsifiable from the container alone —
/// including every declaration that is not encodable at the group size.
///
/// It is not a complete decision procedure: the selector byte count is
/// `rows * (cols / GROUP_SIZE * codebooks / 2)`, and a transpose can give the
/// same product (a `16x32` logical matrix and a `32x16` one both store 32
/// bytes at one codebook). Square matrices always collide, and so do those
/// rectangular pairs. Whatever this check cannot falsify stays unverified:
/// confirming it needs the model definition or a calibration tied to the
/// tensor, never a guess from the name.
fn check_physical_layout(
    reader: &ModelReader,
    name: &str,
    logical: [u64; 2],
) -> Result<bool, String> {
    let (codebook, _) = tensor_by_name(reader, &format!("{name}{VQ_CODEBOOK_SUFFIX}"))
        .ok_or_else(|| format!("{name}: quantized tensor has no codebook tensor"))?;
    let (indices, _) = tensor_by_name(reader, &format!("{name}{VQ_INDICES_SUFFIX}"))
        .ok_or_else(|| format!("{name}: quantized tensor has no index tensor"))?;
    if codebook.rank != 3 || codebook.dtype != DType::F32 {
        return Err(format!(
            "{name}: codebook tensor is not a rank-3 F32 tensor"
        ));
    }
    let codebook_count = codebook.dimensions[0];
    if !(1..=4).contains(&codebook_count)
        || codebook.dimensions[1] != CODEBOOK_SIZE as u64
        || codebook.dimensions[2] != GROUP_SIZE as u64
    {
        return Err(format!(
            "{name}: codebook tensor dimensions do not match the VQ layout"
        ));
    }
    if indices.rank != 1 || indices.dtype != DType::Opaque {
        return Err(format!(
            "{name}: index tensor is not an opaque rank-1 tensor"
        ));
    }
    let expected = index_payload_bytes(logical[0], logical[1], codebook_count)
        .ok_or_else(|| format!("{name}: logical shape is not encodable at the VQ group size"))?;
    if indices.logical_nbytes != expected || indices.dimensions[0] != expected {
        return Err(format!(
            "{name}: declared logical shape {:?} needs {expected} selector bytes but the container stores {}",
            [logical[0], logical[1]],
            indices.logical_nbytes
        ));
    }
    Ok(true)
}

/// Selector payload facts of a VQ pair, when the pair is well formed enough to
/// read them. Absent means "let `check_physical_layout` report why".
fn vq_selector_facts(reader: &ModelReader, name: &str) -> Option<(u64, u64)> {
    let (codebook, _) = tensor_by_name(reader, &format!("{name}{VQ_CODEBOOK_SUFFIX}"))?;
    let (indices, _) = tensor_by_name(reader, &format!("{name}{VQ_INDICES_SUFFIX}"))?;
    if codebook.rank != 3 || indices.rank != 1 {
        return None;
    }
    Some((codebook.dimensions[0], indices.logical_nbytes))
}

/// Audit one declared source tensor entry. Returns `Err` with the finding when
/// the entry is structurally inconsistent or contradicts the architecture.
fn audit_entry(
    reader: &ModelReader,
    entry: &Value,
    facts: Option<&ArchitectureFacts>,
) -> Result<TensorOrientationFacts, String> {
    let name = entry
        .get("name")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| "provenance entry lacks a string `name`".to_owned())?
        .to_owned();
    let disposition = entry
        .get("disposition")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    if disposition != QUANTIZE_DISPOSITION && disposition != RAW_DISPOSITION {
        return Err(format!(
            "{name}: disposition must be '{QUANTIZE_DISPOSITION}' or '{RAW_DISPOSITION}', found {disposition:?}"
        ));
    }
    let declared_orientation = entry.get("weight_orientation").and_then(Value::as_str);
    let declared_source = entry.get("orientation_source").and_then(Value::as_str);
    let logical_value = entry.get("logical_shape");
    let role_token = entry.get("tensor_role").and_then(Value::as_str);
    let role = match role_token {
        None => None,
        Some(token) => Some(
            TensorRole::parse_token(token)
                .map_err(|(violation, detail)| violation.render(&name, &detail))?,
        ),
    };
    let index_axis = match entry.get("index_axis") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let axis = value
                .as_u64()
                .ok_or_else(|| format!("{name}: index_axis must be the integer 0 or 1"))?;
            if axis > 1 {
                return Err(format!("{name}: index_axis must be 0 or 1, found {axis}"));
            }
            Some(axis as usize)
        }
    };

    // Only rank-2 matrices carry orientation semantics.
    let container_tensor = tensor_by_name(reader, &name).map(|(descriptor, _)| descriptor);
    let source_shape_value = entry.get("source_shape");
    let source_shape: Vec<u64> = match source_shape_value {
        Some(value) => {
            let parsed = value
                .as_array()
                .ok_or_else(|| format!("{name}: source_shape must be an array"))?
                .iter()
                .map(|item| {
                    item.as_u64().filter(|value| *value > 0).ok_or_else(|| {
                        format!("{name}: source_shape entries must be positive integers")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(descriptor) = container_tensor.filter(|_| disposition == RAW_DISPOSITION) {
                let physical = physical_shape(descriptor);
                if physical != parsed {
                    return Err(format!(
                        "{name}: source_shape {parsed:?} does not match the stored tensor shape {physical:?}"
                    ));
                }
            }
            parsed
        }
        None => match container_tensor {
            Some(descriptor) => physical_shape(descriptor),
            None => {
                return Err(format!(
                    "{name}: no source_shape and no tensor of that name in the container"
                ));
            }
        },
    };

    let declared = parse_orientation_tokens(&name, declared_orientation, declared_source)?;
    let logical: Option<[u64; 2]> = match logical_value {
        None => None,
        Some(value) => Some(as_u64_pair(value, "logical_shape", &name)?),
    };
    let (orientation, source) = match declared {
        Some((orientation, source)) => (Some(orientation), Some(source)),
        None => (None, None),
    };
    let quantized = disposition == QUANTIZE_DISPOSITION;
    let calibrated = entry.get("calibration").is_some();
    let (codebook_count, stored_index_bytes) = if quantized && source_shape.len() == 2 {
        vq_selector_facts(reader, &name).unwrap_or((0, 0))
    } else {
        (0, 0)
    };
    // One table of rules, one named code per rejection. The Python auditor
    // implements the same table independently, and both suites read
    // tests/fixtures/embedding_layout_cases.json, so a case one rejects, the
    // other rejects for the same code.
    validate_entry_layout(&EntryLayout {
        source_shape: source_shape.clone(),
        orientation,
        logical_shape: logical,
        role,
        index_axis,
        context: LayoutContext::SourceEntry,
        quantized,
        codebook_count,
        stored_index_bytes,
        calibrated,
    })
    .map_err(|(violation, detail)| violation.render(&name, &detail))?;
    let square = source_shape.len() == 2 && source_shape[0] == source_shape[1];
    let mut role_assurance = if role.is_some() {
        RoleAssurance::DeclaredNotConfirmed
    } else {
        RoleAssurance::NotDeclared
    };

    let (Some(orientation), Some(source)) = (orientation, source) else {
        // Nothing about the layout was declared. The entry stays readable and
        // unverified, exactly as a pre-contract artifact always has been.
        return Ok(TensorOrientationFacts {
            name,
            disposition,
            source_shape,
            declared: None,
            source: None,
            logical_shape: None,
            square,
            assurance: OrientationAssurance::LegacyUndeclared,
            architecture: None,
            physical_layout_consistent: false,
            role,
            index_axis,
            role_assurance,
            architecture_roles: Vec::new(),
            architecture_aliases: Vec::new(),
            role_evidence: Vec::new(),
        });
    };
    let orientation_token = orientation.as_str();

    if let Some(calibration) = entry.get("calibration") {
        validate_calibration(
            &name,
            &disposition,
            calibration,
            Some(orientation),
            Some(source),
            role,
            &source_shape,
        )?;
    } else if source == WeightOrientationSource::Calibration {
        return Err(format!(
            "{name}: orientation_source 'calibration' requires a calibration object"
        ));
    }

    let physical_layout_consistent = if disposition == QUANTIZE_DISPOSITION {
        check_physical_layout(
            reader,
            &name,
            logical.ok_or_else(|| {
                format!("{name}: logical_shape is required alongside weight_orientation")
            })?,
        )?
    } else {
        true
    };

    let architecture = facts.and_then(|facts| facts.expectation(&name));
    let mut architecture_roles = Vec::new();
    let mut architecture_aliases = Vec::new();
    let mut role_evidence = Vec::new();
    let assurance = match architecture {
        Some(expectation) => {
            // Role and layout are checked as the two separate claims they are.
            // A manifest that states a role must not contradict the recorded
            // one, and a stated layout must not contradict the orientation the
            // container encodes. An entry that states only a role confirms the
            // role; it says nothing that could verify an orientation, and the
            // orientation an embedding records stays an encoding fact.
            architecture_roles.clone_from(&expectation.roles);
            architecture_aliases.clone_from(&expectation.aliases);
            role_evidence.clone_from(&expectation.role_evidence);
            if !expectation.roles.is_empty() {
                if let Some(declared_role) = role {
                    if expectation.roles.contains(&declared_role) {
                        role_assurance = RoleAssurance::Confirmed;
                    } else {
                        return Err(format!(
                            "{name}: declared role {} but the model definition evidences only [{}], refusing to treat the container as valid",
                            declared_role.as_str(),
                            expectation
                                .roles
                                .iter()
                                .map(|role| role.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ));
                    }
                }
            }
            let expected_orientation = expectation.effective_orientation();
            if let Some(expected_orientation) = expected_orientation {
                if expected_orientation != orientation {
                    return Err(format!(
                        "{name}: declared {orientation_token} but the model definition requires {}; refusing to treat the container as valid",
                        expected_orientation.as_str()
                    ));
                }
            }
            if let Some(shape) = expectation.source_shape {
                if [shape[0], shape[1]] != [source_shape[0], source_shape[1]] {
                    return Err(format!(
                        "{name}: declared source_shape {source_shape:?} but the model definition requires {:?}",
                        [shape[0], shape[1]]
                    ));
                }
            }
            match expected_orientation {
                // The layout was compared against something outside the
                // container, so it may be reported as confirmed.
                Some(_) => OrientationAssurance::ArchitecturallyConfirmed,
                // A manifest that states only a role says nothing about the
                // layout. Confirming a role must not lend the orientation an
                // assurance it never earned.
                None if square => OrientationAssurance::SquareNotVerifiable,
                None => OrientationAssurance::DeclaredNotVerified,
            }
        }
        None if square => OrientationAssurance::SquareNotVerifiable,
        None => OrientationAssurance::DeclaredNotVerified,
    };

    Ok(TensorOrientationFacts {
        name,
        disposition,
        source_shape,
        declared: Some(orientation),
        source: Some(source),
        logical_shape: logical,
        square,
        assurance,
        // The layout the model definition states outright. A layout entailed by
        // an indexing role is reported through `role`/`index_axis`, not here, so
        // the two dimensions never collapse into one claim.
        architecture: architecture.and_then(|item| item.orientation),
        physical_layout_consistent,
        role,
        index_axis,
        role_assurance,
        architecture_roles,
        architecture_aliases,
        role_evidence,
    })
}

/// Audit every source tensor the container declares, plus the rank-2 tensors it
/// does not declare at all.
///
/// Structural problems are collected in `invalid` rather than aborting, so an
/// operator sees every affected tensor in one run. Contradictions with the
/// supplied architecture are among those findings.
pub fn audit_container(
    reader: &ModelReader,
    facts: Option<&ArchitectureFacts>,
) -> OrientationAudit {
    let mut tensors = Vec::new();
    let mut invalid = Vec::new();
    let mut described: Vec<String> = Vec::new();

    if let Some(entries) = provenance_entries(reader) {
        for entry in &entries {
            let entry_name = entry.get("name").and_then(Value::as_str);
            match entry_name {
                Some(entry_name) if !entry_name.trim().is_empty() => {
                    if described.iter().any(|known| known == entry_name) {
                        invalid.push(format!(
                            "{entry_name}: user.source_tensors declares the same tensor more than once"
                        ));
                        continue;
                    }
                    described.push(entry_name.to_owned());
                }
                _ => {
                    invalid.push("provenance entry lacks a string `name`".to_owned());
                    continue;
                }
            }
            match audit_entry(reader, entry, facts) {
                Ok(orientation) if orientation.source_shape.len() == 2 => {
                    tensors.push(orientation);
                }
                Ok(_) => {}
                Err(finding) => invalid.push(finding),
            }
        }
    } else if reader
        .metadata
        .value
        .get("user")
        .and_then(|user| user.get("source_tensors"))
        .is_some()
    {
        invalid.push("user.source_tensors is present but is not an array".to_owned());
    }

    let mut undeclared_rank2_tensors = 0_usize;
    for descriptor in reader.tensors() {
        // Only rank-2 tensors carry orientation semantics; the VQ companions of
        // a quantized matrix are rank-3 and rank-1 and drop out here on shape
        // alone, without looking at their names.
        if usize::from(descriptor.rank) != 2 {
            continue;
        }
        let Some(name) = reader
            .metadata
            .tensor_names
            .get(descriptor.name_ref as usize)
        else {
            continue;
        };
        if described.iter().any(|known| known == name) {
            continue;
        }
        // An undescribed tensor is not a contract violation: it is a matrix
        // whose orientation the container never states. It is reported as
        // undeclared so that `verify` stays usable for pre-contract artifacts
        // while no consumer can treat the layout as established.
        undeclared_rank2_tensors += 1;
        let shape = physical_shape(descriptor);
        tensors.push(TensorOrientationFacts {
            name: name.clone(),
            disposition: "undeclared".to_owned(),
            source_shape: shape.clone(),
            declared: None,
            source: None,
            logical_shape: None,
            square: shape.len() == 2 && shape[0] == shape[1],
            assurance: OrientationAssurance::LegacyUndeclared,
            architecture: None,
            physical_layout_consistent: false,
            role: None,
            index_axis: None,
            role_assurance: RoleAssurance::NotDeclared,
            architecture_roles: Vec::new(),
            architecture_aliases: Vec::new(),
            role_evidence: Vec::new(),
        });
    }

    let rank2_tensors = tensors
        .iter()
        .filter(|facts| facts.source_shape.len() == 2)
        .count();
    let roles_declared = tensors.iter().filter(|facts| facts.role.is_some()).count();
    let roles_confirmed = tensors
        .iter()
        .filter(|facts| facts.role_assurance == RoleAssurance::Confirmed)
        .count();
    let indexed_tables = tensors
        .iter()
        .filter(|facts| facts.role == Some(TensorRole::EmbeddingTable))
        .count();
    OrientationAudit {
        tensors,
        invalid,
        rank2_tensors,
        undeclared_rank2_tensors,
        roles_declared,
        roles_confirmed,
        indexed_tables,
        architecture_supplied: facts.is_some(),
    }
}

/// A VQ tensor pair whose layout was derived from validated metadata.
#[derive(Debug, Clone)]
pub struct ResolvedVqTensor {
    pub layout: HpsiVqLayout,
    pub facts: TensorOrientationFacts,
}

impl ResolvedVqTensor {
    pub fn logical_shape(&self) -> [u64; 2] {
        self.facts
            .logical_shape
            .expect("resolved tensors always carry a logical shape")
    }
}

/// Resolve the executor layout for one source tensor from its metadata alone.
///
/// Consumers must use this instead of passing `rows` and `cols` by hand: the
/// returned layout is tied both to the declared orientation and to the selector
/// payload actually stored. A tensor whose declared orientation contradicts the
/// supplied architecture is always rejected. A tensor that is merely unverified
/// (square matrices without architectural information, or legacy provenance) is
/// rejected unless the caller explicitly accepts unverified orientation, which is
/// how "requires confirmation" is expressed in this contract.
pub fn resolve_vq_layout(
    reader: &ModelReader,
    name: &str,
    facts: Option<&ArchitectureFacts>,
    allow_unverified: bool,
) -> Result<ResolvedVqTensor, OrientationError> {
    let entries = provenance_entries(reader).ok_or_else(|| {
        OrientationError(format!(
            "{name}: container carries no user.source_tensors provenance; orientation is undeclared and a layout cannot be derived from metadata"
        ))
    })?;
    let entry = entries
        .iter()
        .find(|entry| entry.get("name").and_then(Value::as_str) == Some(name))
        .ok_or_else(|| {
            OrientationError(format!(
                "{name}: container does not declare provenance for this tensor"
            ))
        })?;
    let orientation = audit_entry(reader, entry, facts).map_err(OrientationError)?;
    if orientation.disposition != QUANTIZE_DISPOSITION {
        return Err(OrientationError(format!(
            "{name}: tensor was preserved RAW and has no VQ layout to resolve"
        )));
    }
    // A role that is positively known not to be a projection closes this path,
    // and no acceptance flag can open it: an indexed table is looked up, not
    // multiplied, so serving it through matvec would answer a different question
    // than the model asks. `unknown` is deliberately not refused here — it means
    // the role was never established, so it stays subject to the ordinary
    // unverified gate below, which is the explicit acceptance mechanism.
    if matches!(
        orientation.role,
        Some(TensorRole::EmbeddingTable) | Some(TensorRole::Other)
    ) {
        return Err(OrientationError(format!(
            "{name}: role {} is not a projection, so the VQ matvec layout does not apply to it \
             and no lookup kernel is implemented; preserve it RAW or reconstruct it densely",
            orientation.role_token()
        )));
    }
    let Some([rows, cols]) = orientation.logical_shape else {
        return Err(OrientationError(format!(
            "{name}: no logical shape was declared"
        )));
    };
    if !orientation.assurance.is_confirmed() && !allow_unverified {
        return Err(OrientationError(format!(
            "{name}: orientation {} is {} ({}); refusing to reconstruct without architectural confirmation. Supply --architecture <manifest.json>, or --allow-unverified to accept the declared orientation",
            orientation.orientation_token(),
            orientation.assurance.as_str(),
            if orientation.square {
                "square matrix: both orientations give the same layout"
            } else {
                "no architectural information supplied"
            }
        )));
    }
    let (codebook, _) = tensor_by_name(reader, &format!("{name}{VQ_CODEBOOK_SUFFIX}"))
        .ok_or_else(|| OrientationError(format!("{name}: codebook tensor is missing")))?;
    let (indices, _) = tensor_by_name(reader, &format!("{name}{VQ_INDICES_SUFFIX}"))
        .ok_or_else(|| OrientationError(format!("{name}: index tensor is missing")))?;
    let codebook_count = usize::try_from(codebook.dimensions[0])
        .map_err(|_| OrientationError(format!("{name}: codebook count exceeds address space")))?;
    let rows = usize::try_from(rows)
        .map_err(|_| OrientationError(format!("{name}: row count exceeds address space")))?;
    let cols = usize::try_from(cols)
        .map_err(|_| OrientationError(format!("{name}: column count exceeds address space")))?;
    Ok(ResolvedVqTensor {
        layout: HpsiVqLayout {
            codebook_tensor_id: codebook.tensor_id,
            indices_tensor_id: indices.tensor_id,
            rows,
            cols,
            codebook_count,
        },
        facts: orientation,
    })
}
