//! Rendering of the shared orientation audit for the `inspect` and `verify`
//! commands.
//!
//! The CLI owns **no** orientation decision logic. Every label and finding here
//! comes from `hpsi_ops::orientation::OrientationAudit`, so the command output
//! and a programmatic consumer of the library cannot drift apart. The library
//! never reports structural consistency as architectural proof, and neither do
//! these renderers: `assurance` distinguishes `architecturally_confirmed` (an
//! externally supplied model definition agreed with the declaration) from
//! `declared_not_verified`, `square_not_verifiable` and `legacy_undeclared`.
//!
//! Nothing here infers an orientation from a tensor name, from a square shape
//! or from iteration order. A square `[n, n]` matrix without an external
//! reference is reported as unverifiable, not as correct.

use hpsi_ops::{OrientationAssurance, OrientationAudit, RoleAssurance, TensorOrientationFacts};

fn shape_token(shape: &[u64]) -> String {
    shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn physical_layout_token(entry: &TensorOrientationFacts) -> &'static str {
    if entry.declared.is_none() {
        "not-declared"
    } else if entry.disposition == "preserve_raw" {
        "not-applicable"
    } else if entry.physical_layout_consistent {
        "consistent"
    } else {
        "inconsistent"
    }
}

fn tensor_line(entry: &TensorOrientationFacts) -> String {
    let logical = match entry.logical_shape {
        Some([rows, cols]) => format!("{rows}x{cols}"),
        None => "<missing>".to_owned(),
    };
    let mut line = format!(
        "orientation tensor={} disposition={} declared={} orientation_source={} source_shape={} logical_shape={} square={} physical_layout={} assurance={} tensor_role={} role_assurance={}",
        entry.name,
        entry.disposition,
        entry.orientation_token(),
        entry.source_token(),
        shape_token(&entry.source_shape),
        logical,
        entry.square,
        physical_layout_token(entry),
        entry.assurance.as_str(),
        entry.role_token(),
        entry.role_assurance.as_str(),
    );
    if let Some(axis) = entry.index_axis {
        line.push_str(&format!(" index_axis={axis}"));
    }
    if let Some(orientation) = entry.architecture {
        line.push_str(&format!(" architecture={}", orientation.as_str()));
    }
    if !entry.architecture_roles.is_empty() {
        line.push_str(&format!(
            " architecture_roles=[{}]",
            entry
                .architecture_roles
                .iter()
                .map(|role| role.as_str())
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    // The pairing only appears when the model definition states it, so an
    // artifact produced without alias evidence prints exactly as before.
    if !entry.role_evidence.is_empty() {
        line.push_str(&format!(
            " role_evidence=[{}]",
            entry
                .role_evidence
                .iter()
                .map(|item| format!(
                    "{}:{}({})",
                    item.alias,
                    item.role.as_str(),
                    item.module_class
                ))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    if !entry.architecture_aliases.is_empty() {
        line.push_str(&format!(
            " architecture_aliases=[{}]",
            entry.architecture_aliases.join(",")
        ));
    }
    line
}

pub fn summary_lines(audit: &OrientationAudit) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!(
        "orientation summary: rank2={} confirmed={} declared_not_verified={} square_not_verifiable={} legacy_undeclared={} undeclared_tensors={} roles_declared={} roles_confirmed={} indexed_tables={} roles_not_declared={}",
        audit.rank2_tensors,
        audit.count(OrientationAssurance::ArchitecturallyConfirmed),
        audit.count(OrientationAssurance::DeclaredNotVerified),
        audit.count(OrientationAssurance::SquareNotVerifiable),
        audit.count(OrientationAssurance::LegacyUndeclared),
        audit.undeclared_rank2_tensors,
        audit.roles_declared,
        audit.roles_confirmed,
        audit.indexed_tables,
        audit.count_role(RoleAssurance::NotDeclared),
    ));
    for entry in &audit.tensors {
        lines.push(tensor_line(entry));
    }
    for finding in &audit.invalid {
        lines.push(format!("orientation finding: {finding}"));
    }
    lines.extend(note_lines(audit));
    lines
}

fn note_lines(audit: &OrientationAudit) -> Vec<String> {
    let mut lines = Vec::new();
    if audit.architecture_supplied {
        lines.push(
            "orientation note: an architecture manifest was supplied; assurance=architecturally_confirmed means the declared orientation agrees with that external model definition, which is evidence about the model, not a property of the container."
                .to_owned(),
        );
        if !audit.has_architecture() {
            lines.push(
                "orientation note: the supplied manifest matched none of the tensor names this container declares; manifest names must match exactly, so nothing here is architecturally confirmed."
                    .to_owned(),
            );
        }
    } else {
        lines.push(
            "orientation note: no architecture manifest supplied, so nothing here is architecturally confirmed. Pass --architecture <manifest.json> to compare the declarations against the model definition."
                .to_owned(),
        );
    }
    if audit.square_unverified() > 0 {
        lines.push(format!(
            "orientation note: {} square rank-2 tensor(s) are identical under both orientations as far as shape and stored bytes go, so a declared orientation is a claim, not a confirmation; an architecture manifest or a verified calibration is required.",
            audit.square_unverified()
        ));
    }
    let several_roles = audit
        .tensors
        .iter()
        .filter(|entry| entry.architecture_roles.len() > 1)
        .count();
    if several_roles > 0 {
        lines.push(format!(
            "orientation note: {} tensor(s) are reached under several architectural roles at once, typically a tied embedding and output head. The container records the role its payload was encoded under, and role_evidence names which alias of the definition carries each remaining role; nothing here chose one function at the other's expense.",
            several_roles
        ));
    }
    if audit.roles_declared == 0 {
        lines.push(
            "orientation note: no tensor_role is recorded anywhere here, so this artifact predates the role contract; its orientation claims are read exactly as documented for legacy artifacts, and nothing in them implies an architectural role."
                .to_owned(),
        );
    } else {
        lines.push(format!(
            "orientation note: {} of {} rank-2 tensor(s) record an architectural role. Role assurance is reported separately from orientation assurance on purpose: a confirmed role is no evidence about a layout and a confirmed layout is no evidence about a role. tensor_role=unknown means the producer could not establish the function, which is not the same claim as tensor_role=other.",
            audit.roles_declared, audit.rank2_tensors
        ));
    }
    if audit.indexed_tables > 0 {
        lines.push(format!(
            "orientation note: {} tensor(s) are recorded as indexed tables. Their weight_orientation states only the layout the quantizer grouped along; the projection executor refuses them by contract, because a lookup is not a matmul and no lookup kernel ships yet.",
            audit.indexed_tables
        ));
    }
    if !audit.invalid.is_empty() {
        lines.push(
            "orientation note: the findings above are orientation-contract violations. CRC32C integrity can pass while they exist: container integrity, metadata self-consistency, architectural compatibility and reconstruction correctness are separate claims."
                .to_owned(),
        );
    }
    lines
}
