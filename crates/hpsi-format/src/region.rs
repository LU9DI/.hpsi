use crate::{FormatError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub name: &'static str,
    pub offset: u64,
    pub length: u64,
}

impl Region {
    pub fn end(self) -> Result<u64> {
        self.offset
            .checked_add(self.length)
            .ok_or(FormatError::ArithmeticOverflow("region end"))
    }

    pub fn is_empty(self) -> bool {
        self.length == 0
    }
}

/// Validate bounds, alignment and overlap for all non-empty top-level regions.
pub fn validate_regions(regions: &[Region], file_size: u64) -> Result<()> {
    let mut active: Vec<Region> = regions.iter().copied().filter(|r| !r.is_empty()).collect();

    for region in &active {
        if region.offset < 128 {
            return Err(FormatError::InvalidRegion("region overlaps fixed header"));
        }
        if region.offset % 8 != 0 {
            return Err(FormatError::InvalidRegion(
                "region offset is not 8-byte aligned",
            ));
        }
        if region.end()? > file_size {
            return Err(FormatError::InvalidRegion("region extends beyond file"));
        }
    }

    active.sort_unstable_by_key(|region| region.offset);
    for pair in active.windows(2) {
        if pair[0].end()? > pair[1].offset {
            return Err(FormatError::InvalidRegion("top-level regions overlap"));
        }
    }
    Ok(())
}
