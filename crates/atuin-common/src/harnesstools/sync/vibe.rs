//! Vibe's transcripts are neither read nor written by Atuin: there is no local copy of a
//! session for it to fast-forward or restore.

use std::path::Path;

use super::{
    AppendOptions, AppendOutcome, Liveness, LocalTip, ReplaceOutcome, SessionSync, SyncError,
};
use crate::harnesstools::rehydrate::RehydrateMessage;
use crate::harnesstools::vibe::Vibe;

const NO_TRANSCRIPT: &str = "Atuin does not read or write Vibe's transcripts";

impl SessionSync for Vibe {
    async fn local_tip(&self, _id: &str) -> Result<Option<LocalTip>, SyncError> {
        Err(SyncError::Unsupported(NO_TRANSCRIPT))
    }

    async fn is_live(&self, _id: &str, _cwd: Option<&Path>) -> Liveness {
        Liveness::Unknown
    }

    async fn append(
        &self,
        _id: &str,
        _base: &LocalTip,
        _lines: &[RehydrateMessage],
        _options: &AppendOptions<'_>,
    ) -> Result<AppendOutcome, SyncError> {
        Err(SyncError::Unsupported(NO_TRANSCRIPT))
    }

    async fn replace(
        &self,
        _id: &str,
        _base: &LocalTip,
        _branch: &[RehydrateMessage],
        _options: &AppendOptions<'_>,
        _backups: &Path,
    ) -> Result<ReplaceOutcome, SyncError> {
        Err(SyncError::Unsupported(NO_TRANSCRIPT))
    }
}
