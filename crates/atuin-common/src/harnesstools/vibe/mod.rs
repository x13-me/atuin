//! Mistral's Vibe CLI (`vibe`): installing its hooks and resuming its sessions.
//!
//! Mistral's Vibe CLI (`vibe`): installing its hooks, capturing and resuming its sessions.
//!
//! Vibe keeps its configuration and state in `$VIBE_HOME`, by default `~/.vibe`. It loads
//! `hooks.toml` there (and in a trusted project's `.vibe/`) on each run, and registers MCP
//! servers as `[[mcp_servers]]` tables in its `config.toml` (see [`hooks`] and
//! [`super::mcp`]). Sessions are saved under the home and reopened with `vibe --resume <id>`;
//! Atuin records the commands Vibe runs through the hooks, and reads its transcripts (see
//! [`session`]).

use std::path::PathBuf;

use super::rehydrate::{RehydrateError, RehydrateSession, Rehydrated};
use super::resume::{self, CwdRequirement, ResumeError, ResumePlan, ResumeTarget};
use super::{Harness, InstallHookError};
use crate::dirs::home_dir;
use crate::env::var_nonempty;
use crate::path::PathBufExt as _;

pub mod hooks;
pub mod rehydrate;
pub mod session;

/// Vibe's home directory: `$VIBE_HOME`, else `~/.vibe`.
pub(crate) fn home() -> PathBuf {
    var_nonempty("VIBE_HOME").map_or_else(|| home_dir().joined(".vibe"), PathBuf::from)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Vibe;

impl Harness for Vibe {
    fn name(&self) -> &'static str {
        "vibe"
    }

    fn display_name(&self) -> &'static str {
        "Vibe"
    }

    fn program(&self) -> &'static str {
        "vibe"
    }

    fn home_vars(&self) -> &'static [&'static str] {
        &["VIBE_HOME"]
    }

    /// Vibe reads `hooks.toml` on each run, so the hooks apply from the next one.
    async fn install_hooks(&self) -> Result<PathBuf, InstallHookError> {
        hooks::install(&home(), &std::env::current_exe()?).await
    }

    async fn has_hooks(&self) -> Result<bool, InstallHookError> {
        hooks::has(&home()).await
    }

    async fn uninstall_hooks(&self) -> Result<bool, InstallHookError> {
        hooks::uninstall(&home()).await
    }

    /// `vibe --resume <id>`. Vibe finds the session from anywhere, and works in the directory
    /// it is started from.
    fn resume_plan(&self, target: &ResumeTarget) -> Result<ResumePlan, ResumeError> {
        let args = ["--resume".to_owned(), target.id.clone()];
        Ok(resume::plan(target, CwdRequirement::None, self.program(), args))
    }

    /// The session's store or folder under Vibe's save dir (see [`session`]).
    async fn locate(&self, id: &str) -> Option<PathBuf> {
        let root = session::default_root();
        let id = id.to_owned();
        resume::blocking(move || session::locate(&root, &id)).await
    }

    /// Written out as a legacy session folder under the save dir (see [`rehydrate`]), which
    /// `vibe --resume <id>` still reads.
    async fn rehydrate(&self, session: &RehydrateSession) -> Result<Rehydrated, RehydrateError> {
        rehydrate::rehydrate(session).await.map(Rehydrated::at)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn resumes_by_id_from_anywhere() {
        let target = ResumeTarget::new("01a11e4d-75d7-70f2-8c71-7f1cc87cbe85").with_cwd("/work");
        let plan = Vibe.resume_plan(&target).unwrap();
        assert_eq!(plan.program, "vibe");
        assert_eq!(plan.args, ["--resume", "01a11e4d-75d7-70f2-8c71-7f1cc87cbe85"]);
        assert_eq!(plan.cwd_requirement, CwdRequirement::None);
    }

    #[rstest]
    fn is_found_by_its_name() {
        let harness = super::super::AnyHarness::from_name("vibe").unwrap();
        assert_eq!(harness.name(), "vibe");
    }

    /// The home follows `$VIBE_HOME`, else `~/.vibe`.
    #[rstest]
    #[case::unset(None, ".vibe")]
    #[case::elsewhere(Some("/srv/vibe"), "/srv/vibe")]
    fn home_follows_vibe_home(#[case] var: Option<&str>, #[case] want: &str) {
        let home_dir = tempfile::tempdir().unwrap();
        let env = crate::env::MockEnv::install();
        env.set("HOME", home_dir.path());
        if let Some(var) = var {
            env.set("VIBE_HOME", var);
        }

        assert_eq!(home(), home_dir.path().join(want));
    }
}
