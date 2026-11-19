//! Atuin's hooks in Vibe's `hooks.toml`: `$VIBE_HOME/hooks.toml` (by default `~/.vibe/hooks.toml`).
//!
//! Vibe loads `hooks.toml` from a trusted project's `.vibe/` and from the user's home on each
//! run, and runs a `[[hooks]]` entry's `command` with the event as JSON on `stdin` (Vibe's hook
//! contract). The file is the user's, with their own entries to keep, so atuin's two entries are
//! merged in and taken out again by the command they run, leaving everything else as it is.

use std::path::{Path, PathBuf};

use toml_edit::{ArrayOfTables, DocumentMut, Item, Table};

use super::super::{InstallHookError, json_hooks};

/// What Vibe runs the hooks as: `atuin hook vibe`.
const HARNESS: &str = "vibe";

/// Vibe's shell tool: its full name is `file_system.bash`, and `match` accepts the short name
/// `bash`. Commands the agent starts in the background go through `process_start` instead, so
/// they are not matched and never recorded.
const MATCHER: &str = "bash";

/// Write atuin's two hook entries to `<home>/hooks.toml`, running `executable` (atuin, by
/// absolute path: Vibe doesn't necessarily run hooks with the user's `PATH`), and return that
/// path. [`InstallHookError::AlreadyInstalled`] when the file already has exactly these
/// entries, so a rerun after an upgrade or a moved binary rewrites them and nothing else does.
pub async fn install(home: &Path, executable: &Path) -> Result<PathBuf, InstallHookError> {
    let path = path(home);
    let command = json_hooks::hook_command(executable, HARNESS)?;
    let wanted = [
        entry("atuin-pre-tool", "pre_tool", &command),
        entry("atuin-post-tool", "post_tool", &command),
    ];

    let mut doc = match tokio::fs::read_to_string(&path).await {
        Ok(text) => text
            .parse::<DocumentMut>()
            .map_err(|_| InstallHookError::Malformed("Vibe's hooks.toml is not TOML"))?,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => DocumentMut::new(),
        Err(err) => return Err(err.into()),
    };

    {
        let hooks = hooks_array(&mut doc)?;
        if wanted.iter().all(|want| hooks.iter().any(|have| entry_matches(have, want))) {
            return Err(InstallHookError::AlreadyInstalled);
        }
        // Atuin's entries are taken out wherever they sit, so a moved binary or an older entry
        // never leaves a stale or duplicate hook behind.
        let mut at = 0;
        while at < hooks.len() {
            if is_ours(hooks.get(at).expect("at is in range")) {
                hooks.remove(at);
            } else {
                at += 1;
            }
        }
        for want in &wanted {
            hooks.push(want.clone());
        }
    }

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(&path, doc.to_string()).await?;

    Ok(path)
}

/// Whether `<home>/hooks.toml` has an entry running `atuin hook vibe`, as any atuin's
/// [`install`] wrote it.
pub async fn has(home: &Path) -> Result<bool, InstallHookError> {
    let text = match tokio::fs::read_to_string(path(home)).await {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err.into()),
    };
    let doc = text
        .parse::<DocumentMut>()
        .map_err(|_| InstallHookError::Malformed("Vibe's hooks.toml is not TOML"))?;
    let Some(hooks) = doc.get("hooks").and_then(Item::as_array_of_tables) else {
        return Ok(false);
    };
    Ok(hooks.iter().any(is_ours))
}

/// Take atuin's entries out of `<home>/hooks.toml`, leaving the rest of the file as it is, and
/// the file itself when nothing else is in it. `false` when there were none.
pub async fn uninstall(home: &Path) -> Result<bool, InstallHookError> {
    let path = path(home);
    let text = match tokio::fs::read_to_string(&path).await {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err.into()),
    };
    let mut doc = text
        .parse::<DocumentMut>()
        .map_err(|_| InstallHookError::Malformed("Vibe's hooks.toml is not TOML"))?;

    let Some(hooks) = doc.get_mut("hooks").and_then(Item::as_array_of_tables_mut) else {
        return Ok(false);
    };
    let before = hooks.len();
    let mut at = 0;
    while at < hooks.len() {
        if is_ours(hooks.get(at).expect("at is in range")) {
            hooks.remove(at);
        } else {
            at += 1;
        }
    }
    if hooks.len() == before {
        return Ok(false);
    }
    if hooks.is_empty() {
        doc.remove("hooks");
    }
    // A file atuin made goes with its hooks, unless something else was put in it, which stays.
    if doc.as_table().is_empty() {
        return super::super::remove_own_file(&path).await;
    }
    tokio::fs::write(&path, doc.to_string()).await?;
    Ok(true)
}

/// Vibe's hook file under its home `home`.
#[must_use]
pub fn path(home: &Path) -> PathBuf {
    home.join("hooks.toml")
}

/// The `hooks` array of tables of `doc`, made when there is none.
fn hooks_array(doc: &mut DocumentMut) -> Result<&mut ArrayOfTables, InstallHookError> {
    doc.entry("hooks")
        .or_insert_with(|| Item::ArrayOfTables(ArrayOfTables::new()))
        .as_array_of_tables_mut()
        .ok_or(InstallHookError::Malformed(
            "`hooks` in Vibe's hooks.toml is not an array of tables",
        ))
}

/// One `[[hooks]]` entry running `command` for tool hooks matching Vibe's shell tool.
fn entry(name: &str, hook_type: &str, command: &str) -> Table {
    let mut table = Table::new();
    table["name"] = toml_edit::value(name);
    table["type"] = toml_edit::value(hook_type);
    table["match"] = toml_edit::value(MATCHER);
    table["command"] = toml_edit::value(command);
    table
}

/// Whether `table` is the entry `want` writes.
fn entry_matches(table: &Table, want: &Table) -> bool {
    ["name", "type", "match", "command"]
        .iter()
        .all(|key| table.get(key).and_then(Item::as_str) == want.get(key).and_then(Item::as_str))
}

/// Whether `table` runs `atuin hook vibe`: an executable named atuin, or this one, whatever
/// it's called.
fn is_ours(table: &Table) -> bool {
    let Some(command) = table.get("command").and_then(Item::as_str) else {
        return false;
    };
    if json_hooks::invokes_atuin_hook(command, HARNESS) {
        return true;
    }
    let self_command =
        std::env::current_exe().ok().and_then(|exe| json_hooks::hook_command(&exe, HARNESS).ok());
    self_command.as_deref() == Some(command)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn read_tables(path: &Path) -> Vec<Table> {
        let doc = std::fs::read_to_string(path).unwrap().parse::<DocumentMut>().unwrap();
        doc["hooks"].as_array_of_tables().unwrap().iter().cloned().collect()
    }

    /// Both entries run the given atuin for Vibe's shell tool.
    #[rstest]
    #[tokio::test]
    async fn writes_both_entries_for_the_shell_tool() {
        let home = tempfile::tempdir().unwrap();

        let path = install(home.path(), Path::new("/opt/atuin/bin/atuin")).await.unwrap();

        assert_eq!(path, home.path().join("hooks.toml"));
        let command = json_hooks::hook_command(Path::new("/opt/atuin/bin/atuin"), HARNESS).unwrap();
        let tables = read_tables(&path);
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0]["name"].as_str().unwrap(), "atuin-pre-tool");
        assert_eq!(tables[0]["type"].as_str().unwrap(), "pre_tool");
        assert_eq!(tables[0]["match"].as_str().unwrap(), "bash");
        assert_eq!(tables[0]["command"].as_str().unwrap(), command);
        assert_eq!(tables[1]["name"].as_str().unwrap(), "atuin-post-tool");
        assert_eq!(tables[1]["type"].as_str().unwrap(), "post_tool");
        assert_eq!(tables[1]["command"].as_str().unwrap(), command);
    }

    #[rstest]
    #[tokio::test]
    async fn a_second_install_changes_nothing() {
        let home = tempfile::tempdir().unwrap();
        let atuin = Path::new("/opt/atuin/bin/atuin");
        let path = install(home.path(), atuin).await.unwrap();
        let written = std::fs::read_to_string(&path).unwrap();

        assert!(matches!(
            install(home.path(), atuin).await,
            Err(InstallHookError::AlreadyInstalled)
        ));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), written);
    }

    /// The user's hooks and settings stay, with atuin's entries beside them.
    #[rstest]
    #[tokio::test]
    async fn keeps_the_users_own_hooks() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join("hooks.toml");
        std::fs::write(
            &path,
            "# mine\n[[hooks]]\nname = \"deny-rm-rf\"\ntype = \"pre_tool\"\nmatch = \
             \"bash\"\ncommand = \"guard rm\"\n",
        )
        .unwrap();

        install(home.path(), Path::new("/opt/atuin/bin/atuin")).await.unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# mine"), "{text}");
        assert!(text.contains("name = \"deny-rm-rf\""), "{text}");
        let tables = read_tables(&path);
        assert_eq!(tables.len(), 3);
        assert_eq!(tables[0]["name"].as_str().unwrap(), "deny-rm-rf");
        assert_eq!(tables[1]["name"].as_str().unwrap(), "atuin-pre-tool");
        assert_eq!(tables[2]["name"].as_str().unwrap(), "atuin-post-tool");
    }

    /// A stale entry (a moved binary, an older name) is replaced, not duplicated.
    #[rstest]
    #[tokio::test]
    async fn an_outdated_entry_is_rewritten() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join("hooks.toml");
        std::fs::write(
            &path,
            "[[hooks]]\nname = \"atuin\"\ntype = \"pre_tool\"\nmatch = \"bash\"\ncommand = \
             \"/old/atuin hook vibe\"\n",
        )
        .unwrap();

        install(home.path(), Path::new("/opt/atuin/bin/atuin")).await.unwrap();

        let tables = read_tables(&path);
        assert_eq!(tables.len(), 2, "{tables:?}");
        let command = json_hooks::hook_command(Path::new("/opt/atuin/bin/atuin"), HARNESS).unwrap();
        for table in &tables {
            assert_eq!(table["command"].as_str().unwrap(), command);
        }
    }

    /// Not TOML is refused, not written over.
    #[rstest]
    #[tokio::test]
    async fn a_file_that_is_not_toml_is_left_alone() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join("hooks.toml");
        std::fs::write(&path, "not toml").unwrap();

        let err = install(home.path(), Path::new("/opt/atuin/bin/atuin")).await.unwrap_err();

        assert!(matches!(err, InstallHookError::Malformed(_)), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "not toml");
    }

    #[rstest]
    #[tokio::test]
    async fn has_follows_install_and_uninstall() {
        let home = tempfile::tempdir().unwrap();
        let atuin = Path::new("/opt/atuin/bin/atuin");

        assert!(!has(home.path()).await.unwrap());
        install(home.path(), atuin).await.unwrap();
        assert!(has(home.path()).await.unwrap());
        assert!(uninstall(home.path()).await.unwrap());
        assert!(!has(home.path()).await.unwrap());
        assert!(!uninstall(home.path()).await.unwrap());
    }

    /// The user's hooks stay when atuin's come out; a file atuin made goes with them.
    #[rstest]
    #[case::with_theirs(true)]
    #[case::alone(false)]
    #[tokio::test]
    async fn uninstall_keeps_what_is_not_atuins(#[case] with_theirs: bool) {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join("hooks.toml");
        let mut before = String::new();
        if with_theirs {
            before.push_str(
                "[[hooks]]\nname = \"mine\"\ntype = \"post_tool\"\ncommand = \"say done\"\n",
            );
        }
        install(home.path(), Path::new("/opt/atuin/bin/atuin")).await.unwrap();
        if with_theirs {
            let mut text = std::fs::read_to_string(&path).unwrap();
            text.push_str(&before);
            std::fs::write(&path, text).unwrap();
        }

        assert!(uninstall(home.path()).await.unwrap());

        if with_theirs {
            let tables = read_tables(&path);
            assert_eq!(tables.len(), 1, "{tables:?}");
            assert_eq!(tables[0]["name"].as_str().unwrap(), "mine");
        } else {
            assert!(!path.exists());
        }
    }

    #[cfg(not(windows))]
    #[rstest]
    #[tokio::test]
    async fn quotes_the_executable_path() {
        let home = tempfile::tempdir().unwrap();

        let path = install(home.path(), Path::new("/opt/Atuin's bin/atuin")).await.unwrap();

        let tables = read_tables(&path);
        assert_eq!(tables[0]["command"].as_str().unwrap(), r#""/opt/Atuin's bin/atuin" hook vibe"#);
    }
}
