//! Vibe: the unified store `<save>/unified/<id>/`, or a legacy folder
//! `<save>/<prefix>_<hint>_<short id>/` (see [`crate::harnesstools::vibe::session`]).
//!
//! **Tip.** Vibe resumes a legacy folder from the last line of `messages.jsonl`, and a unified
//! store from the last history entry after the snapshot and journal fold — both what capture's
//! reader yields last, so [`SessionSync::local_tip`] reads them the same way.
//!
//! **Liveness.** While Vibe has a session open it holds an exclusive `flock` on
//! `<save>/active/<id>.lock` ([`flock_holder`]); besides that, any `vibe` process that may be
//! working on the session counts, as for the other agents ([`agent_running`]).
//!
//! **Append.** A legacy folder's rows go on the end of `messages.jsonl` ([`append_jsonl`], which
//! refuses a file changed under the tip read). A unified store's go into its recovery journal as
//! `projection_delta` records appending entries, which fold over the snapshot exactly as
//! capture's reader folds them, so Vibe resumes from the last of them.
//!
//! **Replace.** A legacy folder's transcript is rewritten keeping the lines the branch shares and
//! dropping those that went on past it (the transcript as it was backed up first,
//! [`replace_jsonl`]). A unified store's gets a `projection_advanced` record holding the branch's
//! entries, with the journal backed up first.

use std::collections::{HashMap, HashSet};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::liveness::{
    Dirs, Liveness, Processes, Seen, agent_running, changed_lately, flock_holder,
};
use super::{
    AppendOptions, AppendOutcome, LocalTip, ReplaceOutcome, SessionSync, Stamp, SyncError,
    append_jsonl, backup_path, blocking, blocking_liveness, jsonl_lines, kept_lines, modified,
    replace_jsonl, require_idle, with_merged,
};
use crate::harnesstools::rehydrate::{RehydrateMessage, record_merged};
use crate::harnesstools::vibe::session::{
    ACTIVE_DIR, Location, default_root, locate_full, read_meta, read_store,
};
use crate::harnesstools::vibe::{Vibe, rehydrate};

/// The process name Vibe runs under, and its package name as a script's path may show it.
const BINARY: &str = "vibe";
const ALIASES: &[&str] = &["mistral-vibe"];

impl SessionSync for Vibe {
    async fn local_tip(&self, id: &str) -> Result<Option<LocalTip>, SyncError> {
        let (root, id) = (default_root(), id.to_owned());
        blocking(move || tip(&root, &id)).await
    }

    async fn is_live(&self, id: &str, cwd: Option<&Path>) -> Liveness {
        let (root, id, cwd) = (default_root(), id.to_owned(), cwd.map(Path::to_path_buf));
        blocking_liveness(move || {
            let seen = match locate_full(&root, &id) {
                Some(location) => {
                    let transcript = location.transcript();
                    Seen {
                        dirs: Dirs::of(cwd),
                        changing: changed_lately(modified(&transcript)),
                        names: names(&id, Some(&transcript)),
                    }
                }
                None => Seen {
                    dirs: Dirs::of(cwd),
                    changing: false,
                    names: vec![id.clone()],
                },
            };
            liveness(&root, &id, &seen)
        })
        .await
    }

    async fn append(
        &self,
        id: &str,
        base: &LocalTip,
        lines: &[RehydrateMessage],
        options: &AppendOptions<'_>,
    ) -> Result<AppendOutcome, SyncError> {
        let (root, id) = (default_root(), id.to_owned());
        let seen = base.seen(options.cwd, names(&id, Some(&base.native_path)));
        require_idle(blocking_liveness(move || liveness(&root, &id, &seen)).await)?;
        let (base, lines) = (base.clone(), lines.to_vec());
        let taken = options.taken_ids.cloned();
        blocking(move || append(&base, &lines, taken.as_ref())).await
    }

    async fn replace(
        &self,
        id: &str,
        base: &LocalTip,
        branch: &[RehydrateMessage],
        options: &AppendOptions<'_>,
        backups: &Path,
    ) -> Result<ReplaceOutcome, SyncError> {
        let (root, id) = (default_root(), id.to_owned());
        let seen = base.seen(options.cwd, names(&id, Some(&base.native_path)));
        require_idle(blocking_liveness(move || liveness(&root, &id, &seen)).await)?;
        let (base, branch, backups) = (base.clone(), branch.to_vec(), backups.to_path_buf());
        let taken = options.taken_ids.cloned();
        blocking(move || replace(&base, &branch, taken.as_ref(), &backups)).await
    }
}

/// Whether a Vibe process may be writing session `id` under `root`: its lease is held, or a
/// `vibe` process `seen` allows is running.
fn liveness(root: &Path, id: &str, seen: &Seen) -> Liveness {
    let lease = root.join(ACTIVE_DIR).join(format!("{id}.lock"));
    let held = flock_holder(&lease, &Processes::here());
    if !matches!(held, Liveness::NotLive) {
        return held;
    }
    agent_running(&Processes::here(), BINARY, ALIASES, seen)
}

/// What a Vibe command line opening session `id`, kept at `file`, would name: its id, and its
/// file, whole or by name.
fn names(id: &str, file: Option<&Path>) -> Vec<String> {
    let file = file.into_iter().flat_map(|f| [Some(f.as_os_str()), f.file_name()]).flatten();
    std::iter::once(id.to_owned()).chain(file.map(|f| f.to_string_lossy().into_owned())).collect()
}

/// The local copy of session `id` under `root`, as [`SessionSync::local_tip`] reads it.
fn tip(root: &Path, id: &str) -> Result<Option<LocalTip>, SyncError> {
    let Some(location) = locate_full(root, id) else {
        return Ok(None);
    };
    Ok(Some(match &location {
        Location::Legacy(dir) => {
            let path = dir.join("messages.jsonl");
            let bytes = std::fs::read(&path)?;
            let mut known = HashSet::new();
            let mut tip = None;
            for line in jsonl_lines(&bytes) {
                if let Some(id) = crate::json::js::from_slice::<Value>(line)
                    .ok()
                    .and_then(|v| v.get("message_id").and_then(Value::as_str).map(str::to_owned))
                {
                    known.insert(id.clone());
                    tip = Some(id);
                }
            }
            let (known, merged) = with_merged(known, &bytes);
            LocalTip {
                native_path: path,
                known_source_ids: known,
                merged,
                tip_source_id: tip,
                stamp: Stamp::of(&bytes),
                modified: modified(&location.transcript()),
                cwd: read_meta(&dir.join("meta.json")).and_then(|meta| meta.cwd),
                unswitchable: None,
            }
        }
        Location::Unified(store) => {
            let state = read_store(store)?;
            let mut known = HashSet::new();
            let mut tip = None;
            for (_, entry) in &state.entries {
                if let Some(id) = entry.get("id").and_then(Value::as_str).map(str::to_owned) {
                    known.insert(id.clone());
                    tip = Some(id);
                }
            }
            LocalTip {
                native_path: store.join("CURRENT"),
                known_source_ids: known,
                merged: HashMap::new(),
                tip_source_id: tip,
                stamp: state.stamp,
                modified: state.modified,
                cwd: state.metadata.get("cwd").and_then(Value::as_str).map(PathBuf::from),
                unswitchable: None,
            }
        }
    }))
}

/// The rows of `lines` the transcript `base` does not hold yet, as a clean fast-forward of it:
/// each hangs from the tip or from an earlier one of them.
fn new_rows<'a>(
    base: &LocalTip,
    lines: &'a [RehydrateMessage],
    taken: Option<&HashSet<String>>,
) -> Result<Vec<&'a RehydrateMessage>, SyncError> {
    let mut out = Vec::new();
    for row in lines {
        if base.known_source_ids.contains(&row.source_id)
            || base.merged.contains_key(&row.source_id)
        {
            continue;
        }
        if let Some(taken) = taken
            && taken.contains(&row.source_id)
        {
            return Err(SyncError::IdTaken(row.source_id.clone()));
        }
        match &row.parent_source_id {
            Some(parent)
                if *parent != row.source_id
                    && !base.known_source_ids.contains(parent.as_str())
                    && !lines.iter().any(|r| r.source_id == *parent) =>
            {
                return Err(SyncError::Unsupported(
                    "the rows are no clean fast-forward of the transcript",
                ));
            }
            _ => {}
        }
        out.push(row);
    }
    Ok(out)
}

/// Append `lines` to the session `base` read, whose transcript is `base.native_path`.
fn append(
    base: &LocalTip,
    lines: &[RehydrateMessage],
    taken: Option<&HashSet<String>>,
) -> Result<AppendOutcome, SyncError> {
    let new = new_rows(base, lines, taken)?;
    let rows: Vec<RehydrateMessage> = new.into_iter().cloned().collect();
    if base.native_path.file_name().is_some_and(|name| name == "CURRENT") {
        append_unified(base.native_path.parent().unwrap(), base, &rows)
    } else {
        append_legacy(&base.native_path, base, &rows)
    }
}

/// Append `rows` to the legacy folder's `messages.jsonl` at `path`.
fn append_legacy(
    path: &Path,
    base: &LocalTip,
    rows: &[RehydrateMessage],
) -> Result<AppendOutcome, SyncError> {
    let rendered = rehydrate::render_rows(rows);
    let mut lines = rendered.lines;
    record_merged(&mut lines, &rendered.merged, |line| {
        line.get("message_id").and_then(Value::as_str).map(str::to_owned)
    });
    let appended = rendered.appended;
    append_jsonl(path, base.stamp, &lines)?;
    Ok(AppendOutcome {
        native_path: path.to_path_buf(),
        tip_source_id: appended.last().cloned(),
        appended,
    })
}

/// Append `rows` to the unified store at `store` as `projection_delta` records in its recovery
/// journal.
fn append_unified(
    store: &Path,
    base: &LocalTip,
    rows: &[RehydrateMessage],
) -> Result<AppendOutcome, SyncError> {
    let journal = journal_segment(store)?;
    let next = next_sequence(store, &journal)?;
    let calls = call_inputs(rows);
    let mut entries = Vec::new();
    let mut appended = Vec::new();
    for row in rows {
        for entry in rehydrate::render_entry(row, &calls) {
            if let Some(id) = entry.get("id").and_then(Value::as_str) {
                appended.push(id.to_owned());
            }
            entries.push(entry);
        }
    }
    let records: Vec<Value> = entries
        .into_iter()
        .enumerate()
        .map(|(at, entry)| {
            json!({
                "type": "projection_delta",
                "sequence": next + at as u64,
                "payload": {"delta": [{"op": "append_entry", "entry": entry}]},
            })
        })
        .collect();
    append_records(store, base, &journal, &records)?;
    Ok(AppendOutcome {
        native_path: store.join("CURRENT"),
        tip_source_id: appended.last().cloned(),
        appended,
    })
}

/// Switch the session `base` read to `branch`: the lines the branch shares stay, those that went
/// on past them go (the transcript backed up to `backups` first), and the branch's own rows are
/// written after them.
fn replace(
    base: &LocalTip,
    branch: &[RehydrateMessage],
    taken: Option<&HashSet<String>>,
    backups: &Path,
) -> Result<ReplaceOutcome, SyncError> {
    let branch_ids: HashSet<&str> = branch.iter().map(|row| row.source_id.as_str()).collect();
    if !base.known_source_ids.iter().any(|id| branch_ids.contains(id.as_str())) {
        return Err(SyncError::Unsupported("the transcript shares no row with the branch"));
    }
    if base.native_path.file_name().is_some_and(|name| name == "CURRENT") {
        replace_unified(base.native_path.parent().unwrap(), base, branch, backups)
    } else {
        replace_legacy(&base.native_path, base, branch, taken, backups)
    }
}

/// [`replace`] for a legacy folder: rewrite `messages.jsonl` at `path`.
fn replace_legacy(
    path: &Path,
    base: &LocalTip,
    branch: &[RehydrateMessage],
    taken: Option<&HashSet<String>>,
    backups: &Path,
) -> Result<ReplaceOutcome, SyncError> {
    let current = std::fs::read(path)?;
    if Stamp::of(&current) != base.stamp {
        return Err(SyncError::Changed);
    }
    let branch_ids: HashSet<&str> = branch.iter().map(|row| row.source_id.as_str()).collect();
    let kept = kept_lines(&current, |line| {
        let id = crate::json::js::from_slice::<Value>(line)
            .ok()
            .and_then(|v| v.get("message_id").and_then(Value::as_str).map(str::to_owned));
        id.as_deref().map(|id| branch_ids.contains(id))
    });
    let new = new_rows(base, branch, taken)?;
    let rows: Vec<RehydrateMessage> = new.into_iter().cloned().collect();
    let rendered = rehydrate::render_rows(&rows);
    let mut lines = rendered.lines;
    record_merged(&mut lines, &rendered.merged, |line| {
        line.get("message_id").and_then(Value::as_str).map(str::to_owned)
    });
    let mut bytes = kept;
    for line in &lines {
        serde_json::to_writer(&mut bytes, line).map_err(|err| SyncError::Other(err.to_string()))?;
        bytes.push(b'\n');
    }
    let (backup, warning) = replace_jsonl(path, base.stamp, &current, &bytes, backups)?;
    Ok(ReplaceOutcome {
        native_path: path.to_path_buf(),
        appended: rendered.appended,
        tip_source_id: branch.last().map(|row| row.source_id.clone()),
        backup,
        warning,
    })
}

/// [`replace`] for a unified store: a `projection_advanced` record holding the branch's entries
/// (the store's entries the branch shares, then its own rows), appended to the recovery journal
/// after the journal as it was is backed up to `backups`.
fn replace_unified(
    store: &Path,
    base: &LocalTip,
    branch: &[RehydrateMessage],
    backups: &Path,
) -> Result<ReplaceOutcome, SyncError> {
    let state = read_store(store)?;
    if state.stamp != base.stamp {
        return Err(SyncError::Changed);
    }
    let journal = journal_segment(store)?;
    let journal_bytes = std::fs::read(&journal).unwrap_or_default();
    let backup = backup_path(&journal, backups, jiff::Timestamp::now(), 0)
        .ok_or_else(|| SyncError::Other("no backup path".to_owned()))?;
    std::fs::create_dir_all(backups)?;
    crate::fs::write_new(&backup, &journal_bytes)?;

    let branch_ids: HashSet<&str> = branch.iter().map(|row| row.source_id.as_str()).collect();
    let mut entries: Vec<Value> = state
        .entries
        .iter()
        .filter(|(_, entry)| {
            entry.get("id").and_then(Value::as_str).is_some_and(|id| branch_ids.contains(id))
        })
        .map(|(_, entry)| entry.clone())
        .collect();
    let calls = call_inputs(branch);
    let new = new_rows(base, branch, None)?;
    let rows: Vec<RehydrateMessage> = new.into_iter().cloned().collect();
    let mut appended = Vec::new();
    for row in &rows {
        for entry in rehydrate::render_entry(row, &calls) {
            if let Some(id) = entry.get("id").and_then(Value::as_str) {
                appended.push(id.to_owned());
            }
            entries.push(entry);
        }
    }
    let next = next_sequence(store, &journal)?;
    let record = json!({
        "type": "projection_advanced",
        "sequence": next,
        "payload": {"snapshot": {"history": {"entries": entries}}},
    });
    append_records(store, base, &journal, &[record])?;
    Ok(ReplaceOutcome {
        native_path: store.join("CURRENT"),
        appended,
        tip_source_id: branch.last().map(|row| row.source_id.clone()),
        backup,
        warning: None,
    })
}

/// The recovery journal segment of the store at `store`, as its manifest names it.
fn journal_segment(store: &Path) -> Result<PathBuf, SyncError> {
    let read_json = |path: &Path| -> Result<Value, SyncError> {
        let bytes = std::fs::read(path)?;
        crate::json::js::from_slice(&bytes).map_err(|err| SyncError::Other(err.to_string()))
    };
    let current = read_json(&store.join("CURRENT"))?;
    let generation = current
        .get("generation")
        .map(|g| g.as_str().map(str::to_owned).unwrap_or_else(|| g.to_string()))
        .ok_or_else(|| SyncError::Other("CURRENT names no generation".to_owned()))?;
    let manifest = read_json(&store.join("generations").join(&generation).join("manifest.json"))?;
    let segment = manifest
        .get("recovery_journal_segment")
        .and_then(|s| s.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| SyncError::Other("the manifest names no recovery journal".to_owned()))?;
    Ok(store.join(segment))
}

/// The sequence the next journal record of the store at `store` takes: one past the last the
/// journal at `journal` holds, or the snapshot's.
fn next_sequence(store: &Path, journal: &Path) -> Result<u64, SyncError> {
    let current: Value = {
        let bytes = std::fs::read(store.join("CURRENT"))?;
        crate::json::js::from_slice(&bytes).map_err(|err| SyncError::Other(err.to_string()))?
    };
    let snapshot_sequence = current.get("snapshot_sequence").and_then(Value::as_u64).unwrap_or(0);
    let journal_bytes = std::fs::read(journal).unwrap_or_default();
    let mut next = snapshot_sequence;
    for line in jsonl_lines(&journal_bytes) {
        if let Ok(record) = crate::json::js::from_slice::<Value>(line) {
            next = next.max(record.get("sequence").and_then(Value::as_u64).unwrap_or(0));
        }
    }
    Ok(next + 1)
}

/// Append `records` to the recovery journal at `journal`, once the store at `store` is still what
/// `base` read.
fn append_records(
    store: &Path,
    base: &LocalTip,
    journal: &Path,
    records: &[Value],
) -> Result<(), SyncError> {
    let mut file = std::fs::OpenOptions::new().read(true).append(true).open(journal)?;
    let mut current = Vec::new();
    file.read_to_end(&mut current)?;
    // The stamp covers every file of the store; re-reading it is the check that none changed.
    if read_store(store)?.stamp != base.stamp {
        return Err(SyncError::Changed);
    }
    let mut out = Vec::new();
    if !current.is_empty() && !current.ends_with(b"\n") {
        out.push(b'\n');
    }
    for record in records {
        serde_json::to_writer(&mut out, record).map_err(|err| SyncError::Other(err.to_string()))?;
        out.push(b'\n');
    }
    file.write_all(&out)?;
    file.sync_data()?;
    Ok(())
}

/// Every tool call among `rows`, by call id: its name and input.
fn call_inputs(rows: &[RehydrateMessage]) -> HashMap<String, (String, Value)> {
    rows.iter()
        .flat_map(|row| &row.content)
        .filter_map(|content| match content {
            crate::harnesstools::session::Content::ToolUse(call) => {
                Some((call.id.as_ref().to_owned(), (call.name.clone(), call.input.clone())))
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::harnesstools::session::{Content, Role, ToolCallId, ToolResult, ToolUse};
    use crate::harnesstools::vibe::session::UNIFIED_DIR;
    use crate::harnesstools::vibe::session::tests::{write_legacy, write_unified};

    fn row(id: &str, role: Role, content: Vec<Content>) -> RehydrateMessage {
        RehydrateMessage {
            source_id: id.to_owned(),
            parent_source_id: None,
            timestamp: "2026-09-18T10:00:00Z".parse::<jiff::Timestamp>().unwrap(),
            role,
            content,
            model: None,
            provider: None,
            usage: None,
            stop_reason: None,
            turn_id: Some("t1".to_owned()),
            cwd: Some(PathBuf::from("/work/proj")),
            git_branch: None,
        }
    }

    fn text(id: &str, role: Role, text: &str) -> RehydrateMessage {
        row(id, role, vec![Content::Text(text.to_owned())])
    }

    #[rstest]
    fn local_tip_reads_a_legacy_folder() {
        let dir = tempfile::tempdir().unwrap();
        let env = crate::env::MockEnv::install();
        env.set("HOME", dir.path());
        let folder = write_legacy(
            dir.path(),
            "session_hint_aaa",
            &json!({
                "session_id": "legacy-1",
                "environment": {"working_directory": "/work/proj"},
            }),
            &[
                json!({"role": "user", "content": "hi", "message_id": "m1"}),
                json!({"role": "assistant", "content": "hello", "message_id": "m2"}),
            ],
        );
        let tip = tip(dir.path(), "legacy-1").unwrap().unwrap();
        assert_eq!(tip.native_path, folder.join("messages.jsonl"));
        assert_eq!(tip.tip_source_id.as_deref(), Some("m2"));
        assert!(tip.known_source_ids.contains("m1") && tip.known_source_ids.contains("m2"));
        assert_eq!(tip.cwd, Some(PathBuf::from("/work/proj")));
    }

    #[rstest]
    fn append_to_a_legacy_folder_extends_messages_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let env = crate::env::MockEnv::install();
        env.set("HOME", dir.path());
        let folder =
            write_legacy(dir.path(), "session_hint_aaa", &json!({"session_id": "legacy-1"}), &[
                json!({"role": "user", "content": "hi", "message_id": "m1"}),
            ]);
        let base = tip(dir.path(), "legacy-1").unwrap().unwrap();
        let outcome = append(&base, &[text("m2", Role::Assistant, "hello")], None).unwrap();
        assert_eq!(outcome.appended, vec!["m2".to_string()]);
        assert_eq!(outcome.tip_source_id.as_deref(), Some("m2"));
        // The transcript now holds the appended line.
        let body = std::fs::read_to_string(folder.join("messages.jsonl")).unwrap();
        let ids: Vec<String> = body
            .lines()
            .map(|l| {
                serde_json::from_str::<Value>(l).unwrap()["message_id"].as_str().unwrap().to_owned()
            })
            .collect();
        assert_eq!(ids, vec!["m1", "m2"]);
        // The transcript changed under the tip read: a second append is refused.
        assert!(matches!(
            append(&base, &[text("m3", Role::User, "again")], None),
            Err(SyncError::Changed)
        ));
    }

    #[rstest]
    fn local_tip_and_append_for_a_unified_store() {
        let dir = tempfile::tempdir().unwrap();
        let env = crate::env::MockEnv::install();
        env.set("HOME", dir.path());
        write_unified(
            dir.path(),
            "unified-1",
            &[json!({
                "id": "e1",
                "created_at": 1_700_000_000_000u64,
                "turn_id": "t1",
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            })],
            &[],
            &json!({"cwd": "/work/proj"}),
        );
        let base = tip(dir.path(), "unified-1").unwrap().unwrap();
        assert_eq!(base.tip_source_id.as_deref(), Some("e1"));
        assert_eq!(base.cwd, Some(PathBuf::from("/work/proj")));

        let outcome = append(
            &base,
            &[
                text("e2", Role::Assistant, "hello"),
                row("e3", Role::Other("effect".to_owned()), vec![
                    Content::ToolUse(ToolUse {
                        id: ToolCallId::from("c1".to_owned()),
                        name: "bash".to_owned(),
                        input: json!({"cmd": "ls"}),
                    }),
                    Content::ToolResult(ToolResult {
                        call: ToolCallId::from("c1".to_owned()),
                        output: json!("file.txt"),
                        error: false,
                    }),
                ]),
            ],
            None,
        )
        .unwrap();
        assert_eq!(outcome.appended, vec!["e2".to_string(), "e3#c1".to_string()]);

        // The fold now holds the appended entries: capture's reader sees them.
        let state = read_store(&dir.path().join(UNIFIED_DIR).join("unified-1")).unwrap();
        let ids: Vec<&str> =
            state.entries.iter().filter_map(|(_, e)| e.get("id").and_then(Value::as_str)).collect();
        assert_eq!(ids, vec!["e1", "e2", "e3#c1"]);
        let effect = &state.entries[2].1;
        assert_eq!(effect["type"], json!("effect"));
        assert_eq!(effect["detail"]["tool_name"], json!("bash"));
        assert_eq!(effect["state"]["status"], json!("completed"));
        assert_eq!(effect["state"]["output"], json!("file.txt"));

        // Changed under the tip read.
        assert!(matches!(
            append(&base, &[text("e9", Role::User, "x")], None),
            Err(SyncError::Changed)
        ));
    }

    #[rstest]
    fn replace_a_unified_store_switches_to_the_branch() {
        let dir = tempfile::tempdir().unwrap();
        let env = crate::env::MockEnv::install();
        env.set("HOME", dir.path());
        let entry = |id: &str, text: &str| {
            json!({
                "id": id,
                "created_at": 1_700_000_000_000u64,
                "turn_id": "t1",
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": text}],
            })
        };
        write_unified(
            dir.path(),
            "unified-1",
            &[entry("e1", "hi"), entry("e2", "old")],
            &[],
            &json!({}),
        );
        let base = tip(dir.path(), "unified-1").unwrap().unwrap();
        let backups = dir.path().join("backups");
        let outcome = replace(
            &base,
            &[text("e1", Role::User, "hi"), text("e3", Role::Assistant, "new")],
            None,
            &backups,
        )
        .unwrap();
        assert_eq!(outcome.appended, vec!["e3".to_string()]);
        assert_eq!(outcome.tip_source_id.as_deref(), Some("e3"));
        assert!(outcome.backup.is_file());

        let state = read_store(&dir.path().join(UNIFIED_DIR).join("unified-1")).unwrap();
        let ids: Vec<&str> =
            state.entries.iter().filter_map(|(_, e)| e.get("id").and_then(Value::as_str)).collect();
        assert_eq!(ids, vec!["e1", "e3"]);
    }

    #[rstest]
    fn replace_a_legacy_folder_keeps_the_shared_lines() {
        let dir = tempfile::tempdir().unwrap();
        let env = crate::env::MockEnv::install();
        env.set("HOME", dir.path());
        let folder =
            write_legacy(dir.path(), "session_hint_aaa", &json!({"session_id": "legacy-1"}), &[
                json!({"role": "user", "content": "hi", "message_id": "m1"}),
                json!({"role": "assistant", "content": "old", "message_id": "m2"}),
            ]);
        let base = tip(dir.path(), "legacy-1").unwrap().unwrap();
        let backups = dir.path().join("backups");
        let outcome = replace(
            &base,
            &[text("m1", Role::User, "hi"), text("m3", Role::Assistant, "new")],
            None,
            &backups,
        )
        .unwrap();
        assert_eq!(outcome.appended, vec!["m3".to_string()]);
        let body = std::fs::read_to_string(folder.join("messages.jsonl")).unwrap();
        let ids: Vec<String> = body
            .lines()
            .map(|l| {
                serde_json::from_str::<Value>(l).unwrap()["message_id"].as_str().unwrap().to_owned()
            })
            .collect();
        assert_eq!(ids, vec!["m1", "m3"]);
        assert!(outcome.backup.is_file());
    }

    #[rstest]
    fn a_branch_sharing_no_row_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let env = crate::env::MockEnv::install();
        env.set("HOME", dir.path());
        write_legacy(dir.path(), "session_hint_aaa", &json!({"session_id": "legacy-1"}), &[
            json!({"role": "user", "content": "hi", "message_id": "m1"}),
        ]);
        let base = tip(dir.path(), "legacy-1").unwrap().unwrap();
        let backups = dir.path().join("backups");
        assert!(matches!(
            replace(&base, &[text("zz", Role::User, "x")], None, &backups),
            Err(SyncError::Unsupported(_))
        ));
    }
}
