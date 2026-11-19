//! Writing a Vibe session back out from captured messages, as a legacy session folder
//! `<save>/<prefix>_<hint>_<short id>/{meta.json,messages.jsonl}`: the format Vibe 2.x wrote and
//! current Vibe still reads when a session is resumed or imported, so a session recorded on
//! another machine (or whose store was deleted) can be reopened here with `vibe --resume <id>`.
//!
//! `messages.jsonl` holds one LLMMessage per line: `role` (`user`, `assistant`, `tool`), `content`
//! (a string, or content blocks: `text`, `tool_use`), `tool_call_id` and `content` for a tool
//! result, and `message_id` holding the captured row's `source_id`, so re-capturing the folder
//! dedups against the synced rows. `meta.json` names the session, its start time, working and
//! origin directories, title, git branch and parent session.
//!
//! What the format can't carry is left out and counted in the note: reasoning (Vibe replays a
//! thinking block only with its signature, which capture doesn't keep), system-role rows, patches,
//! and tool calls captured without their input (their results too: a result for a call the model
//! never made would corrupt the conversation), each call becoming a note in its row's text
//! ([`tool_note`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::session::{default_root, locate, session_prefix};
use crate::harnesstools::note::tool_note;
use crate::harnesstools::rehydrate::{
    RehydrateError, RehydrateMessage, RehydrateSession, UNCAPTURED_OUTPUT, parent_order,
    record_merged,
};
use crate::harnesstools::resume;
use crate::harnesstools::session::{Content, Role, ToolCallId, ToolResult, ToolUse};

/// Write `session` under Vibe's save dir (see [`super::session`]), as a legacy folder, and return
/// the folder [`locate`] finds it by.
pub async fn rehydrate(session: &RehydrateSession) -> Result<PathBuf, RehydrateError> {
    let root = default_root();
    let session = session.clone();
    tokio::task::spawn_blocking(move || rehydrate_into(&root, &session))
        .await
        .map_err(|err| RehydrateError::Other(err.to_string()))?
}

/// [`rehydrate`] under `root`.
pub(crate) fn rehydrate_into(
    root: &Path,
    session: &RehydrateSession,
) -> Result<PathBuf, RehydrateError> {
    if !resume::is_plain_name(&session.id) {
        return Err(RehydrateError::Other(format!("not a session id: {:?}", session.id)));
    }
    if let Some(existing) = locate(root, &session.id) {
        return Err(RehydrateError::AlreadyExists(existing));
    }
    let short: String = session.id.chars().take(8).collect();
    let folder = root.join(format!("{}_{}_{short}", session_prefix(), hint(session)));
    std::fs::create_dir_all(&folder)?;
    // `meta.json` first: the reader tells a folder a session only once both files are there, and
    // `messages.jsonl` is what the watcher reports.
    std::fs::write(folder.join("meta.json"), meta_json(session).to_string())?;
    let body = render_lines(session);
    crate::fs::write_new(&folder.join("messages.jsonl"), body.as_bytes())?;
    Ok(folder)
}

/// The folder's hint between prefix and short id: the title slugged, else `session`.
fn hint(session: &RehydrateSession) -> String {
    let slug: String = session
        .title
        .as_deref()
        .unwrap_or("session")
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let slug: String =
        slug.split('-').filter(|part| !part.is_empty()).collect::<Vec<_>>().join("-");
    let slug: String = slug.chars().take(40).collect();
    if slug.is_empty() {
        "session".to_owned()
    } else {
        slug
    }
}

/// The folder's `meta.json`.
fn meta_json(session: &RehydrateSession) -> Value {
    let mut meta = Map::new();
    meta.insert("session_id".into(), json!(session.id));
    meta.insert(
        "start_time".into(),
        json!(
            jiff::fmt::strtime::format("%Y-%m-%dT%H:%M:%S%.3fZ", session.started_at)
                .unwrap_or_default()
        ),
    );
    meta.insert("environment".into(), json!({"working_directory": session.cwd}));
    meta.insert(
        "origin_directory".into(),
        json!(session.original_cwd.as_ref().unwrap_or(&session.cwd)),
    );
    if let Some(title) = &session.title {
        meta.insert("title".into(), json!(title));
        meta.insert("title_source".into(), json!("manual"));
    }
    if let Some(branch) = &session.git_branch {
        meta.insert("git_branch".into(), json!(branch));
    }
    if let Some(fork) = &session.fork_of {
        meta.insert("parent_session_id".into(), json!(fork.id));
    }
    Value::Object(meta)
}

/// The folder's `messages.jsonl`: one LLMMessage per captured row, in parent order.
fn render_lines(session: &RehydrateSession) -> String {
    let rendered = render_rows(&parent_order(session.messages.clone()));
    let mut lines = rendered.lines;
    record_merged(&mut lines, &rendered.merged, |line| {
        line.get("message_id").and_then(Value::as_str).map(str::to_owned)
    });
    let mut body = String::new();
    for line in &lines {
        body.push_str(&line.to_string());
        body.push('\n');
    }
    body
}

/// The lines one set of captured rows writes, with the rows left without a line of their own.
#[derive(Debug, Default)]
pub(crate) struct Rendered {
    pub lines: Vec<Value>,
    /// The `message_id` each line carries, in order.
    pub appended: Vec<String>,
    /// Rows holding nothing written for them (a result whose call was captured without its
    /// input), each with the row whose line stands for it.
    pub merged: Vec<(String, String)>,
}

/// Render `rows` as legacy LLMMessage lines, in the order given.
pub(crate) fn render_rows(rows: &[RehydrateMessage]) -> Rendered {
    // Calls captured without their input: their results are dropped with them, a result for a
    // call the model never made would corrupt the conversation.
    let uncaptured: std::collections::HashSet<String> = rows
        .iter()
        .flat_map(|row| &row.content)
        .filter_map(|content| match content {
            Content::ToolUse(call) if call.input.is_null() => Some(call.id.as_ref().to_owned()),
            _ => None,
        })
        .collect();
    let mut call_rows: HashMap<String, String> = HashMap::new();
    for row in rows {
        for content in &row.content {
            if let Content::ToolUse(call) = content {
                call_rows.insert(call.id.as_ref().to_owned(), row.source_id.clone());
            }
        }
    }
    let mut rendered = Rendered::default();
    for row in rows {
        for line in render_row(row, &uncaptured) {
            if let Some(id) = line.get("message_id").and_then(Value::as_str) {
                rendered.appended.push(id.to_owned());
            }
            rendered.lines.push(line);
        }
    }
    for row in rows {
        for content in &row.content {
            if let Content::ToolResult(result) = content
                && uncaptured.contains(result.call.as_ref())
                && let Some(into) = call_rows.get(result.call.as_ref())
            {
                rendered.merged.push((row.source_id.clone(), into.clone()));
            }
        }
    }
    rendered
}

/// One captured row as unified history entries: a `message` entry for its conversation text, an
/// `effect` entry per tool call (its result in `state` when the row holds it, or when `calls`
/// maps the call's id to its name and input). `calls` is every call of the rows being written,
/// by call id.
pub(crate) fn render_entry(
    row: &RehydrateMessage,
    calls: &HashMap<String, (String, Value)>,
) -> Vec<Value> {
    let mut text = String::new();
    let mut uses: Vec<&ToolUse> = Vec::new();
    let mut results: Vec<&ToolResult> = Vec::new();
    for content in &row.content {
        match content {
            Content::Text(part) | Content::Summary(part) | Content::Error(part) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(part);
            }
            Content::ToolUse(call) if !call.input.is_null() => uses.push(call),
            Content::ToolUse(call) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&tool_note(&call.name, &call.input));
            }
            Content::ToolResult(result) => results.push(result),
            Content::Reasoning(_)
            | Content::ReasoningSummary { .. }
            | Content::Patch(_)
            | Content::Other(_)
            | Content::Unknown(_) => {}
        }
    }
    let created_at = u64::try_from(row.timestamp.as_millisecond()).unwrap_or_default();
    let turn_id = || row.turn_id.clone().unwrap_or_else(|| row.source_id.clone());

    let mut out = Vec::new();
    let conversation = matches!(row.role, Role::User | Role::Assistant | Role::Other(_));
    if conversation && (!text.is_empty() || (uses.is_empty() && results.is_empty())) {
        let role = match row.role {
            Role::User => "user",
            _ => "assistant",
        };
        out.push(json!({
            "id": row.source_id,
            "created_at": created_at,
            "turn_id": turn_id(),
            "type": "message",
            "role": role,
            "content": if text.is_empty() {
                Value::Array(Vec::new())
            } else {
                json!([{"type": "text", "text": text}])
            },
        }));
    }
    let mut effect = |call: &str, name: &str, input: &Value, result: Option<&ToolResult>| {
        let (status, output, error) = match result {
            Some(r) => (
                if r.error {
                    "failed"
                } else {
                    "completed"
                },
                if r.output.is_null() {
                    json!(UNCAPTURED_OUTPUT)
                } else {
                    r.output.clone()
                },
                r.error,
            ),
            None => ("pending", Value::Null, false),
        };
        out.push(json!({
            "id": format!("{}#{call}", row.source_id),
            "created_at": created_at,
            "turn_id": turn_id(),
            "type": "effect",
            "tool_call_id": call,
            "detail": {"tool_name": name, "input": input},
            "state": {"status": status, "output": output, "is_error": error},
        }));
    };
    for call in &uses {
        let result = results.iter().find(|r| r.call.as_ref() == call.id.as_ref()).copied();
        effect(call.id.as_ref(), &call.name, &call.input, result);
    }
    for result in &results {
        if uses.iter().any(|u| u.id.as_ref() == result.call.as_ref()) {
            continue;
        }
        let (name, input) = calls
            .get(result.call.as_ref())
            .cloned()
            .unwrap_or_else(|| (String::new(), Value::Null));
        effect(result.call.as_ref(), &name, &input, Some(result));
    }
    out
}

/// One captured row as legacy lines: a conversation line for `user`/`assistant` (with any tool
/// calls in its content blocks), a `tool` line per tool result. Rows that hold nothing the format
/// can carry (reasoning, system, unknown blocks) write nothing, and results of `uncaptured`
/// calls are left out.
fn render_row(
    row: &RehydrateMessage,
    uncaptured: &std::collections::HashSet<String>,
) -> Vec<Value> {
    let mut text = String::new();
    let mut blocks: Vec<Value> = Vec::new();
    let mut results: Vec<(ToolCallId, Value, bool)> = Vec::new();
    for content in &row.content {
        match content {
            Content::Text(part) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(part);
            }
            Content::Summary(part) | Content::Error(part) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(part);
            }
            Content::ToolUse(call) if call.input.is_null() => {
                // A call captured without its input: a note in the text, its result dropped.
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&tool_note(&call.name, &call.input));
            }
            Content::ToolUse(call) => blocks.push(json!({
                "type": "tool_use",
                "id": call.id.as_ref(),
                "name": call.name,
                "input": call.input,
            })),
            Content::ToolResult(result) if !uncaptured.contains(result.call.as_ref()) => {
                results.push((result.call.clone(), result.output.clone(), result.error));
            }
            Content::ToolResult(_) => {}
            Content::Reasoning(_)
            | Content::ReasoningSummary { .. }
            | Content::Patch(_)
            | Content::Other(_)
            | Content::Unknown(_) => {}
        }
    }

    let mut lines = Vec::new();
    let conversation = matches!(row.role, Role::User | Role::Assistant | Role::Other(_));
    if conversation && (!text.is_empty() || !blocks.is_empty()) {
        let mut content: Vec<Value> = Vec::new();
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
        content.extend(blocks);
        let role = match row.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Other(_) => "assistant",
            _ => unreachable!("checked above"),
        };
        let content = if content.len() == 1 {
            match &content[0] {
                Value::Object(block)
                    if block.get("type").and_then(Value::as_str) == Some("text") =>
                {
                    block.get("text").cloned().unwrap_or(Value::Null)
                }
                other => other.clone(),
            }
        } else {
            Value::Array(content)
        };
        lines.push(json!({
            "role": role,
            "content": content,
            "message_id": row.source_id,
        }));
    }
    for (at, (call, output, error)) in results.into_iter().enumerate() {
        let mut line = Map::new();
        line.insert("role".into(), json!("tool"));
        line.insert("tool_call_id".into(), json!(call.as_ref()));
        line.insert(
            "content".into(),
            if output.is_null() {
                json!(UNCAPTURED_OUTPUT)
            } else {
                output
            },
        );
        if error {
            line.insert("is_error".into(), json!(true));
        }
        // One result a row shares its id with the conversation line; the rest get a suffix, so
        // every line keeps its own id.
        let id = if at == 0 && lines.is_empty() {
            row.source_id.clone()
        } else {
            format!("{}#{}", row.source_id, call.as_ref())
        };
        line.insert("message_id".into(), json!(id));
        lines.push(Value::Object(line));
    }
    lines
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::harnesstools::session::{ToolResult, ToolUse};

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
            turn_id: None,
            cwd: Some(PathBuf::from("/work/proj")),
            git_branch: Some("main".to_owned()),
        }
    }

    fn session(id: &str, messages: Vec<RehydrateMessage>) -> RehydrateSession {
        RehydrateSession {
            id: id.to_owned(),
            title: Some("Fix the thing".to_owned()),
            cwd: PathBuf::from("/work/proj"),
            original_cwd: Some(PathBuf::from("/work")),
            git_branch: Some("main".to_owned()),
            model: None,
            started_at: "2026-09-18T10:00:00Z".parse::<jiff::Timestamp>().unwrap(),
            messages,
            fork_of: None,
        }
    }

    #[rstest]
    fn writes_a_legacy_folder_locate_finds() {
        let dir = tempfile::tempdir().unwrap();
        let env = crate::env::MockEnv::install();
        env.set("HOME", dir.path());
        let session = session("01JTEST0000000000000000AAA", vec![
            row("m1", Role::User, vec![Content::Text("hi".to_owned())]),
            row("m2", Role::Assistant, vec![
                Content::Text("hello".to_owned()),
                Content::ToolUse(ToolUse {
                    id: ToolCallId::from("c1".to_owned()),
                    name: "bash".to_owned(),
                    input: json!({"cmd": "ls"}),
                }),
            ]),
            row("m3", Role::Tool, vec![Content::ToolResult(ToolResult {
                call: ToolCallId::from("c1".to_owned()),
                output: json!("file.txt"),
                error: false,
            })]),
        ]);

        let folder = rehydrate_into(dir.path(), &session).unwrap();
        assert_eq!(locate(dir.path(), session.id.as_str()), Some(folder.clone()));
        assert_eq!(folder.file_name().unwrap().to_str().unwrap(), "session_fix-the-thing_01JTEST0");

        let meta: Value =
            serde_json::from_slice(&std::fs::read(folder.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["session_id"], json!(session.id));
        assert_eq!(meta["environment"]["working_directory"], json!("/work/proj"));
        assert_eq!(meta["title"], json!("Fix the thing"));
        assert_eq!(meta["git_branch"], json!("main"));

        let lines: Vec<Value> = std::fs::read_to_string(folder.join("messages.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], json!({"role": "user", "content": "hi", "message_id": "m1"}));
        assert_eq!(
            lines[1],
            json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "id": "c1", "name": "bash", "input": {"cmd": "ls"}},
                ],
                "message_id": "m2",
            })
        );
        assert_eq!(
            lines[2],
            json!({"role": "tool", "tool_call_id": "c1", "content": "file.txt", "message_id": "m3"})
        );

        // Writing it again refuses to replace the transcript.
        assert!(matches!(
            rehydrate_into(dir.path(), &session),
            Err(RehydrateError::AlreadyExists(_))
        ));
    }

    #[rstest]
    fn uncaptured_calls_become_notes_and_drop_their_results() {
        let dir = tempfile::tempdir().unwrap();
        let env = crate::env::MockEnv::install();
        env.set("HOME", dir.path());
        let session = session("s1", vec![
            row("m1", Role::User, vec![Content::Text("run it".to_owned())]),
            row("m2", Role::Assistant, vec![Content::ToolUse(ToolUse {
                id: ToolCallId::from("c1".to_owned()),
                name: "bash".to_owned(),
                input: Value::Null,
            })]),
            row("m3", Role::Tool, vec![Content::ToolResult(ToolResult {
                call: ToolCallId::from("c1".to_owned()),
                output: Value::Null,
                error: false,
            })]),
        ]);
        let folder = rehydrate_into(dir.path(), &session).unwrap();
        let lines: Vec<Value> = std::fs::read_to_string(folder.join("messages.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        // The call became a note in its row's text; the result for it is gone.
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["role"], json!("assistant"));
        assert!(lines[1]["content"].as_str().unwrap().contains("ran a shell command"));
    }
}
