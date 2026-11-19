use std::num::NonZeroUsize;
use std::time::Duration;

use futures::TryStreamExt;
use rstest::rstest;
use serde_json::{Value, json};

use super::*;

fn pool() -> BlockingPool {
    BlockingPool::new(NonZeroUsize::MIN)
}

/// A hermetic environment: `$HOME` at `home`, nothing else set, so `home()` and
/// [`session_prefix`] read nothing of the real machine.
fn mock_env(home: &Path) -> crate::env::MockEnv {
    let env = crate::env::MockEnv::install();
    env.set("HOME", home);
    env
}

/// Write a legacy session folder `<root>/<folder>/` with `meta.json` and `messages.jsonl`.
pub fn write_legacy(root: &Path, folder: &str, meta: &Value, lines: &[Value]) -> PathBuf {
    let dir = root.join(folder);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("meta.json"), meta.to_string()).unwrap();

    let mut body = lines.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
    if !body.is_empty() {
        body.push('\n');
    }
    std::fs::write(dir.join("messages.jsonl"), body).unwrap();
    dir
}

fn legacy_meta(id: &str) -> Value {
    json!({
        "session_id": id,
        "start_time": "2026-09-18T10:00:00.000Z",
        "environment": {"working_directory": "/work/proj"},
        "origin_directory": "/work",
        "title": "Fix the thing",
        "title_source": "manual",
        "parent_session_id": "parent-1",
        "git_branch": "main",
    })
}

/// Write a unified store `<root>/unified/<id>/`: `CURRENT` naming `gen0`, whose manifest points
/// at an inline projection document holding `entries`, a runtime state holding `metadata`, and a
/// recovery journal holding `journal` records.
pub fn write_unified(
    root: &Path,
    id: &str,
    entries: &[Value],
    journal: &[Value],
    metadata: &Value,
) -> PathBuf {
    let store = root.join(UNIFIED_DIR).join(id);
    let gen_dir = store.join("generations").join("gen0");
    std::fs::create_dir_all(&gen_dir).unwrap();
    std::fs::create_dir_all(store.join("journal")).unwrap();
    let envelope = json!({
        "snapshot": {
            "session": {"title": "Unified title"},
            "history": {"entries": entries},
        }
    });
    std::fs::write(gen_dir.join("projection_state.json"), envelope.to_string()).unwrap();
    std::fs::write(
        gen_dir.join("runtime_state.json"),
        json!({"session_metadata": metadata}).to_string(),
    )
    .unwrap();
    std::fs::write(
        gen_dir.join("manifest.json"),
        json!({
            "projection_state": {"path": "projection_state.json"},
            "runtime_state": {"path": "runtime_state.json"},
            "recovery_journal_segment": {"path": "journal/1.jsonl"},
        })
        .to_string(),
    )
    .unwrap();
    let mut body = journal.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
    if !body.is_empty() {
        body.push('\n');
    }
    std::fs::write(store.join("journal").join("1.jsonl"), body).unwrap();
    std::fs::write(
        store.join("CURRENT"),
        json!({"generation": "gen0", "snapshot_sequence": 1}).to_string(),
    )
    .unwrap();
    store
}

fn user_entry(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "created_at": 1_700_000_000_000u64,
        "turn_id": "t1",
        "type": "message",
        "role": "user",
        "content": [{"type": "text", "text": text}],
    })
}

fn assistant_entry(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "created_at": 1_700_000_001_000u64,
        "turn_id": "t2",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    })
}

fn reasoning_entry(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "created_at": 1_700_000_000_500u64,
        "turn_id": "t2",
        "type": "reasoning",
        "text": text,
    })
}

fn effect_entry(id: &str, call: &str, status: &str, output: &Value) -> Value {
    json!({
        "id": id,
        "created_at": 1_700_000_002_000u64,
        "turn_id": "t2",
        "type": "effect",
        "tool_call_id": call,
        "detail": {"tool_name": "bash", "input": {"cmd": "ls"}},
        "state": {"status": status, "output": output},
    })
}

fn open(store: PathBuf, id: &str) -> VibeSession {
    let location = if store.file_name().is_some_and(|n| n == id) {
        Location::Unified(store)
    } else {
        Location::Legacy(store)
    };
    VibeSession::open(SessionId::from(id.to_owned()), location, pool())
}

#[rstest]
#[tokio::test]
async fn existing_finds_legacy_and_unified_sessions_once_each() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    write_legacy(dir.path(), "session_hint_aaa", &legacy_meta("legacy-1"), &[
        json!({"role": "user", "content": "hi", "message_id": "m1"}),
    ]);
    write_unified(
        dir.path(),
        "unified-1",
        &[user_entry("e1", "hello")],
        &[],
        &json!({"cwd": "/w", "active_model": "mistral-large-4-0"}),
    );
    // A folder without the prefix is no session, and nor is a stray meta.json.
    std::fs::create_dir_all(dir.path().join("notasession")).unwrap();
    std::fs::write(dir.path().join("notasession").join("messages.jsonl"), "{\"role\":\"user\"}\n")
        .unwrap();

    let sessions = VibeSessions::builder().root(dir.path().to_path_buf()).pool(pool()).build();
    let mut ids: Vec<String> =
        sessions.existing().unwrap().map(|s| s.unwrap().id().into()).collect().await;
    ids.sort();
    assert_eq!(ids, vec!["legacy-1".to_string(), "unified-1".to_string()]);
}

#[rstest]
fn listener_and_scan_report_a_missing_root() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    let sessions = VibeSessions::builder().root(dir.path().join("nope")).pool(pool()).build();
    assert!(matches!(sessions.listener(), Err(RuntimeError::NotFound(_))));
    assert!(matches!(sessions.existing(), Err(RuntimeError::NotFound(_))));
}

#[rstest]
fn locate_finds_both_formats_and_rejects_odd_names() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    write_legacy(dir.path(), "session_hint_aaa", &legacy_meta("legacy-1"), &[]);
    write_unified(dir.path(), "unified-1", &[], &[], &json!({}));

    assert_eq!(
        locate(dir.path(), "unified-1"),
        Some(dir.path().join(UNIFIED_DIR).join("unified-1"))
    );
    assert_eq!(locate(dir.path(), "legacy-1"), Some(dir.path().join("session_hint_aaa")));
    assert_eq!(locate(dir.path(), "../escape"), None);
    assert_eq!(locate(dir.path(), "missing"), None);
}

#[rstest]
fn last_active_is_the_newest_file_under_the_root() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    let empty = tempfile::tempdir().unwrap();
    let sessions =
        |root: &Path| VibeSessions::builder().root(root.to_path_buf()).pool(pool()).build();
    assert_eq!(sessions(empty.path()).last_active(), None);
    write_legacy(dir.path(), "session_hint_aaa", &legacy_meta("legacy-1"), &[]);
    assert!(sessions(dir.path()).last_active().is_some());
}

#[rstest]
fn save_dir_follows_the_env_then_the_config_then_the_home() {
    let dir = tempfile::tempdir().unwrap();
    let env = mock_env(dir.path());
    assert_eq!(default_root(), dir.path().join(".vibe").join("logs").join("session"));

    let vh = dir.path().join("vh");
    std::fs::create_dir_all(&vh).unwrap();
    env.set("VIBE_HOME", &vh);
    assert_eq!(default_root(), vh.join("logs").join("session"));

    // A relative save_dir is relative to wherever Vibe started: ignored.
    std::fs::write(vh.join("config.toml"), "[session_logging]\nsave_dir = \"relative/dir\"\n")
        .unwrap();
    assert_eq!(default_root(), vh.join("logs").join("session"));

    std::fs::write(vh.join("config.toml"), "[session_logging]\nsave_dir = \"/srv/sessions\"\n")
        .unwrap();
    assert_eq!(default_root(), PathBuf::from("/srv/sessions"));

    // The session root env var beats all of it.
    env.set("MISTRAL_VIBE_SESSION_ROOT", dir.path().join("elsewhere"));
    assert_eq!(default_root(), dir.path().join("elsewhere"));
}

#[rstest]
#[tokio::test]
async fn legacy_session_reads_lines_with_byte_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    let lines = [
        json!({"role": "user", "content": "hi", "message_id": "m1"}),
        json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": {"cmd": "ls"}}}],
            "message_id": "m2",
        }),
        json!({"role": "tool", "tool_call_id": "c1", "content": "file.txt", "message_id": "m3"}),
    ];
    let folder = write_legacy(dir.path(), "session_hint_aaa", &legacy_meta("legacy-1"), &lines);

    let read: Vec<VibeMessage> =
        open(folder.clone(), "legacy-1").read().try_collect().await.unwrap();
    assert_eq!(read.len(), 3);
    assert_eq!(read[0].role(), Role::User);
    assert_eq!(read[0].id(), Some(MessageId::from("m1".to_owned())));
    assert_eq!(read[0].content(), vec![Content::Text("hi".to_owned())]);
    assert!(matches!(read[1].content().as_slice(), [Content::Text(t), Content::ToolUse(u)]
            if t == "hello" && u.name == "bash" && u.id.as_ref() == "c1" && u.input == json!({"cmd": "ls"})));
    assert_eq!(read[2].role(), Role::Tool);
    assert!(matches!(read[2].content().as_slice(), [Content::ToolResult(r)]
            if r.call.as_ref() == "c1" && !r.error && r.output == json!("file.txt")));
    // The session's context comes from the folder's meta.
    assert_eq!(read[0].cwd(), Some(PathBuf::from("/work/proj")));
    assert_eq!(read[0].parent_session(), Some(SessionId::from("parent-1".to_owned())));
    assert_eq!(read[0].parent_kind(), Some(ParentKind::Subagent));
    assert_eq!(read[0].title(), Some(TitleChange::new(TitleSource::Named, "Fix the thing")));
    assert_eq!(read[0].timestamp(), None);

    // Checkpoints are byte offsets into messages.jsonl; resuming from the last yields nothing,
    // from the first yields the rest.
    let pairs: Vec<(Checkpoint, VibeMessage)> =
        open(folder.clone(), "legacy-1").messages_from(None).try_collect().await.unwrap();
    assert_eq!(pairs.len(), 3);
    let body_len = std::fs::read(folder.join("messages.jsonl")).unwrap().len() as u64;
    assert_eq!(pairs.last().map(|(cp, _)| cp.at), Some(body_len));

    let rest: Vec<_> = open(folder.clone(), "legacy-1")
        .messages_from(Some(pairs[2].0))
        .try_collect()
        .await
        .unwrap();
    assert!(rest.is_empty());
    let rest: Vec<_> = open(folder.clone(), "legacy-1")
        .messages_from(Some(pairs[0].0))
        .try_collect()
        .await
        .unwrap();
    assert_eq!(rest.len(), 2);

    // A checkpoint naming another line reads from the start, marked as started over.
    let bogus = Checkpoint::new(pairs[0].0.at, b"some other line");
    let all: Vec<(Checkpoint, VibeMessage)> =
        open(folder, "legacy-1").messages_from(Some(bogus)).try_collect().await.unwrap();
    assert_eq!(all.len(), 3);
    assert!(all.iter().all(|(cp, _)| cp.generation == 1));
}

#[rstest]
#[tokio::test]
async fn unified_session_folds_the_snapshot_with_its_journal() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    let snapshot = vec![
        user_entry("e1", "hi"),
        reasoning_entry("e0", "let me think"),
        assistant_entry("e2", "hello"),
        effect_entry("e3", "c1", "running", &Value::Null),
    ];
    let journal = [
        json!({
            "type": "projection_delta",
            "sequence": 2,
            "payload": {"delta": [{"op": "append_entry", "entry": effect_entry("e4", "c1", "completed", &json!("file.txt"))}]},
        }),
        json!({
            "type": "projection_delta",
            "sequence": 3,
            "payload": {"delta": [{"op": "replace_entry", "id": "e2", "entry": assistant_entry("e2", "hello there")}]},
        }),
        json!({
            "type": "projection_delta",
            "sequence": 4,
            "payload": {"delta": [{"op": "remove_entry", "id": "e3"}]},
        }),
    ];
    let metadata = json!({
        "cwd": "/work/proj",
        "origin_cwd": "/work",
        "active_model": "mistral-large-4-0",
        "parent_session_id": "p1",
        "root_session_id": "r1",
        "subagent_spawn_key": "k",
    });
    let store = write_unified(dir.path(), "unified-1", &snapshot, &journal, &metadata);

    let msgs: Vec<VibeMessage> = open(store, "unified-1").read().try_collect().await.unwrap();
    let ids: Vec<String> = msgs.iter().map(|m| m.id().unwrap().into()).collect();
    assert_eq!(ids, vec!["e1", "e0", "e2", "e4"]);

    assert_eq!(msgs[0].role(), Role::User);
    assert_eq!(msgs[0].content(), vec![Content::Text("hi".to_owned())]);
    assert_eq!(msgs[1].role(), Role::Other("reasoning".to_owned()));
    assert_eq!(msgs[1].content(), vec![Content::Reasoning("let me think".to_owned())]);
    assert_eq!(msgs[2].content(), vec![Content::Text("hello there".to_owned())]);
    assert!(matches!(msgs[3].content().as_slice(), [Content::ToolUse(u), Content::ToolResult(r)]
            if u.name == "bash" && u.id.as_ref() == "c1" && r.call.as_ref() == "c1"
                && r.output == json!("file.txt") && !r.error));
    assert_eq!(msgs[3].role(), Role::Other("effect".to_owned()));

    // The session's context comes from the runtime state and the envelope.
    assert_eq!(msgs[0].cwd(), Some(PathBuf::from("/work/proj")));
    assert_eq!(msgs[0].parent_session(), Some(SessionId::from("p1".to_owned())));
    assert_eq!(msgs[0].parent_kind(), Some(ParentKind::Subagent));
    assert_eq!(msgs[0].title(), Some(TitleChange::new(TitleSource::Generated, "Unified title")));
    assert_eq!(msgs[0].turn_id(), Some("t1".to_owned()));
    assert_eq!(msgs[0].model(), None);
    assert_eq!(msgs[2].model(), Some("mistral-large-4-0".to_owned()));
    assert_eq!(msgs[0].timestamp(), Timestamp::from_millisecond(1_700_000_000_000).ok());
}

#[rstest]
#[tokio::test]
async fn unified_checkpoints_are_journal_sequences_with_entry_digests() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    let snapshot = vec![user_entry("e1", "hi"), assistant_entry("e2", "hello")];
    let journal = [json!({
        "type": "projection_delta",
        "sequence": 5,
        "payload": {"delta": [{"op": "append_entry", "entry": reasoning_entry("e3", "hmm")}]},
    })];
    let store = write_unified(dir.path(), "unified-1", &snapshot, &journal, &json!({}));

    let pairs: Vec<(Checkpoint, VibeMessage)> =
        open(store.clone(), "unified-1").messages_from(None).try_collect().await.unwrap();
    assert_eq!(pairs.len(), 3);
    // Snapshot entries carry the snapshot sequence; the journal's append carries its own.
    assert_eq!(pairs[0].0.at, 1);
    assert_eq!(pairs[2].0.at, 5);

    // Resuming just past the first entry yields the rest.
    let rest: Vec<_> = open(store.clone(), "unified-1")
        .messages_from(Some(pairs[0].0))
        .try_collect()
        .await
        .unwrap();
    assert_eq!(rest.len(), 2);

    // A checkpoint that names no entry reads from the start, marked as started over.
    let bogus = Checkpoint::new(5, b"no such entry");
    let all: Vec<(Checkpoint, VibeMessage)> =
        open(store, "unified-1").messages_from(Some(bogus)).try_collect().await.unwrap();
    assert_eq!(all.len(), 3);
    assert!(all.iter().all(|(cp, _)| cp.generation == 1));
}

#[rstest]
#[tokio::test]
async fn a_chunked_projection_reads_its_entries_from_the_chunk_pool() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    let store = dir.path().join(UNIFIED_DIR).join("chunked-1");
    let gen_dir = store.join("generations").join("gen0");
    std::fs::create_dir_all(&gen_dir).unwrap();
    std::fs::create_dir_all(store.join("chunks")).unwrap();
    std::fs::write(
        store.join("chunks").join("aaaa.json"),
        json!([user_entry("e1", "hi"), assistant_entry("e2", "hello")]).to_string(),
    )
    .unwrap();
    std::fs::write(
        store.join("chunks").join("bbbb.json"),
        json!([reasoning_entry("e3", "hmm")]).to_string(),
    )
    .unwrap();
    std::fs::write(
        gen_dir.join("projection_state.json"),
        json!({"snapshot": {"history": {"entries": []}}}).to_string(),
    )
    .unwrap();
    std::fs::write(
        gen_dir.join("runtime_state.json"),
        json!({"session_metadata": {"cwd": "/w"}}).to_string(),
    )
    .unwrap();
    std::fs::write(
        gen_dir.join("manifest.json"),
        json!({
            "projection_state": {"chunks": ["aaaa", "bbbb"]},
            "runtime_state": {"path": "runtime_state.json"},
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        store.join("CURRENT"),
        json!({"generation": "gen0", "snapshot_sequence": 7}).to_string(),
    )
    .unwrap();

    let pairs: Vec<(Checkpoint, VibeMessage)> =
        open(store, "chunked-1").messages_from(None).try_collect().await.unwrap();
    let ids: Vec<String> = pairs.iter().map(|(_, m)| m.id().unwrap().into()).collect();
    assert_eq!(ids, vec!["e1", "e2", "e3"]);
    assert!(pairs.iter().all(|(cp, _)| cp.at == 7));
}

#[rstest]
#[tokio::test]
async fn a_projection_advanced_record_replaces_the_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    let snapshot = vec![user_entry("e1", "old")];
    let journal = [json!({
        "type": "projection_advanced",
        "sequence": 9,
        "payload": {"snapshot": {"history": {"entries": [user_entry("e9", "new"), assistant_entry("e10", "fresh")]}}},
    })];
    let store = write_unified(dir.path(), "unified-1", &snapshot, &journal, &json!({}));

    let pairs: Vec<(Checkpoint, VibeMessage)> =
        open(store, "unified-1").messages_from(None).try_collect().await.unwrap();
    let ids: Vec<String> = pairs.iter().map(|(_, m)| m.id().unwrap().into()).collect();
    assert_eq!(ids, vec!["e9", "e10"]);
    assert!(pairs.iter().all(|(cp, _)| cp.at == 9));
}

#[rstest]
#[tokio::test]
async fn watch_emits_a_unified_store_once_current_names_its_generation() {
    let dir = tempfile::tempdir().unwrap();
    let _env = mock_env(dir.path());
    let store = dir.path().join(UNIFIED_DIR).join("s1");
    let gen_dir = store.join("generations").join("gen0");
    std::fs::create_dir_all(&gen_dir).unwrap();
    std::fs::create_dir_all(store.join("chunks")).unwrap();
    // Mid-publication: a chunk is there, but no CURRENT names the generation yet.
    std::fs::write(
        store.join("chunks").join("aaaa.json"),
        json!([user_entry("e1", "hi")]).to_string(),
    )
    .unwrap();

    let listener = VibeSessions::builder().root(dir.path().to_path_buf()).pool(pool()).build();
    let mut sessions = std::pin::pin!(listener.listener().unwrap().watch());

    std::fs::write(
        gen_dir.join("projection_state.json"),
        json!({"snapshot": {"history": {"entries": []}}}).to_string(),
    )
    .unwrap();
    std::fs::write(
        gen_dir.join("runtime_state.json"),
        json!({"session_metadata": {"cwd": "/w"}}).to_string(),
    )
    .unwrap();
    std::fs::write(
        gen_dir.join("manifest.json"),
        json!({"projection_state": {"chunks": ["aaaa"]}, "runtime_state": {"path": "runtime_state.json"}})
            .to_string(),
    )
    .unwrap();
    std::fs::write(
        store.join("CURRENT"),
        json!({"generation": "gen0", "snapshot_sequence": 1}).to_string(),
    )
    .unwrap();

    let session = tokio::time::timeout(Duration::from_secs(10), sessions.next())
        .await
        .expect("watch() did not emit a session within 10s")
        .unwrap()
        .unwrap();
    assert_eq!(session.id(), SessionId::from("s1".to_owned()));
    // A watched session follows its store, so it does not end: take its first message only.
    let mut messages = std::pin::pin!(session.messages_from(None));
    let (_, message) = tokio::time::timeout(Duration::from_secs(10), messages.next())
        .await
        .expect("the session did not yield its entry within 10s")
        .unwrap()
        .unwrap();
    assert_eq!(message.content(), vec![Content::Text("hi".to_owned())]);
    assert_eq!(message.cwd(), Some(PathBuf::from("/w")));
}
