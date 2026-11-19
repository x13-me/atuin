//! Vibe sessions on disk.
//!
//! Vibe keeps its sessions under a *save dir*: `[session_logging].save_dir` in
//! `$VIBE_HOME/config.toml`, else `$MISTRAL_VIBE_SESSION_ROOT`, else `$VIBE_HOME/logs/session`.
//! Two formats live there side by side:
//!
//! - **Unified** (what current Vibe writes): `<save>/unified/<session_id>/`, a
//!   `mistral.vibe.unified-session-store/v1` store. `CURRENT` names the generation and snapshot
//!   sequence; `generations/<gen>/manifest.json` names the projection-state document (whole, or
//!   as content-addressed `chunks/<sha256>.json` arrays whose items are the history entries),
//!   the runtime state (session metadata: `cwd`, `parent_session_id`, `active_model`, ...) and a
//!   recovery journal `journal/<seq>.jsonl`. The journal's `projection_advanced` records replace
//!   the snapshot whole; `projection_delta`/`session_relocated` records carry ops over
//!   `snapshot.history.entries` (`append_entry`, `replace_entry`, `remove_entry`,
//!   `set_history_entries`). Folding snapshot + journal gives the public history entries: each
//!   an object with `id`, `created_at` (ms), `turn_id`, `type` (`message` with `role` and
//!   content blocks, `reasoning`, `effect` with `detail.tool_name`/`input` and a `state` that
//!   holds the output, `notice`, ...).
//! - **Legacy** (Vibe 2.x folders, still resumed by import): `<save>/<prefix>_<hint>_<short id>/`
//!   with `meta.json` (`session_id`, `start_time`, `environment.working_directory`,
//!   `origin_directory`, `title`, `parent_session_id`, ...) and `messages.jsonl`, one
//!   LLMMessage per line (`role`, `content` as a string or blocks, `tool_calls`, `tool_call_id`,
//!   `message_id`).
//!
//! Checkpoints are a byte offset into `messages.jsonl` for a legacy folder, and a journal
//! sequence (with the entry's digest) for a unified store.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt};
use jiff::Timestamp;
use serde_json::Value;
use tokio::sync::watch;
use typed_builder::TypedBuilder;

use super::home;
use crate::env::var_nonempty;
use crate::fs::tree_watcher::{FileStat, TreeWatcher};
use crate::harnesstools::resume;
use crate::harnesstools::session::model::{
    Content, MessageId, ParentKind, Role, TitleChange, TitleSource, ToolCallId, ToolResult, ToolUse,
};
use crate::harnesstools::session::{
    Checkpoint, Listener, Message, MessageError, Observable, RuntimeError, Session, SessionId,
    Sessions, WatchError, checkpoint_offset, modified_after, scan_sessions,
};
use crate::io::{FollowLines, PathLineReader, PooledReadLines};
use crate::json::jsonl::JsonlExt;
use crate::path::PathBufExt as _;
use crate::sync::BlockingPool;

/// The directory under the save dir where unified stores live.
pub(crate) const UNIFIED_DIR: &str = "unified";
/// Vibe's lease directory for running sessions: `<save>/active/<id>.lock`, an `flock` held
/// while Vibe has the session open.
pub(crate) const ACTIVE_DIR: &str = "active";

/// Where Vibe keeps its sessions by default (see the module docs).
pub(crate) fn default_root() -> PathBuf {
    save_dir()
}

/// The save dir: `$MISTRAL_VIBE_SESSION_ROOT`, else `[session_logging].save_dir` in
/// `$VIBE_HOME/config.toml`, else `$VIBE_HOME/logs/session`.
fn save_dir() -> PathBuf {
    if let Some(root) = var_nonempty("MISTRAL_VIBE_SESSION_ROOT") {
        return expand_tilde(Path::new(&root));
    }
    if let Some(dir) = configured("save_dir") {
        return dir;
    }
    home().joined("logs").joined("session")
}

/// The `[session_logging]` prefix of legacy session folders (default `session`).
pub(crate) fn session_prefix() -> String {
    configured("session_prefix")
        .and_then(|p| p.to_str().map(str::to_owned))
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "session".to_owned())
}

/// `[session_logging].<key>` in `$VIBE_HOME/config.toml`, when it is set and absolute.
fn configured(key: &str) -> Option<PathBuf> {
    let text = std::fs::read_to_string(home().joined("config.toml")).ok()?;
    let doc = text.parse::<toml_edit::DocumentMut>().ok()?;
    let value = doc.get("session_logging")?.get(key)?;
    let path = PathBuf::from(value.as_str()?);
    let path = expand_tilde(&path);
    path.is_absolute().then_some(path)
}

/// A leading `~` expanded against the home directory, as Vibe resolves its configured paths.
fn expand_tilde(path: &Path) -> PathBuf {
    match path.strip_prefix("~") {
        Ok(rest)
            if path.to_str().is_some_and(|p| {
                p == "~" || p.starts_with("~/") || (cfg!(windows) && p.starts_with("~\\"))
            }) =>
        {
            crate::dirs::home_dir().joined(rest)
        }
        _ => path.to_path_buf(),
    }
}

/// Where a session lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    /// A legacy folder: `<save>/<prefix>_<hint>_<short id>/`.
    Legacy(PathBuf),
    /// A unified store: `<save>/unified/<session_id>/`.
    Unified(PathBuf),
}

impl Location {
    /// The session's transcript: `messages.jsonl` for a legacy folder, `CURRENT` for a unified
    /// store.
    pub(crate) fn transcript(&self) -> PathBuf {
        match self {
            Self::Legacy(dir) => dir.join("messages.jsonl"),
            Self::Unified(dir) => dir.join("CURRENT"),
        }
    }
}

/// The session `id`'s store or folder under `root`: a unified store named by `id`, else a legacy
/// folder whose `meta.json` names it. `None` when it is not there.
pub(crate) fn locate(root: &Path, id: &str) -> Option<PathBuf> {
    locate_full(root, id).map(|location| match location {
        Location::Legacy(dir) | Location::Unified(dir) => dir,
    })
}

/// [`locate`], with the session's [`Location`] rather than just its directory.
pub(crate) fn locate_full(root: &Path, id: &str) -> Option<Location> {
    if !resume::is_plain_name(id) {
        return None;
    }
    let unified = root.join(UNIFIED_DIR).join(id);
    if unified.join("CURRENT").is_file() {
        return Some(Location::Unified(unified));
    }
    let prefix = format!("{}_", session_prefix());
    let wanted = id.to_owned();
    resume::find_file(root, move |path| {
        if path.file_name().is_none_or(|name| name != "meta.json") {
            return false;
        }
        let dir = path.parent().unwrap();
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if !name.starts_with(&prefix) {
            return false;
        }
        read_meta(path).is_some_and(|meta| meta.session_id == wanted)
    })
    .and_then(|meta| meta.parent().map(Path::to_path_buf))
    .map(Location::Legacy)
}

/// A legacy folder's `meta.json`, the parts Atuin reads.
#[derive(Debug, Clone, Default)]
pub(crate) struct Meta {
    pub(crate) session_id: String,
    pub(crate) cwd: Option<PathBuf>,
    pub(crate) parent: Option<String>,
    pub(crate) title: Option<String>,
    pub(crate) title_source: Option<String>,
    pub(crate) git_branch: Option<String>,
}

/// Read a legacy `meta.json`; `None` when it is not there or not an object.
pub(crate) fn read_meta(path: &Path) -> Option<Meta> {
    let value: Value = crate::json::js::from_slice(&std::fs::read(path).ok()?).ok()?;
    let object = value.as_object()?;
    let str_of = |key: &str| object.get(key).and_then(Value::as_str).map(str::to_owned);
    let cwd = object
        .get("environment")
        .and_then(|e| e.get("working_directory"))
        .and_then(Value::as_str)
        .map(PathBuf::from);
    Some(Meta {
        session_id: str_of("session_id").unwrap_or_default(),
        cwd,
        parent: str_of("parent_session_id"),
        title: str_of("title"),
        title_source: str_of("title_source"),
        git_branch: str_of("git_branch"),
    })
}

#[derive(Debug, Clone, TypedBuilder)]
pub struct VibeSessions {
    #[builder(default, setter(strip_option, into))]
    root: Option<PathBuf>,
    /// Runs every file read of the sessions this finds.
    pool: BlockingPool,
}

impl VibeSessions {
    fn resolve_root(&self) -> PathBuf {
        self.root.clone().unwrap_or_else(default_root)
    }
}

impl Sessions for VibeSessions {
    type Listener = VibeListener;

    fn last_active(&self) -> Option<SystemTime> {
        crate::fs::newest_modified(self.resolve_root())
    }

    fn listener(&self) -> Result<VibeListener, RuntimeError> {
        let root = self.resolve_root();
        if !root.is_dir() {
            return Err(RuntimeError::NotFound(root));
        }
        Ok(VibeListener {
            root,
            pool: self.pool.clone(),
        })
    }

    fn existing(
        &self,
    ) -> Result<impl Stream<Item = Result<VibeSession, RuntimeError>> + Send + 'static, RuntimeError>
    {
        let root = self.resolve_root();
        if !root.is_dir() {
            return Err(RuntimeError::NotFound(root));
        }
        let pool = self.pool.clone();
        let prefix = session_prefix();
        let sessions_pool = pool.clone();
        Ok(async_stream::stream! {
            let scan = pool
                .run(move || {
                    scan_sessions(root, |path, is_file| {
                        open_session(path, is_file, None, &sessions_pool, &prefix)
                    })
                })
                .await;
            match scan {
                Ok(items) => {
                    // Every file of a unified store names the same session; report it once.
                    let mut seen = std::collections::HashSet::new();
                    for item in items {
                        match item {
                            Ok(session) if seen.insert(session.id()) => yield Ok(session),
                            Ok(_) => {}
                            Err(err) => yield Err(err),
                        }
                    }
                }
                Err(cancelled) => yield Err(RuntimeError::Io(std::io::Error::other(cancelled))),
            }
        })
    }
}

impl Observable for super::Vibe {
    type Sessions = VibeSessions;

    fn sessions(&self, pool: BlockingPool) -> VibeSessions {
        VibeSessions::builder().pool(pool).build()
    }
}

/// Whether `path` can hold (part of) a session: anything under a `unified/` store, or a legacy
/// folder's `messages.jsonl`.
fn is_watched(path: &Path) -> bool {
    if path.components().any(|c| c.as_os_str() == UNIFIED_DIR) {
        return true;
    }
    path.file_name().is_some_and(|name| name == "messages.jsonl")
}

/// The session a watched `path` belongs to, once it can be told: its id and location. `prefix`
/// is the legacy folder prefix, resolved once by the caller (see [`session_prefix`]).
fn identify(path: &Path, prefix: &str) -> Option<(SessionId, Location)> {
    let components: Vec<_> = path.components().collect();
    // A file inside a unified store: the store dir is the component after `unified`.
    if let Some(at) = components.iter().position(|c| c.as_os_str() == UNIFIED_DIR) {
        let id = components.get(at + 1)?;
        let store: PathBuf = components[..=at + 1].iter().collect();
        if !store.join("CURRENT").is_file() {
            return None;
        }
        let id = id.as_os_str().to_str()?;
        return Some((SessionId::from(id.to_owned()), Location::Unified(store)));
    }
    // A legacy folder's transcript.
    if path.file_name().is_none_or(|name| name != "messages.jsonl") {
        return None;
    }
    let dir = path.parent()?;
    let name = dir.file_name()?.to_str()?;
    if !name.starts_with(&format!("{prefix}_")) {
        return None;
    }
    let meta = read_meta(&dir.join("meta.json"))?;
    if meta.session_id.is_empty() {
        return None;
    }
    Some((SessionId::from(meta.session_id), Location::Legacy(dir.to_path_buf())))
}

/// Build a session for a watched or scanned `path`, with `changes` when it is watched.
fn open_session(
    path: &Path,
    is_file: bool,
    changes: Option<watch::Receiver<FileStat>>,
    pool: &BlockingPool,
    prefix: &str,
) -> Option<VibeSession> {
    if !is_file {
        return None;
    }
    let (id, location) = identify(path, prefix)?;
    Some(VibeSession {
        id,
        location,
        changes,
        pool: pool.clone(),
    })
}

#[derive(Debug, Clone)]
pub struct VibeListener {
    root: PathBuf,
    pool: BlockingPool,
}

impl VibeListener {
    /// The live session over a watched file, read again on each change until it is one (a store
    /// mid-publication, a folder mid-creation), as pi's reader does for a headerless file.
    async fn await_session(
        path: PathBuf,
        mut stat: watch::Receiver<FileStat>,
        pool: BlockingPool,
        prefix: String,
    ) -> Option<VibeSession> {
        loop {
            stat.borrow_and_update();
            if let Some(session) = open_session(&path, true, Some(stat.clone()), &pool, &prefix) {
                return Some(session);
            }
            stat.changed().await.ok()?;
        }
    }
}

impl Listener for VibeListener {
    type Session = VibeSession;

    fn watch(self) -> impl Stream<Item = Result<VibeSession, WatchError>> + Send + 'static {
        let (root, pool) = (self.root, self.pool);
        let prefix = session_prefix();
        async_stream::stream! {
            let files = TreeWatcher::builder().filter(is_watched).watch(&root);
            let mut files = match files {
                Ok(files) => files,
                Err(err) => {
                    yield Err(WatchError::from(err));
                    return;
                }
            };
            let mut pending = FuturesUnordered::new();
            let mut files_done = false;
            loop {
                tokio::select! {
                    file = files.next(), if !files_done => match file {
                        Some(file) => {
                            let (path, stat) = file.into_parts();
                            pending.push(Self::await_session(
                                path.to_path_buf(),
                                stat,
                                pool.clone(),
                                prefix.clone(),
                            ));
                        }
                        None => files_done = true,
                    },
                    Some(session) = pending.next(), if !pending.is_empty() => {
                        if let Some(session) = session {
                            yield Ok(session);
                        }
                    }
                    else => break,
                }
            }
        }
    }
}

/// What a session's metadata says about it: where it works, its parent, its title, branch and
/// model.
#[derive(Debug, Clone, Default)]
struct Ctx {
    cwd: Option<PathBuf>,
    model: Option<String>,
    git_branch: Option<String>,
    parent: Option<(SessionId, Option<ParentKind>)>,
    title: Option<TitleChange>,
}

impl Ctx {
    fn from_meta(meta: &Meta) -> Self {
        let parent = meta.parent.as_ref().filter(|p| !p.is_empty()).map(|p| {
            // A subagent names its spawner; anything else with a parent is treated cautiously as one.
            (SessionId::from(p.clone()), Some(ParentKind::Subagent))
        });
        let title = meta.title.as_ref().map(|text| {
            let source = match meta.title_source.as_deref() {
                Some("manual") => TitleSource::Named,
                _ => TitleSource::Generated,
            };
            TitleChange::new(source, text)
        });
        Self {
            cwd: meta.cwd.clone(),
            model: None,
            git_branch: meta.git_branch.clone(),
            parent,
            title,
        }
    }

    fn from_store(metadata: &Value, envelope: &Value) -> Self {
        let str_of = |key: &str| metadata.get(key).and_then(Value::as_str);
        let parent = match (str_of("parent_session_id"), str_of("root_session_id")) {
            (Some(parent), Some(root)) if parent != root => {
                let kind = if metadata.get("subagent_spawn_key").is_some()
                    || str_of("started_by_session_id").is_some()
                {
                    Some(ParentKind::Subagent)
                } else {
                    None
                };
                Some((SessionId::from(parent.to_owned()), kind))
            }
            _ => None,
        };
        let title = envelope
            .get("snapshot")
            .and_then(|s| s.get("session"))
            .and_then(|s| s.get("title"))
            .and_then(Value::as_str)
            .map(|text| TitleChange::new(TitleSource::Generated, text));
        Self {
            cwd: str_of("cwd").map(PathBuf::from),
            model: str_of("active_model").map(str::to_owned),
            git_branch: str_of("git_branch").map(str::to_owned),
            parent,
            title,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VibeSession {
    id: SessionId,
    location: Location,
    /// Wakes [`Session::messages_from`] on each change to the store; `None` reads it once.
    changes: Option<watch::Receiver<FileStat>>,
    pool: BlockingPool,
}

impl VibeSession {
    #[must_use]
    pub fn open(id: SessionId, location: Location, pool: BlockingPool) -> Self {
        Self {
            id,
            location,
            changes: None,
            pool,
        }
    }

    /// The legacy folder's metadata, read now.
    fn meta(&self) -> Option<Meta> {
        match &self.location {
            Location::Legacy(dir) => read_meta(&dir.join("meta.json")),
            Location::Unified(_) => None,
        }
    }
}

impl Session for VibeSession {
    type Message = VibeMessage;

    fn id(&self) -> SessionId {
        self.id.clone()
    }

    fn messages_from(
        self,
        from: Option<Checkpoint>,
    ) -> impl Stream<Item = Result<(Checkpoint, VibeMessage), MessageError>> + Send + 'static {
        async_stream::stream! {
            let transcript = self.location.transcript();
            match &self.location {
                Location::Legacy(_) => {
                    let path = transcript;
                    let meta = self.meta();
                    let ctx = meta.as_ref().map(Ctx::from_meta).unwrap_or_default();
                    let start = match from {
                        Some(from) => checkpoint_offset(&path, &self.pool, from).await,
                        None => 0,
                    };
                    let over = from.is_some() && start == 0;
                    let lines = FollowLines::new(PooledReadLines::new(
                        PathLineReader::at(&path, start),
                        self.pool.clone(),
                    ));
                    let messages = match &self.changes {
                        Some(changes) => lines.follow(changes.clone()).left_stream(),
                        None => lines.read_to_end().right_stream(),
                    }
                    .js_json::<Value>();
                    for await item in messages {
                        match item {
                            Ok((line, value)) => {
                                let message = VibeMessage {
                                    line: LineKind::Legacy(value),
                                    ctx: ctx.clone(),
                                };
                                yield Ok((Checkpoint::after(&line).started_over(over), message));
                            }
                            Err(err) => yield Err(MessageError::from(err)),
                        }
                    }
                }
                Location::Unified(dir) => {
                    let dir = dir.clone();
                    let pool = self.pool.clone();
                    let mut after = from;
                    loop {
                        let read = pool
                            .run({
                                let dir = dir.clone();
                                move || read_store(&dir)
                            })
                            .await;
                        let state = match read {
                            Ok(Ok(state)) => state,
                            Ok(Err(err)) => {
                                yield Err(MessageError::from(crate::json::jsonl::JsonlError::from(err)));
                                break;
                            }
                            Err(cancelled) => {
                                yield Err(MessageError::from(crate::json::jsonl::JsonlError::from(
                                    std::io::Error::other(cancelled),
                                )));
                                break;
                            }
                        };
                        let ctx = Ctx::from_store(&state.metadata, &state.envelope);
                        let entries = &state.entries;
                        let start = match after {
                            None => 0,
                            Some(cp) => match entries.iter().position(|(seq, entry)| {
                                *seq == cp.at && cp.names(&canonical(entry))
                            }) {
                                Some(at) => at + 1,
                                None => 0,
                            },
                        };
                        let over = after.is_some() && start == 0;
                        for (seq, entry) in entries.iter().skip(start) {
                            let checkpoint = Checkpoint::new(*seq, &canonical(entry)).started_over(over);
                            yield Ok((
                                checkpoint,
                                VibeMessage {
                                    line: LineKind::Unified(entry.clone()),
                                    ctx: ctx.clone(),
                                },
                            ));
                        }
                        match entries.last() {
                            Some((seq, entry)) => {
                                after = Some(Checkpoint::new(*seq, &canonical(entry)));
                            }
                            None => after = None,
                        }
                        match &self.changes {
                            Some(changes) => {
                                if changes.clone().changed().await.is_err() {
                                    break;
                                }
                            }
                            None => break,
                        }
                    }
                }
            }
        }
    }

    fn read(&self) -> impl Stream<Item = Result<VibeMessage, MessageError>> + Send + 'static {
        let session = self.clone();
        async_stream::stream! {
            let mut messages = std::pin::pin!(session.messages_from(None));
            while let Some(item) = messages.next().await {
                match item {
                    Ok((_, message)) => yield Ok(message),
                    Err(err) => yield Err(err),
                }
            }
        }
    }

    async fn changed_after(mut self, since: SystemTime) -> Option<Self> {
        modified_after(self.changes.as_mut(), since).await?;
        Some(self)
    }
}

/// The store a unified session holds: its history entries, each with the journal sequence that
/// last wrote it, plus the session metadata and projection envelope.
pub(crate) struct UnifiedStore {
    pub(crate) entries: Vec<(u64, Value)>,
    pub(crate) metadata: Value,
    pub(crate) envelope: Value,
    /// A digest of every file the fold read, so a writer can tell the store changed under it.
    pub(crate) stamp: crate::harnesstools::sync::Stamp,
    /// The newest modification time among those files.
    pub(crate) modified: Option<SystemTime>,
}

/// Canonical JSON bytes of an entry, what its checkpoint digests.
fn canonical(entry: &Value) -> Vec<u8> {
    serde_json::to_vec(entry).unwrap_or_default()
}

/// Read `path`, folding its bytes into `folded` and its modification time into `newest`, so the
/// store's stamp and newest write are what the fold read.
fn read_store_file(
    path: &Path,
    folded: &mut Vec<u8>,
    newest: &mut Option<SystemTime>,
) -> std::io::Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    folded.extend_from_slice(&bytes);
    if let Ok(at) = std::fs::metadata(path).and_then(|m| m.modified()) {
        *newest = Some(newest.map_or(at, |t: SystemTime| t.max(at)));
    }
    Ok(bytes)
}

/// Read a unified store at `dir`: fold the projection snapshot with its recovery journal.
pub(crate) fn read_store(dir: &Path) -> std::io::Result<UnifiedStore> {
    let mut folded = Vec::new();
    let mut newest = None;
    macro_rules! read_json {
        ($path:expr) => {{
            let bytes = read_store_file($path, &mut folded, &mut newest)?;
            crate::json::js::from_slice(&bytes).map_err(|err| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string())
            })
        }};
    }
    let current: Value = read_json!(&dir.join("CURRENT"))?;
    let generation = current
        .get("generation")
        .map(|g| g.as_str().map(str::to_owned).unwrap_or_else(|| g.to_string()))
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "CURRENT names no generation")
        })?;
    let snapshot_sequence = current.get("snapshot_sequence").and_then(Value::as_u64).unwrap_or(0);
    let generation_dir = dir.join("generations").join(&generation);
    let manifest: Value = read_json!(&generation_dir.join("manifest.json"))?;

    // The projection document: the envelope at its path, with the history entries either in it
    // or in the pool chunks it names (whose items concatenate to the entry list).
    let projection = manifest.get("projection_state").cloned().unwrap_or(Value::Null);
    let mut envelope: Value =
        read_json!(&generation_dir.join(
            projection.get("path").and_then(Value::as_str).unwrap_or("projection_state.json"),
        ))?;
    let mut entries: Vec<(u64, Value)> = Vec::new();
    if let Some(chunks) = projection.get("chunks").and_then(Value::as_array) {
        for digest in chunks.iter().filter_map(Value::as_str) {
            let chunk: Value = read_json!(&dir.join("chunks").join(format!("{digest}.json")))?;
            if let Some(items) = chunk.as_array() {
                entries.extend(items.iter().map(|item| (snapshot_sequence, item.clone())));
            }
        }
    } else if let Some(list) = envelope
        .get("snapshot")
        .and_then(|s| s.get("history"))
        .and_then(|h| h.get("entries"))
        .and_then(Value::as_array)
    {
        entries = list.iter().map(|item| (snapshot_sequence, item.clone())).collect();
    }

    // The session metadata lives in the runtime state.
    let runtime = manifest.get("runtime_state").cloned().unwrap_or(Value::Null);
    let runtime_doc: Value = read_json!(
        &generation_dir
            .join(runtime.get("path").and_then(Value::as_str).unwrap_or("runtime_state.json"))
    )
    .unwrap_or(Value::Null);
    let metadata = runtime_doc.get("session_metadata").cloned().unwrap_or(Value::Null);

    // The recovery journal: projection records fold over the snapshot.
    if let Some(segment) =
        manifest.get("recovery_journal_segment").and_then(|s| s.get("path")).and_then(Value::as_str)
    {
        let journal = dir.join(segment);
        if let Ok(bytes) = read_store_file(&journal, &mut folded, &mut newest) {
            for line in bytes.split(|b| *b == b'\n') {
                if line.iter().all(|b| b.is_ascii_whitespace()) {
                    continue;
                }
                let Ok(record) = crate::json::js::from_slice::<Value>(line) else {
                    continue;
                };
                let sequence = record.get("sequence").and_then(Value::as_u64).unwrap_or(0);
                let payload = record.get("payload").cloned().unwrap_or(Value::Null);
                match record.get("type").and_then(Value::as_str) {
                    Some("projection_advanced") => {
                        if let Some(snapshot) = payload.get("snapshot") {
                            envelope = snapshot.clone();
                            entries = snapshot
                                .get("history")
                                .and_then(|h| h.get("entries"))
                                .and_then(Value::as_array)
                                .map(|list| {
                                    list.iter().map(|item| (sequence, item.clone())).collect()
                                })
                                .unwrap_or_default();
                        }
                    }
                    Some("projection_delta" | "session_relocated") => {
                        apply_delta(&mut entries, payload.get("delta"), sequence);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(UnifiedStore {
        entries,
        metadata,
        envelope,
        stamp: crate::harnesstools::sync::Stamp::of(&folded),
        modified: newest,
    })
}

/// Fold one journal record's projection `delta` over `entries`, tagging each entry it writes
/// with `sequence`.
fn apply_delta(entries: &mut Vec<(u64, Value)>, delta: Option<&Value>, sequence: u64) {
    let Some(ops) = delta.and_then(Value::as_array) else {
        return;
    };
    for op in ops {
        match op.get("op").and_then(Value::as_str) {
            Some("append_entry") => {
                if let Some(entry) = op.get("entry") {
                    entries.push((sequence, entry.clone()));
                }
            }
            Some("replace_entry") => {
                let Some(id) = op.get("id").and_then(Value::as_str) else {
                    continue;
                };
                if let Some(entry) = op.get("entry")
                    && let Some((seq, slot)) = entries
                        .iter_mut()
                        .find(|(_, e)| e.get("id").and_then(Value::as_str) == Some(id))
                {
                    *seq = sequence;
                    *slot = entry.clone();
                }
            }
            Some("remove_entry") => {
                let Some(id) = op.get("id").and_then(Value::as_str) else {
                    continue;
                };
                entries.retain(|(_, e)| e.get("id").and_then(Value::as_str) != Some(id));
            }
            Some("set_history_entries") => {
                if let Some(list) = op.get("entries").and_then(Value::as_array) {
                    *entries = list.iter().map(|item| (sequence, item.clone())).collect();
                }
            }
            _ => {}
        }
    }
}

/// One line of a session: a legacy `messages.jsonl` line, or a unified history entry.
#[derive(Debug, Clone)]
enum LineKind {
    Legacy(Value),
    Unified(Value),
}

/// One row of a Vibe session.
#[derive(Debug, Clone)]
pub struct VibeMessage {
    line: LineKind,
    ctx: Ctx,
}

impl VibeMessage {
    /// A text content block's text, as legacy content holds it (a string or blocks).
    fn legacy_blocks(content: &Value) -> Vec<Content> {
        match content {
            Value::String(text) => vec![Content::Text(text.clone())],
            Value::Array(blocks) => blocks
                .iter()
                .map(|block| match block.get("type").and_then(Value::as_str) {
                    Some("text") => Content::Text(
                        block.get("text").and_then(Value::as_str).unwrap_or_default().to_owned(),
                    ),
                    Some("tool_use") => Content::ToolUse(ToolUse {
                        id: ToolCallId::from(
                            block.get("id").and_then(Value::as_str).unwrap_or_default().to_owned(),
                        ),
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        input: block
                            .get("input")
                            .or_else(|| block.get("arguments"))
                            .cloned()
                            .unwrap_or(Value::Null),
                    }),
                    Some("tool_result") => Content::ToolResult(ToolResult {
                        call: ToolCallId::from(
                            block
                                .get("tool_use_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                        ),
                        output: block.get("content").cloned().unwrap_or(Value::Null),
                        error: block.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                    }),
                    _ => Content::Other(block.clone()),
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// An assistant message's `tool_calls`: `[{id, function: {name, arguments}}]`.
    fn legacy_tool_calls(calls: &Value) -> Vec<Content> {
        calls
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|call| {
                let function = call.get("function")?;
                Some(Content::ToolUse(ToolUse {
                    id: ToolCallId::from(
                        call.get("id").and_then(Value::as_str).unwrap_or_default().to_owned(),
                    ),
                    name: function
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    input: function.get("arguments").cloned().unwrap_or(Value::Null),
                }))
            })
            .collect()
    }

    /// A unified message entry's content blocks.
    fn unified_blocks(blocks: &Value) -> Vec<Content> {
        blocks
            .as_array()
            .into_iter()
            .flatten()
            .map(|block| match block.get("type").and_then(Value::as_str) {
                Some("text") => Content::Text(
                    block.get("text").and_then(Value::as_str).unwrap_or_default().to_owned(),
                ),
                _ => Content::Other(block.clone()),
            })
            .collect()
    }

    /// A unified `effect` entry: the call in `detail`, the result in its `state`.
    fn unified_effect(entry: &Value) -> Vec<Content> {
        let detail = entry.get("detail").cloned().unwrap_or(Value::Null);
        let state = entry.get("state").cloned().unwrap_or(Value::Null);
        let tool_call_id = entry
            .get("tool_call_id")
            .and_then(Value::as_str)
            .or_else(|| entry.get("id").and_then(Value::as_str))
            .unwrap_or_default();
        let name = detail.get("tool_name").and_then(Value::as_str).unwrap_or_default();
        let mut content = vec![Content::ToolUse(ToolUse {
            id: ToolCallId::from(tool_call_id.to_owned()),
            name: name.to_owned(),
            input: detail.get("input").cloned().unwrap_or(Value::Null),
        })];
        if let Some("completed" | "failed" | "cancelled" | "skipped") =
            state.get("status").and_then(Value::as_str)
        {
            let failed = state.get("status").and_then(Value::as_str) == Some("failed");
            let output =
                state.get("output").filter(|o| !o.is_null()).cloned().unwrap_or_else(|| {
                    Value::String(
                        state
                            .get("output_text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    )
                });
            content.push(Content::ToolResult(ToolResult {
                call: ToolCallId::from(tool_call_id.to_owned()),
                output,
                error: failed,
            }));
        }
        content
    }
}

impl Message for VibeMessage {
    fn id(&self) -> Option<MessageId> {
        match &self.line {
            LineKind::Legacy(line) => line
                .get("message_id")
                .and_then(Value::as_str)
                .map(|id| MessageId::from(id.to_owned())),
            LineKind::Unified(entry) => {
                entry.get("id").and_then(Value::as_str).map(|id| MessageId::from(id.to_owned()))
            }
        }
    }

    fn role(&self) -> Role {
        match &self.line {
            LineKind::Legacy(line) => match line.get("role").and_then(Value::as_str) {
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                Some("system") => Role::System,
                Some("tool") => Role::Tool,
                Some(other) => Role::Other(other.to_owned()),
                None => Role::Other("unknown".to_owned()),
            },
            LineKind::Unified(entry) => match entry.get("type").and_then(Value::as_str) {
                Some("message") => match entry.get("role").and_then(Value::as_str) {
                    Some("user") => Role::User,
                    Some("assistant") => Role::Assistant,
                    Some("system") => Role::System,
                    Some(other) => Role::Other(other.to_owned()),
                    None => Role::Other("message".to_owned()),
                },
                Some("reasoning") => Role::Other("reasoning".to_owned()),
                Some("effect") => Role::Other("effect".to_owned()),
                Some(other) => Role::Other(other.to_owned()),
                None => Role::Other("unknown".to_owned()),
            },
        }
    }

    fn timestamp(&self) -> Option<Timestamp> {
        match &self.line {
            // Legacy lines carry no timestamp; the folder's meta has the session's start.
            LineKind::Legacy(_) => None,
            LineKind::Unified(entry) => entry
                .get("created_at")
                .and_then(Value::as_u64)
                .and_then(|ms| i64::try_from(ms).ok())
                .and_then(|ms| Timestamp::from_millisecond(ms).ok()),
        }
    }

    fn content(&self) -> Vec<Content> {
        match &self.line {
            LineKind::Legacy(line) => {
                let role = line.get("role").and_then(Value::as_str);
                if role == Some("tool")
                    || line.get("tool_call_id").and_then(Value::as_str).is_some()
                {
                    // A tool result line: the call it answers, the output as content.
                    let output = line
                        .get("tool_result")
                        .filter(|r| !r.is_null())
                        .cloned()
                        .or_else(|| line.get("content").cloned())
                        .unwrap_or(Value::Null);
                    return vec![Content::ToolResult(ToolResult {
                        call: ToolCallId::from(
                            line.get("tool_call_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                        ),
                        output,
                        error: false,
                    })];
                }
                let mut content = Self::legacy_blocks(line.get("content").unwrap_or(&Value::Null));
                if let Some(calls) = line.get("tool_calls") {
                    content.extend(Self::legacy_tool_calls(calls));
                }
                content
            }
            LineKind::Unified(entry) => match entry.get("type").and_then(Value::as_str) {
                Some("message") => {
                    Self::unified_blocks(entry.get("content").unwrap_or(&Value::Null))
                }
                Some("reasoning") => vec![Content::Reasoning(
                    entry.get("text").and_then(Value::as_str).unwrap_or_default().to_owned(),
                )],
                Some("effect") => Self::unified_effect(entry),
                _ => Vec::new(),
            },
        }
    }

    fn model(&self) -> Option<String> {
        match &self.line {
            LineKind::Legacy(line)
                if line.get("role").and_then(Value::as_str) == Some("assistant") =>
            {
                self.ctx.model.clone()
            }
            LineKind::Unified(entry)
                if entry.get("type").and_then(Value::as_str) == Some("message")
                    && entry.get("role").and_then(Value::as_str) == Some("assistant") =>
            {
                self.ctx.model.clone()
            }
            _ => None,
        }
    }

    fn cwd(&self) -> Option<PathBuf> {
        self.ctx.cwd.clone()
    }

    fn git_branch(&self) -> Option<String> {
        self.ctx.git_branch.clone()
    }

    fn parent_session(&self) -> Option<SessionId> {
        self.ctx.parent.as_ref().map(|(id, _)| id.clone())
    }

    fn parent_kind(&self) -> Option<ParentKind> {
        self.ctx.parent.as_ref().and_then(|(_, kind)| *kind)
    }

    fn turn_id(&self) -> Option<String> {
        match &self.line {
            LineKind::Legacy(_) => None,
            LineKind::Unified(entry) => {
                entry.get("turn_id").and_then(Value::as_str).map(str::to_owned)
            }
        }
    }

    fn title(&self) -> Option<TitleChange> {
        self.ctx.title.clone()
    }
}

#[cfg(test)]
pub(crate) mod tests;
