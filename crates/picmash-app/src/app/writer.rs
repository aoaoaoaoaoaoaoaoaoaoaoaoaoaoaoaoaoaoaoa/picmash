use super::*;
use std::{
    fmt,
    marker::PhantomData,
    sync::mpsc::{self, Receiver, SyncSender},
    time::Instant,
};
use tracing::{Span, debug_span, field, warn};

const DEFAULT_SLOW_WRITER_COMMAND_MS: u128 = 50;
const BULK_SLOW_WRITER_COMMAND_MS: u128 = 100;

trait ErasedWriterCommand: Send {
    fn label(&self) -> &'static str;
    fn apply_box(self: Box<Self>, store: &mut Store) -> anyhow::Result<Box<dyn Any + Send>>;
}

fn slow_writer_threshold_ms(label: &str) -> u128 {
    match label {
        "upsert_external_stream_batch" | "bootstrap_maintenance_batch" => {
            BULK_SLOW_WRITER_COMMAND_MS
        }
        _ => DEFAULT_SLOW_WRITER_COMMAND_MS,
    }
}

struct ClosureCommand<T, F> {
    label: &'static str,
    f: Option<F>,
    _output: PhantomData<fn() -> T>,
}

impl<T, F> ClosureCommand<T, F> {
    fn forge(label: &'static str, f: F) -> Self {
        Self {
            label,
            f: Some(f),
            _output: PhantomData,
        }
    }
}

impl<T, F> ErasedWriterCommand for ClosureCommand<T, F>
where
    T: Send + 'static,
    F: FnOnce(&mut Store) -> anyhow::Result<T> + Send + 'static,
{
    fn label(&self) -> &'static str {
        self.label
    }

    fn apply_box(mut self: Box<Self>, store: &mut Store) -> anyhow::Result<Box<dyn Any + Send>> {
        let f = self
            .f
            .take()
            .context("writer command consumed more than once")?;
        f(store).map(|output| Box::new(output) as Box<dyn Any + Send>)
    }
}

struct WriterRequest {
    command_id: String,
    command: Box<dyn ErasedWriterCommand>,
    parent_span: Span,
    reply: SyncSender<anyhow::Result<Box<dyn Any + Send>>>,
}

enum WriterMessage {
    Execute(WriterRequest),
    Shutdown,
}

pub(super) struct DbWriter {
    tx: mpsc::Sender<WriterMessage>,
    join: Mutex<Option<thread::JoinHandle<()>>>,
}

impl fmt::Debug for DbWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DbWriter").finish_non_exhaustive()
    }
}

impl DbWriter {
    pub(super) fn spawn(db_path: &Path) -> anyhow::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let db_path = db_path.to_path_buf();
        let (boot_tx, boot_rx) = mpsc::sync_channel(1);
        let join = thread::Builder::new()
            .name("picmash-db-writer".to_owned())
            .spawn(move || {
                let mut store = match Store::open_hot(&db_path) {
                    Ok(store) => {
                        let _ = boot_tx.send(Ok(()));
                        store
                    }
                    Err(error) => {
                        let _ = boot_tx.send(Err(anyhow::anyhow!("{error:#}")));
                        return;
                    }
                };
                drain_writer_loop(&mut store, rx);
            })
            .context("spawning db writer thread")?;
        boot_rx
            .recv()
            .context("waiting for db writer boot")?
            .context("booting db writer")?;
        Ok(Self {
            tx,
            join: Mutex::new(Some(join)),
        })
    }

    pub(super) fn perform<T, F>(&self, label: &'static str, f: F) -> anyhow::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Store) -> anyhow::Result<T> + Send + 'static,
    {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let command =
            Box::new(ClosureCommand::<T, F>::forge(label, f)) as Box<dyn ErasedWriterCommand>;
        let command_id = crate::telemetry::fresh_writer_command_id();
        self.tx
            .send(WriterMessage::Execute(WriterRequest {
                command_id,
                command,
                parent_span: Span::current(),
                reply: reply_tx,
            }))
            .map_err(|_| anyhow::anyhow!("sending writer command `{label}`"))?;
        let reply = reply_rx.recv().context("waiting for writer reply")?;
        let output = reply?;
        output
            .downcast::<T>()
            .map(|boxed| *boxed)
            .map_err(|_| anyhow::anyhow!("writer reply type mismatch for {label}"))
    }

    fn shutdown(&self) {
        let _ = self.tx.send(WriterMessage::Shutdown);
        if let Some(join) = self.join.lock().take()
            && let Err(error) = join.join()
        {
            warn!(?error, "db writer thread panicked during shutdown");
        }
    }
}

impl Drop for DbWriter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn drain_writer_loop(store: &mut Store, rx: Receiver<WriterMessage>) {
    while let Ok(message) = rx.recv() {
        match message {
            WriterMessage::Execute(request) => {
                let WriterRequest {
                    command_id,
                    command,
                    parent_span,
                    reply,
                } = request;
                let label = command.label();
                let started = Instant::now();
                let span = debug_span!(
                    parent: &parent_span,
                    "writer.commit",
                    writer_cmd_id = %command_id,
                    label,
                    elapsed_ms = field::Empty,
                );
                let result = {
                    let _entered = span.enter();
                    command.apply_box(store).map_err(|error| {
                        anyhow::anyhow!("writer command `{label}` failed: {error:#}")
                    })
                };
                let elapsed_ms = started.elapsed().as_millis();
                span.record("elapsed_ms", field::display(elapsed_ms));
                if elapsed_ms > slow_writer_threshold_ms(label) {
                    let _entered = span.enter();
                    warn!(writer_cmd_id = %command_id, label, elapsed_ms, "slow writer command");
                }
                if let Err(error) = &result {
                    let _entered = span.enter();
                    warn!(
                        writer_cmd_id = %command_id,
                        label,
                        elapsed_ms,
                        error = %format!("{error:#}"),
                        "writer command failed"
                    );
                }
                let _ = reply.send(result);
            }
            WriterMessage::Shutdown => break,
        }
    }
}

impl AppState {
    pub(super) fn read_store(&self) -> anyhow::Result<Store> {
        Store::open_hot(&self.db_path)
    }

    pub(super) fn with_locked_store_write<T>(
        &self,
        f: impl FnOnce(&mut Store) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let mut store = Store::open_hot(&self.db_path)?;
        f(&mut store)
    }

    pub(super) fn with_write_store<T, F>(&self, label: &'static str, f: F) -> anyhow::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Store) -> anyhow::Result<T> + Send + 'static,
    {
        self.writer.perform(label, f)
    }

    pub(super) fn invalidate_session_field_cache(&self) {
        *self.session_field_cache.write() = None;
    }

    pub(super) fn replace_session_field_cache(&self, field: SessionField) {
        *self.session_field_cache.write() = Some(field);
    }
}
