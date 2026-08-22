//! Effect interpreter for the remote transition calculus.

use std::{
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, ensure};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};

use super::{
    CatalogIntent, Epoch, FetchIntent, FetchSettlement, Harvest, Harvester, Machine, Moment,
    Prepared, RemoteStore, Summary,
};
use crate::{configuration::RemoteConfig, xdg::Lair};

const EFFECT_CAPACITY: usize = 2;
const DORMANT_WAIT: Duration = Duration::from_hours(24);

pub enum Effect {
    Catalog {
        intent: CatalogIntent,
        result: std::result::Result<Harvest, String>,
    },
    Fetch(Box<FetchEffect>),
}

pub struct FetchEffect {
    intent: FetchIntent,
    result: std::result::Result<Prepared, String>,
}

pub struct Reactor {
    born: Instant,
    epoch: Epoch,
    active: bool,
    poisoned: bool,
    machine: Machine,
    store: RemoteStore,
    completions: Receiver<Effect>,
    catalog_lane: Option<Sender<CatalogIntent>>,
    fetch_lane: Option<Sender<FetchIntent>>,
    threads: Vec<JoinHandle<()>>,
}

impl Reactor {
    pub fn open(lair: &Lair, config: &RemoteConfig) -> Result<Self> {
        let born = Instant::now();
        let cache = lair.remote_cache();
        let harvester = Arc::new(Harvester::new(cache.clone())?);
        let store = RemoteStore::open(&lair.remote_database(), cache)?;
        let mut machine = Machine::new(config, Epoch::INITIAL, Moment::ZERO)?;
        restore(&mut machine, &store, config)?;
        let (completion_tx, completions) = bounded(EFFECT_CAPACITY);
        let (catalog_lane, catalog_rx) = bounded::<CatalogIntent>(1);
        let (fetch_lane, fetch_rx) = bounded::<FetchIntent>(1);
        let catalog_harvester = Arc::clone(&harvester);
        let catalog_completions = completion_tx.clone();
        let catalog_thread = thread::Builder::new()
            .name("picmash-remote-catalog".to_owned())
            .spawn(move || {
                for intent in catalog_rx {
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        catalog_harvester.catalog(
                            intent.source(),
                            intent.config(),
                            intent.ordinal(),
                        )
                    }))
                    .map_or_else(
                        |_| Err("remote catalog effect panicked".to_owned()),
                        |result| result.map_err(|error| format!("{error:#}")),
                    );
                    if catalog_completions
                        .send(Effect::Catalog { intent, result })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .context("raise bounded remote catalog lane")?;
        let fetch_thread = match thread::Builder::new()
            .name("picmash-remote-fetch".to_owned())
            .spawn(move || {
                for intent in fetch_rx {
                    let result =
                        catch_unwind(AssertUnwindSafe(|| harvester.fetch(intent.discovery())))
                            .map_or_else(
                                |_| Err("remote fetch effect panicked".to_owned()),
                                |result| result.map_err(|error| format!("{error:#}")),
                            );
                    if completion_tx
                        .send(Effect::Fetch(Box::new(FetchEffect { intent, result })))
                        .is_err()
                    {
                        break;
                    }
                }
            }) {
            Ok(thread) => thread,
            Err(error) => {
                drop(catalog_lane);
                let _joined = catalog_thread.join();
                return Err(error).context("raise bounded remote fetch lane");
            }
        };
        Ok(Self {
            born,
            epoch: Epoch::INITIAL,
            active: false,
            poisoned: false,
            machine,
            store,
            completions,
            catalog_lane: Some(catalog_lane),
            fetch_lane: Some(fetch_lane),
            threads: vec![catalog_thread, fetch_thread],
        })
    }

    pub fn activate(&mut self) -> Result<()> {
        self.active = true;
        self.drive()
    }

    pub fn reconfigure(&mut self, config: &RemoteConfig) -> Result<()> {
        self.epoch = self.epoch.successor();
        let retired = self.machine.reconfigure(config, self.epoch, self.now())?;
        for candidate in retired {
            self.store.release(&candidate)?;
        }
        restore(&mut self.machine, &self.store, config)?;
        self.poisoned = false;
        self.drive()
    }

    pub(crate) const fn completions(&self) -> &Receiver<Effect> {
        &self.completions
    }

    pub(crate) fn settle(&mut self, effect: Effect) -> Result<()> {
        match effect {
            Effect::Catalog { intent, result } => {
                let result = if intent.epoch() == self.epoch {
                    result.and_then(|harvest| {
                        self.store
                            .absorb(harvest)
                            .map_err(|error| format!("{error:#}"))
                    })
                } else {
                    result
                };
                self.machine
                    .settle_catalog(intent.complete(result), self.now())?;
            }
            Effect::Fetch(effect) => {
                self.settle_fetch(effect.intent, effect.result)?;
            }
        }
        self.drive()
    }

    pub fn drive(&mut self) -> Result<()> {
        if !self.active || self.poisoned {
            return Ok(());
        }
        if let Some(intent) = self.machine.plan_catalog(self.now()) {
            let lane = self
                .catalog_lane
                .as_ref()
                .context("remote catalog lane retired")?;
            if let Err(error) = lane.try_send(intent) {
                let intent = recover(error);
                self.machine.settle_catalog(
                    intent.complete(Err("remote catalog lane unavailable".to_owned())),
                    self.now(),
                )?;
            }
        }
        if let Some(intent) = self.machine.plan_fetch() {
            let item = intent.discovery().item_id.clone();
            if let Err(error) = self.store.begin_fetch(&item) {
                self.settle_fetch(intent, Err(format!("{error:#}")))?;
                return Ok(());
            }
            let lane = self
                .fetch_lane
                .as_ref()
                .context("remote fetch lane retired")?;
            if let Err(error) = lane.try_send(intent) {
                let intent = recover(error);
                self.settle_fetch(intent, Err("remote fetch lane unavailable".to_owned()))?;
            }
        }
        Ok(())
    }

    pub fn wait(&self) -> Duration {
        if !self.active || self.poisoned {
            return DORMANT_WAIT;
        }
        let now = self.now().milliseconds();
        self.machine.next_deadline().map_or(DORMANT_WAIT, |due| {
            Duration::from_millis(due.milliseconds().saturating_sub(now))
        })
    }

    pub fn offer(&mut self) -> Result<Option<Prepared>> {
        Ok(self.machine.offer()?.cloned())
    }

    pub fn reject_offer(&mut self) -> Result<()> {
        let candidate = self
            .machine
            .retire_offer()?
            .context("there is no remote offer to reject")?;
        self.store.reject(&candidate)?;
        self.drive()
    }

    pub fn veto_offer_stream(&mut self) -> Result<()> {
        let candidate = self
            .machine
            .offer()?
            .cloned()
            .context("there is no remote offer to veto")?;
        let _retired = self.machine.retire_stream(
            candidate.discovery.source_identity.as_str(),
            &candidate.discovery.stream_id,
        )?;
        self.store.veto_stream(&candidate)?;
        self.drive()
    }

    pub fn seal_promoted(&mut self, expected: &Prepared) -> Result<()> {
        let candidate = self
            .machine
            .retire_offer()?
            .context("there is no remote offer to promote")?;
        ensure!(
            candidate.discovery.item_id == expected.discovery.item_id,
            "remote offer changed during promotion"
        );
        self.store.promoted(&candidate)?;
        self.drive()
    }

    pub fn note_duel(&self, candidate: &Prepared, remote_won: bool) -> Result<()> {
        self.store
            .note_duel(&candidate.discovery.item_id, remote_won)
    }

    pub fn summary(&self) -> Summary {
        self.machine.summary()
    }

    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    fn settle_fetch(
        &mut self,
        intent: FetchIntent,
        result: std::result::Result<Prepared, String>,
    ) -> Result<()> {
        match self.machine.settle_fetch(intent.complete(result))? {
            FetchSettlement::Accepted(candidate) => self.store.prepared(&candidate),
            FetchSettlement::Discarded(candidate) => self.store.discard_fetch(&candidate),
            FetchSettlement::Failed { discovery, message } => {
                self.store.fetch_failed(&discovery.item_id, &message)
            }
        }
    }

    fn now(&self) -> Moment {
        Moment::from_millis(u64::try_from(self.born.elapsed().as_millis()).unwrap_or(u64::MAX))
    }
}

impl Drop for Reactor {
    fn drop(&mut self) {
        drop(self.catalog_lane.take());
        drop(self.fetch_lane.take());
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                eprintln!("Picmash remote effect lane panicked during retirement");
            }
        }
    }
}

fn recover<T>(error: TrySendError<T>) -> T {
    match error {
        TrySendError::Full(value) | TrySendError::Disconnected(value) => value,
    }
}

fn restore(machine: &mut Machine, store: &RemoteStore, config: &RemoteConfig) -> Result<()> {
    let (discovered, prepared) = store.restore(config)?;
    if !config.enabled {
        for candidate in prepared {
            store.release(&candidate)?;
        }
        return Ok(());
    }
    let discarded = machine.restore_prepared(prepared)?;
    let mut released = HashMap::new();
    for candidate in discarded {
        store.release(&candidate)?;
        released
            .entry(candidate.discovery.source)
            .or_insert_with(|| (candidate.discovery.source_identity.clone(), Vec::new()))
            .1
            .push(candidate.discovery);
    }
    machine.restore_discovered(discovered.into_iter().chain(released.into_iter().map(
        |(source, (source_identity, discoveries))| Harvest {
            source,
            source_identity,
            discoveries,
        },
    )))
}
