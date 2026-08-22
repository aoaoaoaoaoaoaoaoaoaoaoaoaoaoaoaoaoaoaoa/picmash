//! Demand-bounded remote acquisition as a closed transition system.
//!
//! Let `R` be configured reservoir capacity. Only `Fetching`, `Prepared`, and
//! `Offered` own media. Every transition preserves
//! `|Fetching| + |Prepared| + |Offered| ≤ R`. Catalog metadata is separately
//! bounded by `M` and partitioned among positive-weight sources. Effect
//! creation consumes one unique permit; completion consumes that permit before
//! inspecting its generation, so stale work cannot increase physical
//! concurrency or enter the current frontier.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
};

use anyhow::{Result, ensure};

use super::{
    Discovery, Epoch, Harvest, Prepared, RemoteItemId, SourceIx, StreamId, Summary,
    promotion::ARCHIVE_CAPACITY,
};
use crate::configuration::{RemoteConfig, SourceConfig};

const CATALOG_CONCURRENCY: usize = 1;
const FETCH_CONCURRENCY: usize = 1;
const MAX_SOURCES: usize = 32;
const SERVICE_REBASE: u64 = 1 << 60;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Moment(u64);

impl Moment {
    pub const ZERO: Self = Self(0);

    pub const fn from_millis(milliseconds: u64) -> Self {
        Self(milliseconds)
    }

    const fn after_seconds(self, seconds: u64) -> Self {
        Self(self.0.saturating_add(seconds.saturating_mul(1_000)))
    }

    pub const fn milliseconds(self) -> u64 {
        self.0
    }
}

#[derive(Debug)]
struct CatalogKind;

#[derive(Debug)]
struct FetchKind;

#[derive(Debug)]
struct Permit<K> {
    slot: u8,
    epoch: Epoch,
    _kind: PhantomData<K>,
}

#[derive(Debug)]
struct PermitPool<K, const N: usize> {
    occupied: [bool; N],
    _kind: PhantomData<K>,
}

impl<K, const N: usize> PermitPool<K, N> {
    const fn new() -> Self {
        Self {
            occupied: [false; N],
            _kind: PhantomData,
        }
    }

    fn acquire(&mut self, epoch: Epoch) -> Option<Permit<K>> {
        let slot = self.occupied.iter().position(|occupied| !occupied)?;
        self.occupied[slot] = true;
        Some(Permit {
            slot: slot as u8,
            epoch,
            _kind: PhantomData,
        })
    }

    fn release(&mut self, permit: Permit<K>) -> Epoch {
        let slot = usize::from(permit.slot);
        assert!(
            slot < N && self.occupied[slot],
            "effect returned a forged permit"
        );
        self.occupied[slot] = false;
        permit.epoch
    }

    fn occupied(&self) -> usize {
        self.occupied.iter().filter(|occupied| **occupied).count()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FairQueue {
    services: [u64; MAX_SOURCES],
}

impl FairQueue {
    fn choose(
        &mut self,
        sources: &[SourceState],
        eligible: impl Fn(SourceIx) -> bool,
    ) -> Option<SourceIx> {
        let chosen = sources
            .iter()
            .enumerate()
            .filter(|(index, source)| source.weight > 0 && eligible(SourceIx::new(*index)))
            .min_by(|(left_index, left), (right_index, right)| {
                let left_finish = u128::from(self.services[*left_index]) * u128::from(right.weight);
                let right_finish =
                    u128::from(self.services[*right_index]) * u128::from(left.weight);
                left_finish
                    .cmp(&right_finish)
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| SourceIx::new(index))?;
        self.services[chosen.get()] += 1;
        if self.services[chosen.get()] >= SERVICE_REBASE {
            let rounds = sources
                .iter()
                .enumerate()
                .filter(|(index, source)| source.weight > 0 && eligible(SourceIx::new(*index)))
                .map(|(index, source)| self.services[index] / u64::from(source.weight))
                .min()
                .unwrap_or(0);
            if rounds > 0 {
                for (index, source) in sources
                    .iter()
                    .enumerate()
                    .filter(|(index, source)| source.weight > 0 && eligible(SourceIx::new(*index)))
                {
                    self.services[index] -= rounds * u64::from(source.weight);
                }
            }
        }
        Some(chosen)
    }
}

#[derive(Clone, Debug)]
enum CatalogPhase {
    Due,
    Cataloging { prior_strikes: u8 },
    Waiting(Moment),
    Backoff { until: Moment, strikes: u8 },
}

impl CatalogPhase {
    fn due(&self, now: Moment) -> bool {
        match self {
            Self::Due => true,
            Self::Waiting(due) => *due <= now,
            Self::Backoff { until, .. } => *until <= now,
            Self::Cataloging { .. } => false,
        }
    }

    fn strikes(&self) -> u8 {
        match self {
            Self::Backoff { strikes, .. } => *strikes,
            Self::Cataloging { prior_strikes } => *prior_strikes,
            Self::Due | Self::Waiting(_) => 0,
        }
    }
}

#[derive(Clone, Debug)]
struct SourceState {
    config: SourceConfig,
    weight: u32,
    phase: CatalogPhase,
    catalog_ordinal: u64,
}

#[derive(Debug)]
pub struct CatalogIntent {
    permit: Permit<CatalogKind>,
    source: SourceIx,
    config: SourceConfig,
    ordinal: u64,
}

impl CatalogIntent {
    pub const fn epoch(&self) -> Epoch {
        self.permit.epoch
    }

    pub const fn source(&self) -> SourceIx {
        self.source
    }

    pub const fn config(&self) -> &SourceConfig {
        &self.config
    }

    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    pub fn complete(self, result: std::result::Result<Harvest, String>) -> CatalogCompletion {
        CatalogCompletion {
            permit: self.permit,
            source: self.source,
            result,
        }
    }
}

#[derive(Debug)]
pub struct FetchIntent {
    permit: Permit<FetchKind>,
    discovery: Discovery,
}

impl FetchIntent {
    pub const fn discovery(&self) -> &Discovery {
        &self.discovery
    }

    pub fn complete(self, result: std::result::Result<Prepared, String>) -> FetchCompletion {
        FetchCompletion {
            permit: self.permit,
            discovery: self.discovery,
            result,
        }
    }
}

#[derive(Debug)]
pub struct CatalogCompletion {
    permit: Permit<CatalogKind>,
    source: SourceIx,
    result: std::result::Result<Harvest, String>,
}

#[derive(Debug)]
pub struct FetchCompletion {
    permit: Permit<FetchKind>,
    discovery: Discovery,
    result: std::result::Result<Prepared, String>,
}

#[derive(Debug)]
pub enum FetchSettlement {
    Accepted(Prepared),
    Discarded(Prepared),
    Failed {
        discovery: Discovery,
        message: String,
    },
}

#[derive(Debug)]
pub struct Machine {
    epoch: Epoch,
    enabled: bool,
    reservoir_capacity: usize,
    metadata_capacity: usize,
    sources: Vec<SourceState>,
    catalog_fairness: FairQueue,
    fetch_fairness: FairQueue,
    offer_fairness: FairQueue,
    catalog_permits: PermitPool<CatalogKind, CATALOG_CONCURRENCY>,
    fetch_permits: PermitPool<FetchKind, FETCH_CONCURRENCY>,
    discovered: VecDeque<Discovery>,
    fetching: HashMap<RemoteItemId, Discovery>,
    prepared: VecDeque<Prepared>,
    offered: Option<Prepared>,
    promoting: HashMap<RemoteItemId, Prepared>,
    blocked_streams: HashSet<(String, StreamId)>,
}

impl Machine {
    pub fn new(config: &RemoteConfig, epoch: Epoch, now: Moment) -> Result<Self> {
        let mut machine = Self {
            epoch,
            enabled: false,
            reservoir_capacity: 1,
            metadata_capacity: 8,
            sources: Vec::new(),
            catalog_fairness: FairQueue::default(),
            fetch_fairness: FairQueue::default(),
            offer_fairness: FairQueue::default(),
            catalog_permits: PermitPool::new(),
            fetch_permits: PermitPool::new(),
            discovered: VecDeque::new(),
            fetching: HashMap::new(),
            prepared: VecDeque::new(),
            offered: None,
            promoting: HashMap::new(),
            blocked_streams: HashSet::new(),
        };
        machine.reconfigure(config, epoch, now)?;
        Ok(machine)
    }

    pub fn reconfigure(
        &mut self,
        config: &RemoteConfig,
        epoch: Epoch,
        _now: Moment,
    ) -> Result<Vec<Prepared>> {
        ensure!(
            config.sources.len() <= MAX_SOURCES,
            "at most {MAX_SOURCES} remote sources are admitted"
        );
        self.epoch = epoch;
        self.enabled = config.enabled;
        self.reservoir_capacity = usize::from(config.reservoir_capacity.get());
        self.metadata_capacity = usize::from(config.metadata_capacity.get());
        self.sources = config
            .sources
            .iter()
            .cloned()
            .map(|config| SourceState {
                weight: config.weight.quantum(),
                config,
                phase: CatalogPhase::Due,
                catalog_ordinal: 0,
            })
            .collect();
        self.catalog_fairness = FairQueue::default();
        self.fetch_fairness = FairQueue::default();
        self.offer_fairness = FairQueue::default();
        self.discovered.clear();
        self.blocked_streams.clear();
        let mut retired = self.prepared.drain(..).collect::<Vec<_>>();
        retired.extend(self.offered.take());
        self.audit()?;
        Ok(retired)
    }

    pub fn plan_catalog(&mut self, now: Moment) -> Option<CatalogIntent> {
        if !self.enabled || self.discovered.len() >= self.metadata_capacity {
            return None;
        }
        let permit = self.catalog_permits.acquire(self.epoch)?;
        let source = self.catalog_fairness.choose(&self.sources, |source| {
            self.sources[source.get()].phase.due(now)
        });
        let Some(source) = source else {
            let _epoch = self.catalog_permits.release(permit);
            return None;
        };
        let state = &mut self.sources[source.get()];
        let prior_strikes = state.phase.strikes();
        state.phase = CatalogPhase::Cataloging { prior_strikes };
        let ordinal = state.catalog_ordinal;
        state.catalog_ordinal = state.catalog_ordinal.saturating_add(1);
        Some(CatalogIntent {
            permit,
            source,
            config: self.sources[source.get()].config.clone(),
            ordinal,
        })
    }

    pub fn settle_catalog(&mut self, completion: CatalogCompletion, now: Moment) -> Result<()> {
        let epoch = self.catalog_permits.release(completion.permit);
        if epoch != self.epoch || completion.source.get() >= self.sources.len() {
            return self.audit();
        }
        let state = &mut self.sources[completion.source.get()];
        if let Ok(harvest) = completion.result {
            ensure!(
                harvest.source == completion.source,
                "catalog returned another source's harvest"
            );
            ensure!(
                harvest.source_identity == state.config.identity(),
                "catalog returned another source's identity"
            );
            state.phase = CatalogPhase::Waiting(
                now.after_seconds(u64::from(state.config.scan_interval_seconds.get())),
            );
            self.absorb(harvest);
        } else {
            let strikes = state.phase.strikes().saturating_add(1);
            let delay = 5_u64.saturating_mul(1_u64 << strikes.min(6));
            state.phase = CatalogPhase::Backoff {
                until: now.after_seconds(delay.min(300)),
                strikes,
            };
        }
        self.audit()
    }

    pub fn plan_fetch(&mut self) -> Option<FetchIntent> {
        if !self.enabled || self.occupancy() >= self.reservoir_capacity {
            return None;
        }
        let permit = self.fetch_permits.acquire(self.epoch)?;
        let source = self.fetch_fairness.choose(&self.sources, |source| {
            self.discovered.iter().any(|item| item.source == source)
        });
        let Some(source) = source else {
            let _epoch = self.fetch_permits.release(permit);
            return None;
        };
        let Some(slot) = self
            .discovered
            .iter()
            .position(|item| item.source == source)
        else {
            let _epoch = self.fetch_permits.release(permit);
            return None;
        };
        let Some(discovery) = self.discovered.remove(slot) else {
            let _epoch = self.fetch_permits.release(permit);
            return None;
        };
        let _old = self
            .fetching
            .insert(discovery.item_id.clone(), discovery.clone());
        Some(FetchIntent { permit, discovery })
    }

    pub fn settle_fetch(&mut self, completion: FetchCompletion) -> Result<FetchSettlement> {
        let epoch = self.fetch_permits.release(completion.permit);
        let stale = epoch != self.epoch;
        let _fetching = self.fetching.remove(&completion.discovery.item_id);
        let settlement = match completion.result {
            Ok(prepared) => {
                ensure!(
                    prepared.discovery.item_id == completion.discovery.item_id,
                    "fetch returned another candidate's payload"
                );
                let blocked = self.blocked_streams.contains(&stream_key(&prepared));
                if !stale && !blocked && self.occupancy() < self.reservoir_capacity {
                    self.prepared.push_back(prepared.clone());
                    FetchSettlement::Accepted(prepared)
                } else {
                    FetchSettlement::Discarded(prepared)
                }
            }
            Err(message) => FetchSettlement::Failed {
                discovery: completion.discovery,
                message,
            },
        };
        self.audit()?;
        Ok(settlement)
    }

    pub fn offer(&mut self) -> Result<Option<&Prepared>> {
        if self.offered.is_none() && self.promoting.len() == ARCHIVE_CAPACITY {
            return Ok(None);
        }
        if self.offered.is_none() {
            let source = self.offer_fairness.choose(&self.sources, |source| {
                self.prepared
                    .iter()
                    .any(|candidate| candidate.discovery.source == source)
            });
            if let Some(source) = source {
                let Some(slot) = self
                    .prepared
                    .iter()
                    .position(|candidate| candidate.discovery.source == source)
                else {
                    return Err(anyhow::anyhow!(
                        "fairness selected a source with no prepared candidate"
                    ));
                };
                self.offered = self.prepared.remove(slot);
            }
        }
        self.audit()?;
        Ok(self.offered.as_ref())
    }

    pub fn retire_offer(&mut self) -> Result<Option<Prepared>> {
        let retired = self.offered.take();
        self.audit()?;
        Ok(retired)
    }

    pub fn begin_promotion(&mut self) -> Result<Prepared> {
        ensure!(
            self.promoting.len() < ARCHIVE_CAPACITY,
            "remote archive queue is full"
        );
        let candidate = self
            .offered
            .take()
            .ok_or_else(|| anyhow::anyhow!("there is no remote offer to promote"))?;
        ensure!(
            !self.promoting.contains_key(&candidate.discovery.item_id),
            "remote candidate is already being promoted"
        );
        let _prior = self
            .promoting
            .insert(candidate.discovery.item_id.clone(), candidate.clone());
        self.audit()?;
        Ok(candidate)
    }

    pub fn restore_promoting(
        &mut self,
        candidates: impl IntoIterator<Item = Prepared>,
    ) -> Result<()> {
        for candidate in candidates {
            if self.promoting.contains_key(&candidate.discovery.item_id) {
                continue;
            }
            ensure!(
                self.promoting.len() < ARCHIVE_CAPACITY,
                "persisted promotions exceed the remote archive bound"
            );
            let _prior = self
                .promoting
                .insert(candidate.discovery.item_id.clone(), candidate);
        }
        self.audit()
    }

    pub fn finish_promotion(&mut self, item: &RemoteItemId) -> Result<Prepared> {
        let candidate = self
            .promoting
            .remove(item)
            .ok_or_else(|| anyhow::anyhow!("remote promotion {item} is not pending"))?;
        self.audit()?;
        Ok(candidate)
    }

    pub fn promotion(&self, item: &RemoteItemId) -> Result<&Prepared> {
        self.promoting
            .get(item)
            .ok_or_else(|| anyhow::anyhow!("remote promotion {item} is not pending"))
    }

    pub fn abort_promotion(&mut self, item: &RemoteItemId) -> Result<()> {
        let candidate = self
            .promoting
            .remove(item)
            .ok_or_else(|| anyhow::anyhow!("remote promotion {item} is not pending"))?;
        ensure!(
            self.offered.is_none(),
            "remote offer was replaced before promotion submission failed"
        );
        self.offered = Some(candidate);
        self.audit()
    }

    pub fn restore_discovered(
        &mut self,
        harvests: impl IntoIterator<Item = Harvest>,
    ) -> Result<()> {
        for harvest in harvests {
            ensure!(
                harvest.source.get() < self.sources.len(),
                "persisted remote source left configuration"
            );
            self.absorb(harvest);
        }
        self.audit()
    }

    pub fn restore_prepared(
        &mut self,
        candidates: impl IntoIterator<Item = Prepared>,
    ) -> Result<Vec<Prepared>> {
        let mut discarded = Vec::new();
        for candidate in candidates {
            let admissible_source = candidate.discovery.source.get() < self.sources.len();
            let blocked = self.blocked_streams.contains(&stream_key(&candidate));
            let duplicate = self
                .prepared
                .iter()
                .any(|present| present.discovery.item_id == candidate.discovery.item_id);
            if self.occupancy() < self.reservoir_capacity
                && admissible_source
                && !blocked
                && !duplicate
            {
                self.prepared.push_back(candidate);
            } else {
                discarded.push(candidate);
            }
        }
        self.audit()?;
        Ok(discarded)
    }

    pub fn retire_stream(
        &mut self,
        source_identity: &str,
        stream: &StreamId,
    ) -> Result<Vec<Prepared>> {
        let key = (source_identity.to_owned(), stream.clone());
        let _new = self.blocked_streams.insert(key.clone());
        self.discovered.retain(|candidate| {
            (candidate.source_identity.as_str(), &candidate.stream_id) != (source_identity, stream)
        });
        let mut retired = Vec::new();
        self.prepared.retain(|candidate| {
            let survives = stream_key(candidate) != key;
            if !survives {
                retired.push(candidate.clone());
            }
            survives
        });
        if self
            .offered
            .as_ref()
            .is_some_and(|candidate| stream_key(candidate) == key)
        {
            retired.extend(self.offered.take());
        }
        self.audit()?;
        Ok(retired)
    }

    pub fn summary(&self) -> Summary {
        let enabled_sources = if self.enabled {
            self.sources
                .iter()
                .filter(|source| source.weight > 0)
                .count()
        } else {
            0
        };
        Summary {
            enabled_sources,
            cataloging: self.catalog_permits.occupied(),
            backing_off: self
                .sources
                .iter()
                .filter(|source| matches!(source.phase, CatalogPhase::Backoff { .. }))
                .count(),
            discovered: self.discovered.len(),
            fetching: self.fetching.len(),
            prepared: self.prepared.len(),
            offered: self.offered.is_some(),
            promoting: self.promoting.len(),
        }
    }

    pub fn next_deadline(&self) -> Option<Moment> {
        if !self.enabled
            || self.discovered.len() >= self.metadata_capacity
            || self.catalog_permits.occupied() == CATALOG_CONCURRENCY
        {
            return None;
        }
        self.sources
            .iter()
            .filter(|source| source.weight > 0)
            .filter_map(|source| match source.phase {
                CatalogPhase::Due => Some(Moment::ZERO),
                CatalogPhase::Waiting(due) => Some(due),
                CatalogPhase::Backoff { until, .. } => Some(until),
                CatalogPhase::Cataloging { .. } => None,
            })
            .min()
    }

    fn absorb(&mut self, harvest: Harvest) {
        let source_identity = harvest.source_identity;
        let enabled = self
            .sources
            .iter()
            .filter(|source| source.weight > 0)
            .count()
            .max(1);
        let quota = (self.metadata_capacity / enabled).max(1);
        self.discovered
            .retain(|candidate| candidate.source != harvest.source);
        let known = self
            .fetching
            .keys()
            .chain(
                self.prepared
                    .iter()
                    .map(|prepared| &prepared.discovery.item_id),
            )
            .chain(
                self.offered
                    .iter()
                    .map(|prepared| &prepared.discovery.item_id),
            )
            .chain(self.promoting.keys())
            .cloned()
            .collect::<HashSet<_>>();
        self.discovered.extend(
            harvest
                .discoveries
                .into_iter()
                .filter(|candidate| candidate.source == harvest.source)
                .filter(|candidate| candidate.source_identity == source_identity)
                .filter(|candidate| {
                    !self.blocked_streams.contains(&(
                        candidate.source_identity.as_str().to_owned(),
                        candidate.stream_id.clone(),
                    ))
                })
                .filter(|candidate| !known.contains(&candidate.item_id))
                .take(quota),
        );
        self.discovered.truncate(self.metadata_capacity);
    }

    fn occupancy(&self) -> usize {
        self.fetching.len() + self.prepared.len() + usize::from(self.offered.is_some())
    }

    fn audit(&self) -> Result<()> {
        ensure!(
            self.occupancy() <= self.reservoir_capacity,
            "remote reservoir overflow"
        );
        ensure!(
            self.discovered.len() <= self.metadata_capacity,
            "remote metadata overflow"
        );
        ensure!(
            self.catalog_permits.occupied() <= CATALOG_CONCURRENCY,
            "catalog permit overflow"
        );
        ensure!(
            self.fetch_permits.occupied() <= FETCH_CONCURRENCY,
            "fetch permit overflow"
        );
        ensure!(
            self.promoting.len() <= ARCHIVE_CAPACITY,
            "remote archive queue overflow"
        );
        ensure!(self.sources.len() <= MAX_SOURCES, "remote source overflow");
        Ok(())
    }
}

fn stream_key(candidate: &Prepared) -> (String, StreamId) {
    (
        candidate.discovery.source_identity.as_str().to_owned(),
        candidate.discovery.stream_id.clone(),
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use proptest::prelude::*;

    use super::*;
    use crate::{
        configuration::{
            ImageFilter, ImportPolicy, LocalDirectory, MetadataCapacity, Probability,
            ReservoirCapacity, ScanInterval, SourceIdentity, Upstream, Weight,
        },
        remote::{Origin, StreamId},
    };

    fn config(weights: &[u8], capacity: u8) -> Result<RemoteConfig> {
        Ok(RemoteConfig {
            enabled: true,
            sample_probability: Probability::try_from(1.0)?,
            reservoir_capacity: ReservoirCapacity::try_from(capacity)?,
            metadata_capacity: MetadataCapacity::try_from(64)?,
            sources: weights
                .iter()
                .enumerate()
                .map(|(index, weight)| {
                    Ok(SourceConfig {
                        weight: Weight::try_from(f64::from(*weight))?,
                        import_policy: ImportPolicy::NotX,
                        scan_interval_seconds: ScanInterval::try_from(15)?,
                        upstream: Upstream::LocalDirectory(LocalDirectory {
                            root: PathBuf::from(format!("/source/{index}")),
                            recurse: true,
                            filters: ImageFilter::default(),
                        }),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn discovery(
        source: SourceIx,
        source_identity: &SourceIdentity,
        serial: u64,
    ) -> Result<Discovery> {
        Ok(Discovery {
            source,
            source_identity: source_identity.clone(),
            source_name: format!("source {}", source.get()),
            import_policy: ImportPolicy::NotX,
            stream_id: StreamId::new(format!("stream-{serial}")),
            stream_title: format!("stream {serial}"),
            item_id: RemoteItemId::new(format!("item-{}-{serial}", source.get())),
            title: format!("item {serial}"),
            origin: Origin::Local {
                path: PathBuf::from(format!("/item/{serial}")),
                stamp: crate::remote::FileStamp::parse(vec![0; 56])?,
            },
            extension: "png".to_owned(),
            width: 1_000,
            height: 1_000,
            byte_len: 4_000,
            max_pixels: 40_000_000,
        })
    }

    proptest! {
        // The original remote pipeline repeatedly violated this law under
        // unlucky completion order; arbitrary interleavings are the durable oracle.
        #[test]
        fn every_interleaving_preserves_effect_and_reservoir_bounds(
            actions in prop::collection::vec(0_u8..=10, 1..300),
            capacity in 1_u8..=8,
        ) {
            let result = (|| -> Result<()> {
                let mut machine = Machine::new(&config(&[1, 2, 3], capacity)?, Epoch::INITIAL, Moment::ZERO)?;
                let mut catalogs = VecDeque::new();
                let mut fetches = VecDeque::new();
                let mut promotions = VecDeque::new();
                let mut serial = 0_u64;
                for action in actions {
                    match action {
                        0 => if let Some(intent) = machine.plan_catalog(Moment::from_millis(serial * 20_000)) {
                            catalogs.push_back(intent);
                        },
                        1 => if let Some(intent) = catalogs.pop_front() {
                            let source = intent.source();
                            let source_identity = intent.config().identity();
                            let harvest = Harvest {
                                source,
                                source_identity: source_identity.clone(),
                                discoveries: (0..32).map(|_| {
                                    serial += 1;
                                    discovery(source, &source_identity, serial)
                                }).collect::<Result<Vec<_>>>()?,
                            };
                            machine.settle_catalog(intent.complete(Ok(harvest)), Moment::from_millis(serial * 20_000))?;
                        },
                        2 => if let Some(intent) = catalogs.pop_front() {
                            machine.settle_catalog(intent.complete(Err("fault".to_owned())), Moment::from_millis(serial * 20_000))?;
                        },
                        3 => if let Some(intent) = machine.plan_fetch() {
                            fetches.push_back(intent);
                        },
                        4 => if let Some(intent) = fetches.pop_front() {
                            let candidate = intent.discovery().clone();
                            let prepared = Prepared {
                                cache_path: PathBuf::from(format!("/cache/{}", candidate.item_id)),
                                payload_digest: format!("digest-{serial}"),
                                discovery: candidate,
                            };
                            let _settlement = machine.settle_fetch(intent.complete(Ok(prepared)))?;
                        },
                        5 => if let Some(intent) = fetches.pop_front() {
                            let _settlement = machine.settle_fetch(intent.complete(Err("fault".to_owned())))?;
                        },
                        6 => { let _offered = machine.offer()?; },
                        7 => { let _retired = machine.retire_offer()?; },
                        8 => if let Ok(candidate) = machine.begin_promotion() {
                            promotions.push_back(candidate.discovery.item_id);
                        },
                        9 => if let Some(item) = promotions.pop_front() {
                            let _candidate = machine.finish_promotion(&item)?;
                        },
                        _ => {
                            let next = machine.epoch.successor();
                            let _retired = machine.reconfigure(&config(&[3, 1], capacity)?, next, Moment::ZERO)?;
                        }
                    }
                    ensure!(
                        machine.catalog_permits.occupied() == catalogs.len(),
                        "catalog effect escaped its permit"
                    );
                    ensure!(
                        machine.fetching.len() == fetches.len()
                            && machine.fetch_permits.occupied() == fetches.len(),
                        "physical fetch escaped the reservoir"
                    );
                    ensure!(
                        machine.promoting.len() == promotions.len(),
                        "archive effect escaped its bound"
                    );
                    machine.audit()?;
                }
                Ok(())
            })();
            prop_assert!(result.is_ok(), "{result:#?}");
        }
    }

    #[test]
    fn weighted_fair_queue_cannot_starve_an_eligible_source() -> Result<()> {
        let config = config(&[1, 3, 9], 4)?;
        let sources = config
            .sources
            .into_iter()
            .map(|config| SourceState {
                weight: config.weight.quantum(),
                config,
                phase: CatalogPhase::Due,
                catalog_ordinal: 0,
            })
            .collect::<Vec<_>>();
        let mut queue = FairQueue::default();
        let mut services = [0_u64; 3];
        for _ in 0..1_300 {
            let chosen = queue
                .choose(&sources, |_| true)
                .ok_or_else(|| anyhow::anyhow!("no eligible source"))?;
            services[chosen.get()] += 1;
        }
        assert!(services.into_iter().all(|count| count > 0));
        assert!(services[0] < services[1] && services[1] < services[2]);
        Ok(())
    }
}
