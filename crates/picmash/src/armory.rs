//! Prompt blades forged ahead of the pair that shows them.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex, PoisonError},
    thread::{self, JoinHandle},
};

use anyhow::{Context as _, Result};
use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::worker::{Blade, PROMPT_EDGE, decode_thumbnail};

/// Smiths forging at once: both blades of the next pair.
const SMITHS: usize = 2;

/// The stock a prompt blade is forged from: one image file under one display
/// rotation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Billet {
    pub path: PathBuf,
    pub rotation_quarters: u8,
}

impl Billet {
    fn forge(&self) -> Result<Blade> {
        decode_thumbnail(&self.path, self.rotation_quarters, PROMPT_EDGE)
    }
}

/// A smith's return: the blade, or nothing when forging failed or the order
/// was withdrawn before the smith reached it.
pub struct Forged {
    billet: Billet,
    blade: Option<Blade>,
}

/// Smiths forging the blades the next pair will show while the current pair
/// is judged. The armory keeps only what its last order wants; a blade drawn
/// before it is ready is awaited from its smith, or forged on the spot.
pub struct Armory {
    orders: Option<Sender<Billet>>,
    forged: Receiver<Forged>,
    wanted: Arc<Mutex<HashSet<Billet>>>,
    pending: HashSet<Billet>,
    ready: HashMap<Billet, Blade>,
    smiths: Vec<JoinHandle<()>>,
}

impl Armory {
    pub fn raise() -> Result<Self> {
        let (orders, inbox) = unbounded::<Billet>();
        let (outbox, forged) = unbounded();
        let wanted = Arc::new(Mutex::new(HashSet::new()));
        let smiths = (0..SMITHS)
            .map(|index| {
                let inbox = inbox.clone();
                let outbox = outbox.clone();
                let wanted = Arc::clone(&wanted);
                thread::Builder::new()
                    .name(format!("picmash-smith-{index}"))
                    .spawn(move || {
                        for billet in inbox {
                            let blade = wants(&wanted, &billet)
                                .then(|| billet.forge().ok())
                                .flatten();
                            if outbox.send(Forged { billet, blade }).is_err() {
                                break;
                            }
                        }
                    })
                    .context("spawn picmash smith")
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            orders: Some(orders),
            forged,
            wanted,
            pending: HashSet::new(),
            ready: HashMap::new(),
            smiths,
        })
    }

    pub const fn forged(&self) -> &Receiver<Forged> {
        &self.forged
    }

    /// Wants exactly `billets`: orders those neither ready nor ordered, and
    /// drops every other blade.
    pub fn order(&mut self, billets: impl IntoIterator<Item = Billet>) {
        let wanted: HashSet<Billet> = billets.into_iter().collect();
        self.ready.retain(|billet, _| wanted.contains(billet));
        let fresh: Vec<Billet> = wanted
            .iter()
            .filter(|billet| !self.ready.contains_key(*billet) && !self.pending.contains(*billet))
            .cloned()
            .collect();
        *self.wanted.lock().unwrap_or_else(PoisonError::into_inner) = wanted;
        let Some(orders) = &self.orders else {
            return;
        };
        for billet in fresh {
            if orders.send(billet.clone()).is_ok() {
                let _ordered = self.pending.insert(billet);
            }
        }
    }

    pub fn receive(&mut self, forged: Forged) {
        let _settled = self.pending.remove(&forged.billet);
        if let Some(blade) = forged.blade
            && wants(&self.wanted, &forged.billet)
        {
            let _stocked = self.ready.insert(forged.billet, blade);
        }
    }

    /// The blade for `billet`: forged ahead, awaited from its smith, or
    /// forged now.
    pub fn draw(&mut self, billet: &Billet) -> Result<Blade> {
        while self.pending.contains(billet) {
            let Ok(forged) = self.forged.recv() else {
                break;
            };
            self.receive(forged);
        }
        self.ready.remove(billet).map_or_else(|| billet.forge(), Ok)
    }
}

impl Drop for Armory {
    fn drop(&mut self) {
        self.wanted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        drop(self.orders.take());
        for smith in self.smiths.drain(..) {
            let _joined = smith.join();
        }
    }
}

fn wants(wanted: &Mutex<HashSet<Billet>>, billet: &Billet) -> bool {
    wanted
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains(billet)
}
