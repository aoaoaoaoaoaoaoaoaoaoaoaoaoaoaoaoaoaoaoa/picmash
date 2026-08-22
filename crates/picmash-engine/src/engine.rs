use std::{fs, path::Path, sync::Arc};

use parking_lot::Mutex;
use rusqlite::Connection;
use time::OffsetDateTime;

use crate::{
    fault::{Fault, IoResultExt, Result},
    schema,
};

#[derive(Clone)]
pub struct Engine {
    pub(crate) connection: Arc<Mutex<Connection>>,
}

impl Engine {
    pub fn open(database: impl AsRef<Path>) -> Result<Self> {
        let database = database.as_ref();
        if let Some(parent) = database
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).at(parent)?;
        }
        let mut connection = Connection::open(database)?;
        schema::configure(&connection)?;
        schema::migrate(&mut connection, now_ns()?)?;
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
        })
    }
}

pub fn now_ns() -> Result<i64> {
    i64::try_from(OffsetDateTime::now_utc().unix_timestamp_nanos())
        .map_err(|_| Fault::Corrupt("system time lies outside the database range".to_owned()))
}
