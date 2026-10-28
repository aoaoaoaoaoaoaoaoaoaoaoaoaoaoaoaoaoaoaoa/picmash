//! Where admitted assets came from: the sources that yield images, their
//! streams, and the items admitted from them.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use rusqlite::{Connection, OptionalExtension as _, Transaction, params};

use crate::{
    Engine,
    catalog::{decode_path, encode_path},
    fault::{Fault, Result},
    ids::{AssetId, CollectionId, ItemId, SourceId, StreamId},
};

/// An upstream that yields images.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Source {
    /// A 4chan board, by short name.
    FourChanBoard(String),
    /// A local directory tree, by root.
    LocalDirectory(PathBuf),
}

impl Source {
    #[must_use]
    pub fn id(&self) -> SourceId {
        match self {
            Self::FourChanBoard(board) => SourceId::four_chan_board(board),
            Self::LocalDirectory(root) => SourceId::local_directory(root),
        }
    }

    /// Inverse of [`Self::parts`].
    pub fn from_parts(kind: &str, locator: Vec<u8>) -> Result<Self> {
        match kind {
            "4chan_board" => String::from_utf8(locator)
                .map(Self::FourChanBoard)
                .map_err(|_| Fault::Corrupt("persisted 4chan board is not UTF-8".to_owned())),
            "local_directory" => Ok(Self::LocalDirectory(decode_path(locator))),
            other => Err(Fault::Corrupt(format!("unknown source kind `{other}`"))),
        }
    }

    /// The `(kind, locator)` spelling every store persists a source under.
    #[must_use]
    pub fn parts(&self) -> (&'static str, Vec<u8>) {
        match self {
            Self::FourChanBoard(board) => ("4chan_board", board.as_bytes().to_vec()),
            Self::LocalDirectory(root) => ("local_directory", encode_path(root)),
        }
    }
}

/// A subdivision of a source: a 4chan thread, or a folder of a directory tree.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Stream {
    pub id: StreamId,
    pub title: String,
}

impl Stream {
    /// Folder `folder` of the tree at `root`. Files directly beneath the root
    /// form the root's own stream, titled by the root's name.
    #[must_use]
    pub fn folder(root: &Path, folder: &Path) -> Self {
        let title = if folder.as_os_str().is_empty() {
            root.file_name().map_or_else(
                || root.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            )
        } else {
            folder.display().to_string()
        };
        Self {
            id: StreamId::folder(folder),
            title,
        }
    }
}

/// Where an admitted item's bytes were read.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Origin {
    /// An image URL, with the MD5 its upstream published for those bytes.
    Url {
        url: String,
        md5: Option<String>,
    },
    File(PathBuf),
}

/// The item one asset was admitted from, and where that item lives.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Provenance {
    pub source: Source,
    pub stream: Stream,
    pub item: ItemId,
    pub origin: Origin,
}

impl Engine {
    /// Attests provenance for the asset at `relative` beneath any collection
    /// root: the seam for admissions that predate provenance. Returns whether
    /// some collection holds that occurrence.
    pub fn attest(
        &self,
        relative: &Path,
        provenance: &Provenance,
        admitted_at_ns: i64,
    ) -> Result<bool> {
        let mut connection = self.connection.lock();
        let tx = connection.transaction()?;
        let roots = {
            let mut statement = tx.prepare("SELECT id, root FROM pm_collections")?;
            statement
                .query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        let mut attested = false;
        for (collection, root) in roots {
            let path = decode_path(root).join(relative);
            let asset = tx
                .query_row(
                    "SELECT asset_id FROM pm_occurrences WHERE collection_id = ?1 AND path = ?2",
                    params![collection, encode_path(&path)],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            if let Some(asset) = asset {
                inscribe(&tx, &AssetId::parse(asset)?, provenance, admitted_at_ns)?;
                attested = true;
            }
        }
        tx.commit()?;
        Ok(attested)
    }

    /// Retitles a stream that provenance already names. As when provenance
    /// is inscribed, the newest title wins.
    pub fn retitle(&self, source: &SourceId, stream: &Stream) -> Result<()> {
        self.connection.lock().execute(
            "UPDATE pm_streams SET title = ?3 WHERE source_id = ?1 AND id = ?2",
            params![source.as_str(), stream.id.as_str(), stream.title],
        )?;
        Ok(())
    }
}

/// Records that `asset` was admitted from `provenance`'s item. An item is
/// admitted once; a stream takes its newest title.
pub fn inscribe(
    tx: &Transaction<'_>,
    asset: &AssetId,
    provenance: &Provenance,
    admitted_at_ns: i64,
) -> Result<()> {
    let Provenance {
        source,
        stream,
        item,
        origin,
    } = provenance;
    let source_id = source.id();
    let (kind, locator) = source.parts();
    tx.execute(
        "INSERT INTO pm_sources(id, kind, locator) VALUES (?1, ?2, ?3)
         ON CONFLICT(id) DO NOTHING",
        params![source_id.as_str(), kind, locator],
    )?;
    tx.execute(
        "INSERT INTO pm_streams(source_id, id, title) VALUES (?1, ?2, ?3)
         ON CONFLICT(source_id, id) DO UPDATE SET title = excluded.title",
        params![source_id.as_str(), stream.id.as_str(), stream.title],
    )?;
    let (origin_kind, origin, md5) = match origin {
        Origin::Url { url, md5 } => ("url", url.as_bytes().to_vec(), md5.as_deref()),
        Origin::File(path) => ("file", encode_path(path), None),
    };
    tx.execute(
        "INSERT INTO pm_provenance(
             item_id, asset_id, source_id, stream_id, origin_kind, origin, md5, admitted_at_ns
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(item_id) DO NOTHING",
        params![
            item.as_str(),
            asset.as_str(),
            source_id.as_str(),
            stream.id.as_str(),
            origin_kind,
            origin,
            md5,
            admitted_at_ns,
        ],
    )?;
    Ok(())
}

/// Provenance of each visible asset of `collection`, oldest admission first.
pub fn by_asset(
    connection: &Connection,
    collection: CollectionId,
) -> Result<HashMap<AssetId, Vec<Provenance>>> {
    let mut statement = connection.prepare(
        "SELECT p.asset_id, s.kind, s.locator, p.stream_id, st.title, p.item_id,
                p.origin_kind, p.origin, p.md5
         FROM pm_provenance p
         JOIN pm_collection_assets ca ON ca.asset_id = p.asset_id
         JOIN pm_sources s ON s.id = p.source_id
         JOIN pm_streams st ON st.source_id = p.source_id AND st.id = p.stream_id
         WHERE ca.collection_id = ?1 AND ca.hidden = 0
         ORDER BY p.admitted_at_ns, p.item_id",
    )?;
    let rows = statement.query_map([collection.get()], |row| {
        Ok(ProvenanceRow {
            asset: row.get(0)?,
            kind: row.get(1)?,
            locator: row.get(2)?,
            stream: row.get(3)?,
            title: row.get(4)?,
            item: row.get(5)?,
            origin_kind: row.get(6)?,
            origin: row.get(7)?,
            md5: row.get(8)?,
        })
    })?;
    let mut ledger = HashMap::<AssetId, Vec<Provenance>>::new();
    for row in rows {
        let (asset, provenance) = row?.finish()?;
        ledger.entry(asset).or_default().push(provenance);
    }
    Ok(ledger)
}

struct ProvenanceRow {
    asset: String,
    kind: String,
    locator: Vec<u8>,
    stream: String,
    title: String,
    item: String,
    origin_kind: String,
    origin: Vec<u8>,
    md5: Option<String>,
}

impl ProvenanceRow {
    fn finish(self) -> Result<(AssetId, Provenance)> {
        let origin = match self.origin_kind.as_str() {
            "url" => Origin::Url {
                url: String::from_utf8(self.origin)
                    .map_err(|_| Fault::Corrupt("persisted origin URL is not UTF-8".to_owned()))?,
                md5: self.md5,
            },
            "file" => Origin::File(decode_path(self.origin)),
            other => return Err(Fault::Corrupt(format!("unknown origin kind `{other}`"))),
        };
        Ok((
            AssetId::parse(self.asset)?,
            Provenance {
                source: Source::from_parts(&self.kind, self.locator)?,
                stream: Stream {
                    id: StreamId::parse(self.stream)?,
                    title: self.title,
                },
                item: ItemId::parse(self.item)?,
                origin,
            },
        ))
    }
}

/// The text of 4chan markup, in which subjects, comments and file names
/// arrive.
///
/// Tags are dropped, line breaks spaced, entities decoded and whitespace
/// collapsed. Entities decode after tags are dropped, so an escaped
/// `&lt;b&gt;` survives as text, and only `<br>` separates words: 4chan
/// splits long words with `<wbr>`.
pub fn plain_text(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut rest = html;
    while let Some(start) = rest.find(['<', '&']) {
        text.push_str(&rest[..start]);
        rest = &rest[start..];
        if let Some(tag) = rest.strip_prefix('<') {
            let end = tag.find('>').unwrap_or(tag.len());
            if tag[..end]
                .trim_end_matches('/')
                .trim()
                .eq_ignore_ascii_case("br")
            {
                text.push(' ');
            }
            rest = tag.get(end + 1..).unwrap_or_default();
        } else if let Some((decoded, consumed)) = entity(rest) {
            text.push(decoded);
            rest = &rest[consumed..];
        } else {
            text.push('&');
            rest = &rest[1..];
        }
    }
    text.push_str(rest);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The character an HTML entity at the head of `text` names, and its length.
fn entity(text: &str) -> Option<(char, usize)> {
    let end = text.bytes().take(12).position(|byte| byte == b';')?;
    let decoded = match &text[1..end] {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        name => {
            let number = name.strip_prefix('#')?;
            let code = match number.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => number.parse().ok()?,
            };
            char::from_u32(code)?
        }
    };
    Some((decoded, end + 1))
}
