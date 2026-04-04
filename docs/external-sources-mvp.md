# External Sources MVP

This is the current first-pass design for external arena challengers.

It is active code, not a speculative future note.

## Shape

- A `source` is an upstream feed. For now the only implemented kind is a 4chan board.
- A `stream` is an exhaustible sub-feed. For 4chan, a stream is a thread.
- A `remote item` is an image-bearing post discovered under a stream.
- A remote item is not part of the local corpus until it survives local judgment and is imported.

## Config

- Config lives at the XDG config root in `picmash/config.toml`.
- On first boot the app writes a default config if none exists.
- The default source is `4chan:s`.
- `allow_nsfw` defaults to `true`.
- `allow_video` defaults to `false`.
- `min_shortest_edge` defaults to `800`.
- The default import policy is `not_x`.
- Arena external probability is persisted in config and can be changed from the menu.

## Scanner

- The server boots the local corpus first.
- External harvesting runs in the background after boot so arena readiness is not held hostage by upstream latency.
- The scanner:
  - polls the enabled source on an interval
  - discovers live streams from the board catalog
  - fetches a bounded number of streams per scan
  - downloads and caches a bounded number of image posts per stream
  - embeds cached images with the same DINO stack used locally
- Discovered streams/items and their event history live in SQLite.

## Arena

- With probability `p_external`, arena samples an external challenger.
- Sampling is source-local and stream-aware:
  - streams are weighted by a mix of session relevance, posterior yield, freshness, and recency penalties
  - items are weighted by their current session match and their own local success/failure history
- After choosing the external item, arena chooses the best local match jointly rather than replacing one side after a local pair is already formed.
- Arena URLs are explicit typed handles:
  - local: `asset_<sha256>`
  - remote: `remote_<sqlite_id>`

## Outcomes

- `X` on a remote item does not import it.
  - the item is hidden from future remote sampling
  - the parent stream is penalized
- `TX` on a remote item vetoes the entire parent stream.
  - the thread is blocked from future sampling
  - the block persists across rescans
- Non-`X` outcomes under `not_x` import the remote image into the local corpus under `.picmash-imported/<source>/`.
- Imported remote items are then treated as ordinary local assets and participate in the standard duel update path.
- A local win over a remote challenger is a mild stream/item penalty.
- A remote win is a positive signal for the stream/item.
- `♥` on a remote item forces import and hearts the resulting local asset.

## Current Limits

- Only one enabled top-level source is honored at a time.
- The only implemented source kind is `4chan_board`.
- Remote intake is image-only. `allow_video = true` is not implemented yet.
- `allow_nsfw` exists in config now for policy shape and defaults, but the first 4chan board implementation does not apply additional content filtering beyond source choice and image constraints.
