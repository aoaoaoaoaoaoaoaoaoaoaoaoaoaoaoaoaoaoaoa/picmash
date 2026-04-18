
## Rust Style Doctrine

This repository inherits the local Rust style doctrine:
[/home/main/programming/projects/rust_starter/docs/rust-style-doctrine.md](/home/main/programming/projects/rust_starter/docs/rust-style-doctrine.md).

## JavaScript Supply Chain Doctrine

First-party browser JavaScript and TypeScript are allowed. Third-party JavaScript
or TypeScript packages are forbidden by default: no `node_modules`, no npm/pnpm/
yarn/bun package manager surfaces, no JS bundlers, no CDN scripts, and no
third-party browser libraries. TypeScript checking may use the system Arch
`typescript` package (`tsc`) as part of the machine toolchain. Any exception
requires an explicit design discussion and a committed rationale.

<!-- forgejo-assimilation:start -->
## Forgejo Swarm Integration

This repository is integrated with the Forgejo/orchd control plane.

- Use `forgejoctl` for issue workflow mutations (`claim`, `release`, `transition`, `comment`, `assign`).
- Keep issue comments terse, natural-language, and decision-focused.
- Put role directives on their own line when routing work (for example `@codex-lead design`, `@codex-dev impl`).
- Keep orchestration metadata out of issue comments; treat labels as the machine-visible control plane.
- If workflow tooling blocks progress, open a concise bug report in `main/forgejo-agent`.

Canonical swarm docs are injected into fresh agent contexts via `orchd` **Reading material** (DocPlan).
Do not maintain local “go read X” lists in issue comments or repo docs.
<!-- forgejo-assimilation:end -->
