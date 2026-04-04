# Unified Quality Model DAG Legend

## Node Styles

| style | meaning |
| --- | --- |
| white ellipse | latent random variable |
| gray ellipse | deterministic / derived node |
| gray box | observed variable |
| thin black box with faint gray fill | plate |

## Symbols

| symbol | meaning |
| --- | --- |
| `x̄_u` | pooled subject-beauty feature source |
| `b_u` | subject beauty |
| `x_i` | frozen asset embedding |
| `τ_i` | 3D technical nuisance descriptor |
| `v_i` | standardized 3D vibe descriptor |
| `δ_i` | hard 3D gate |
| `c_i` | fixed semantic basis coordinates |
| `g_i` | dominant-face beauty contribution |
| `h_i` | fixed perturbation basis `[c_i ; δ_i · v_i]` |
| `a_i` | cross-session baseline quality |
| `κ_i` | 3D technical quality |
| `Q0_i` | canonical asset quality |
| `ρ_s` | raw session-importance gate parameter |
| `λ_s` | nonnegative session-importance gate |
| `w_s` | session perturbation weights |
| `T_s` | session unary threshold |
| `η_is` | explicit asset-session memory offset |
| `p_is` | gated perturbation `λ_s · w_sᵀ h_i` |
| `U_is` | total asset utility in session `s` |
| `Δ_face` | subject-beauty gap |
| `y_face` | facemash duel outcome |
| `Δ_asset` | asset-utility gap |
| `y_asset` | asset duel outcome |
| `r_is` | unary reject / keep / heart observation |
