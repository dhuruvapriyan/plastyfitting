# Simplified Two-Stage CICR: IP3R Trigger + RyR Amplifier

Final system of equations for the simplified CICR-augmented Graupner-Brunel
plasticity model. Replaces the full Li-Rinzel IP3R formulation with a minimal
biophysically-motivated cascade: IP3R provides the coincidence trigger,
RyR provides regenerative amplification.

Source: [cicr_er_ip3_simple.py](file:///project/rrg-emuller/dhuruva/plastyfitting/scripts/cicr_er_ip3_simple.py)

---

## Design Principles

1. **IP3 is required for release** — bAPs alone cannot trigger CICR (Hong & Ross 2007)
2. **SERCA loads ER during bAPs** — this is the slow priming/memory variable (~1–2 s)
3. **RyR amplifies IP3R-initiated release** — steep Hill function provides switch-like gain
4. **No gating ODEs** — all dynamics are simple exponential decays, cnexp-solvable in NEURON
5. **LTD is NOT from CICR** — LTD comes from the Graupner-Brunel depression zone, which implicitly captures the eCB/mGluR pathway (Nevian & Sakmann 2006)

---

## State Variables

| Variable | Description | Units |
|----------|-------------|-------|
| $IP_3$ | Inositol trisphosphate (mGluR coincidence signal) | mM |
| $Ca_{ER}$ | ER luminal calcium (priming state) | mM |
| $ca_{cicr}$ | Cytosolic calcium released from ER | mM |
| $h_{ref}$ | Refractory inactivation gate (Ca²⁺-dependent) | — |
| $effcai$ | Temporally-smoothed effective calcium | mM |
| $\rho$ | Synaptic efficacy (Graupner-Brunel weight) | — |

---

## 1. IP3 Dynamics

IP3 is produced by presynaptic vesicle release ($n_{ves}$) via mGluR → PLC activation.
Decays exponentially (cnexp-solvable):

$$IP_3(t + dt) = \left(IP_3(t) + \delta_{IP3} \cdot n_{ves}\right) \cdot e^{-dt/\tau_{IP3}}$$

---

## 2. Total Cytosolic Calcium

$$Ca_{ext} = \max(0,\; Ca_{raw} - Ca_{rest})$$

$$Ca_i = Ca_{ext} + ca_{cicr}$$

---

## 3. Stage 1 — IP3R Trigger (Graded)

IP3R activation saturates via a Hill-like function (analogous to Li-Rinzel $m_\infty$).
No IP3 → no release, regardless of ER loading or cytosolic calcium. This is the coincidence gate.

$$ip3_{act} = \frac{IP_3}{IP_3 + K_{IP3}}$$

$$\boxed{J_{IP3R} = V_{IP3R} \cdot ip3_{act} \cdot h_{ref} \cdot Gradient}$$

where $Gradient = \max(0,\; Ca_{ER} - Ca_i)$ and $K_{IP3} = 0.001$ mM (1 µM).

---

## 4. Stage 2 — RyR Amplifier (Switch-Like)

RyR2 opens when **ER-released calcium** ($ca_{cicr}$) exceeds $K_{RyR}$.
RyR is gated by $ca_{cicr}$ only (not total $Ca_i$), ensuring that VDCC calcium
from bAPs alone cannot trigger CICR — IP3R must release Ca first (Hong & Ross 2007).
Algebraic Hill function — no ODE, no stiffness.

$$P_{RyR} = \frac{ca_{cicr}^{\,3}}{K_{RyR}^{\,3} + ca_{cicr}^{\,3}}$$

$$\boxed{J_{RyR} = V_{RyR} \cdot P_{RyR} \cdot h_{ref} \cdot Gradient}$$

**Cascade logic:** Pre fires → IP3 produced → IP3R releases small Ca from ER →
$ca_{cicr}$ rises above $K_{RyR}$ → RyR opens → large amplified release.

Without presynaptic input: no IP3 → no IP3R release →
$ca_{cicr}$ stays at 0 → RyR stays closed (even if VDCC Ca is high).

---

## 4b. Refractory Inactivation Gate ($h_{ref}$)

Ca²⁺-dependent inactivation of IP3R/RyR channels, inspired by the Li-Rinzel $h$ gate
and constrained by Caya-Bissonnette et al. (2023) refractory timescale ($\tau_r = 665$ ms
in L5 mPFC pyramidal neurons).

When CICR releases calcium ($ca_{cicr}$ rises), $h_{ref}$ drops toward 0 → channels inactivate.
When $ca_{cicr}$ is cleared, $h_{ref}$ recovers toward 1.0 with time constant $\tau_{ref}$.

$$h_\infty = \frac{K_{h,ref}^{\,2}}{K_{h,ref}^{\,2} + ca_{cicr}^{\,2}}$$

$$\frac{dh_{ref}}{dt} = \frac{h_\infty - h_{ref}}{\tau_{ref}}$$

**BTSP mechanism:** At 125–250 ms post→pre delays, $h_{ref}$ is still suppressed from the
first CICR event → weak/no amplification → LTD dip. At 500–750 ms, $h_{ref}$ has recovered
→ full CICR available → LTP. This produces the characteristic biphasic timing curve.

---

## 5. SERCA Pump (ER Loading / Priming)

SERCA pumps **VDCC calcium only** ($Ca_{ext}$) into the ER, not CICR-released calcium.
During bAP trains, elevated $Ca_{ext}$ drives SERCA → $Ca_{ER}$ rises → ER becomes "primed."
Released $ca_{cicr}$ is extruded from the cell, not recycled to ER — this gives
**single-shot refractory behavior**: after CICR depletes the ER, it stays depleted
until the next bAP train reprimes it via SERCA.

$$J_{SERCA} = V_{SERCA} \cdot \frac{Ca_{ext}^{\,2}}{K_{SERCA}^{\,2} + Ca_{ext}^{\,2}}$$

---

## 6. Passive ER Leak

Sets the priming decay timescale: $\tau_{priming} = \gamma_{ER} / V_{leak}$.
With $V_{leak}$ bounded to [6×10⁻⁵, 2×10⁻⁴] /ms, priming lasts **0.6–2.1 seconds**.

$$J_{leak} = V_{leak} \cdot Gradient$$

---

## 7. ER and Cytosolic Pool Dynamics

#### ER store (capped at $Ca_{ER,max} = 2.0$ mM)

$$\frac{dCa_{ER}}{dt} = \frac{J_{SERCA} - J_{IP3R} - J_{RyR} - J_{leak}}{\gamma_{ER}}, \qquad Ca_{ER} \in [0,\; Ca_{ER,max}]$$

#### Cytosolic CICR pool

$$\frac{dca_{cicr}}{dt} = J_{IP3R} + J_{RyR} + J_{leak} - J_{SERCA} - \frac{ca_{cicr}}{\tau_{extrusion}}$$

---

## 8. Plasticity (Graupner-Brunel)

#### Effective calcium integrator (boosted by CICR)

$$\frac{d(effcai)}{dt} = \frac{(Ca_{ext} + ca_{cicr}) - effcai}{\tau_{eff}}$$

#### Thresholds (per-synapse, location-dependent)

$$\theta_d = \begin{cases} a_{20}\,c_{pre} + a_{21}\,c_{post} & \text{apical} \\ a_{00}\,c_{pre} + a_{01}\,c_{post} & \text{basal} \end{cases}$$

$$\theta_p = \max\!\left(\begin{cases} a_{30}\,c_{pre} + a_{31}\,c_{post} & \text{apical} \\ a_{10}\,c_{pre} + a_{11}\,c_{post} & \text{basal} \end{cases},\; \theta_d + 0.01\right)$$

where $c_{pre}, c_{post}$ are peak effcai from isolated pre/post-only calcium traces, scaled by $\tau_{eff}/\tau_{ref}$.

#### Bistable weight dynamics

$$pot = \mathbb{1}[effcai > \theta_p], \qquad dep = \mathbb{1}[effcai > \theta_d]$$

$$\frac{d\rho}{dt} = \frac{-\rho(1-\rho)(0.5-\rho) + \gamma_p(1-\rho)\cdot pot - \gamma_d\,\rho\cdot dep}{\tau_\rho}$$

---

## BTSP Mechanism (Behavioral Timescale Synaptic Plasticity)

The model naturally supports two BTSP orderings at ~750 ms delays:

### Post → Pre (750 ms): LTP

```
bAP train → SERCA loads ER (Ca_ER rises above baseline)
  ... 750 ms pass ... (Ca_ER stays elevated, tau_priming ~ 1-2 s)
Pre stimulus → fresh IP3 bolus → IP3R opens on LOADED ER (large gradient)
  → IP3R release raises Ca_i above K_RyR → RyR amplifies → strong CICR → LTP
```

### Pre → Post (750 ms): Weak/No Change

```
Pre stimulus → IP3 produced (tau_IP3 ~ 330 ms)
  ... 750 ms pass ... (IP3 decays to ~10%, ER modestly loaded)
bAP train → Ca_i rises from VDCCs, but IP3 depleted → IP3R weak
  → insufficient IP3R release to cross K_RyR → no RyR amplification → no CICR
  → Ca from VDCCs alone may land in depression zone → LTD or no change
```

### Why Bidirectionality Works

| Ordering (750 ms) | IP3 at 2nd stim | Ca_ER at 2nd stim | h_ref | CICR | Result |
|---|---|---|---|---|---|
| Post → Pre | **Fresh** bolus | **Loaded** | ~1.0 (recovered) | Strong | **LTP** |
| Pre → Post | ~10% residual | Moderate | ~1.0 | Weak/None | **LTD or no Δ** |

The asymmetry arises because the **primer** (Ca_ER) lasts 1–2 s but the
**trigger** (IP3) decays in ~330 ms. Post → Pre preserves the trigger;
Pre → Post loses it.

### Refractory LTD Dip (125–250 ms)

At short delays (125–250 ms), even if IP3 and Ca_ER are favorable, $h_{ref}$ is still
suppressed from the initial CICR event → channels remain inactivated → weak/no CICR
→ calcium lands in the depression zone → **LTD dip**.

| Delay | h_ref state | CICR available | Result |
|---|---|---|---|
| 0–50 ms | ~1.0 (no prior CICR) | Yes | LTP (if coincident) |
| 125–250 ms | Suppressed (~0.2–0.5) | Reduced | **LTD** |
| 500–750 ms | Recovered (~0.8–1.0) | Full | **LTP** (BTSP) |

---

## Fixed Biophysical Constants

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| $K_{IP3}$ | 0.001 mM (1 µM) | IP3R half-activation for IP3 | Li-Rinzel $d_1$ |
| $K_{RyR}$ | 0.0005 mM (0.5 µM) | RyR2 half-activation for Ca | Bhatt et al.; Bhatt & Bhalla 2022 |
| $n_{RyR}$ | 3 | RyR Hill coefficient | Bhatt & Bhalla 2022 |
| $\gamma_{ER}$ | 0.1255 | ER/cytosol volume ratio | Morphological |
| $Ca_{rest}$ | 70 nM | Resting cytosolic calcium | Standard |
| $Ca_{ER,0}$ | 0.4 mM | Initial ER calcium (resting SS) | — |
| $Ca_{ER,max}$ | 2.0 mM | Maximum ER calcium (finite capacity) | — |
| $\tau_\rho$ | 70000 ms | Graupner-Brunel time constant | Graupner & Brunel 2012 |

---

## Free Parameters (21)

### Plasticity (10)

| Parameter | Bounds | Description |
|-----------|--------|-------------|
| $\gamma_d$ | [50, 250] | Depression rate |
| $\gamma_p$ | [50, 300] | Potentiation rate |
| $a_{00}$ | [1, 5] | $\theta_d$ basal, pre weight |
| $a_{01}$ | [1, 5] | $\theta_d$ basal, post weight |
| $a_{10}$ | [1, 5] | $\theta_p$ basal, pre weight |
| $a_{11}$ | [1, 5] | $\theta_p$ basal, post weight |
| $a_{20}$ | [1, 5] | $\theta_d$ apical, pre weight |
| $a_{21}$ | [1, 5] | $\theta_d$ apical, post weight |
| $a_{30}$ | [1, 10] | $\theta_p$ apical, pre weight |
| $a_{31}$ | [1, 5] | $\theta_p$ apical, post weight |

### CICR / Scaling (11)

| Parameter | Bounds | Description |
|-----------|--------|-------------|
| $\delta_{IP3}$ | [0.0005, 0.05] mM | IP3 bolus per vesicle (0.5–50 µM) |
| $\tau_{IP3}$ | [200, 1000] ms | IP3 decay time constant |
| $V_{IP3R}$ | [1e-5, 0.1] /ms | IP3R release conductance |
| $V_{RyR}$ | [1e-5, 0.1] /ms | RyR release conductance |
| $V_{SERCA}$ | [1e-5, 0.1] mM/ms | SERCA max pump rate |
| $K_{SERCA}$ | [0.0005, 0.01] mM | SERCA half-activation (0.5–10 µM) |
| $V_{leak}$ | [6e-5, 2e-4] /ms | ER leak (sets $\tau_{priming}$ = 0.6–2.1 s) |
| $\tau_{extrusion}$ | [10, 100] ms | Cytosolic CICR clearance (Caya $\tau_c$ = 23.5 ms) |
| $\tau_{eff}$ | [50, 500] ms | Effective calcium time constant |
| $\tau_{ref}$ | [200, 1500] ms | Refractory recovery time (Caya $\tau_r$ = 665 ms) |
| $K_{h,ref}$ | [0.0001, 0.005] mM | Inactivation half-point (0.1–5 µM) |
