# Li-Rinzel CICR with theta_p ReLU RyR Gate

Final system of equations for the CICR-augmented Graupner-Brunel plasticity model.
The key innovation is gating ER calcium release by a ReLU on `effcai - theta_p`,
tying CICR to the CaMKII activation regime and preventing spurious release in
distal dendrites.

Source: [cicr_er_ip3_thetap_ryr.py](file:///project/rrg-emuller/dhuruva/plastyfitting/scripts/cicr_er_ip3_thetap_ryr.py)
MOD:   [GluSynapse_CICR.mod](file:///project/rrg-emuller/dhuruva/plastyfitting/GluSynapse_CICR.mod)

---

## State Variables

| Variable | Description | Units |
|----------|-------------|-------|
| $IP_3$ | Inositol trisphosphate (mGluR coincidence detector) | mM |
| $Ca_{ER}$ | ER luminal calcium concentration | mM |
| $ca_{cicr}$ | Cytosolic calcium released from ER | mM |
| $h$ | Li-Rinzel IP3R slow inactivation gate | — |
| $effcai$ | Temporally-smoothed effective calcium | mM |
| $\rho$ | Synaptic efficacy (Graupner-Brunel weight) | — |

---

## 1. IP3 Dynamics

IP3 is produced by presynaptic vesicle release ($n_{ves}$) via mGluR activation:

$$\frac{dIP_3}{dt} = \delta_{IP3} \cdot n_{ves} - \frac{IP_3}{\tau_{IP3}}$$

---

## 2. Total Cytosolic Calcium

$$Ca_{ext} = \max(0,\; Ca_{raw} - Ca_{rest})$$

$$Ca_i = Ca_{ext} + ca_{cicr}$$

---

## 3. Li-Rinzel IP3R Open Probability

#### Fast equilibrium gates (instantaneous)

$$m_\infty = \frac{IP_3}{IP_3 + d_1}$$

$$n_\infty = \frac{Ca_i}{Ca_i + d_5}$$

#### Open probability

$$P_{open} = m_\infty^3 \cdot n_\infty^3 \cdot h^3$$

---

## 4. Inactivation Gate $h$

$$Q_2 = d_2 \cdot \frac{IP_3 + d_1}{IP_3 + d_3}$$

$$h_\infty = \frac{Q_2}{Q_2 + Ca_i}, \qquad \tau_h = \frac{1}{a_2\,(Q_2 + Ca_i)}$$

$$\frac{dh}{dt} = \frac{h_\infty - h}{\tau_h}$$

---

## 5. theta_p ReLU RyR Gate (NEW)

CICR release is gated by how far effective calcium exceeds the CaMKII
potentiation threshold $\theta_p$. Below $\theta_p$: zero release. Above: linearly
proportional to the excess.

$$\boxed{G_{RyR} = \max\!\big(0,\; effcai - \theta_p\big)}$$

**Rationale:** CaMKII activation and RyR sensitization share calmodulin as
the calcium sensor. $\theta_p$ represents the calcium level at which CaMKII
transitions to the UP (phosphorylated) state. The ReLU provides graded
amplification — stronger CaMKII activation drives more CICR — and naturally
normalizes across dendritic locations since $\theta_p$ is computed per-synapse.

---

## 6. Calcium Fluxes

$$Gradient = \max(0,\; Ca_{ER} - Ca_i)$$

$$J_{CICR} = V_{CICR} \cdot P_{open} \cdot Gradient \cdot G_{RyR}$$

$$J_{leak} = V_{leak} \cdot Gradient$$

$$J_{SERCA} = V_{SERCA} \cdot \frac{Ca_i^2}{K_{SERCA}^2 + Ca_i^2}$$

Note: $V_{CICR}$ absorbs the ReLU gain (units: /mM/ms rather than /ms).

---

## 7. ER and Cytosolic Pool Dynamics

#### ER store

$$\frac{dCa_{ER}}{dt} = \frac{J_{SERCA} - J_{CICR} - J_{leak}}{\gamma_{ER}}$$

#### Cytosolic CICR pool

$$\frac{dca_{cicr}}{dt} = J_{CICR} + J_{leak} - J_{SERCA} - \frac{ca_{cicr}}{\tau_{extrusion}}$$

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

## Fixed Biophysical Constants

From Li-Rinzel 1994 / GluSynapse_CICR.mod calibration:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $d_1$ | 0.7115 mM | IP3 dissociation constant |
| $d_2$ | 0.5164 mM | Ca inactivation dissociation |
| $d_3$ | 0.6019 mM | IP3 offset in $h$ gate |
| $d_5$ | 0.0533 mM | Ca activation dissociation |
| $a_2$ | 1.8171 /mM/ms | Inactivation rate constant |
| $\gamma_{ER}$ | 0.1255 | ER/cytosol volume ratio |
| $K_{SERCA}$ | 0.8934 mM | SERCA half-activation |
| $V_{leak}$ | 0.0016 /ms | Passive ER leak conductance |
| $Ca_{rest}$ | 70 nM | Resting cytosolic calcium |
| $Ca_{ER,0}$ | 100 mM | Initial ER calcium |
| $h_0$ | 0.8 | Initial $h$ gate (mostly open) |
| $\tau_\rho$ | 70000 ms | Graupner-Brunel time constant |

---

## Fitted Parameters (CMA-ES, 2026-03-01)

16 free parameters, optimized against experimental STDP ratios:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\gamma_d$ | 244.27 | Depression rate |
| $\gamma_p$ | 201.89 | Potentiation rate |
| $a_{00}$ | 1.38 | $\theta_d$ basal, pre weight |
| $a_{01}$ | 2.65 | $\theta_d$ basal, post weight |
| $a_{10}$ | 4.07 | $\theta_p$ basal, pre weight |
| $a_{11}$ | 4.20 | $\theta_p$ basal, post weight |
| $a_{20}$ | 3.97 | $\theta_d$ apical, pre weight |
| $a_{21}$ | 1.47 | $\theta_d$ apical, post weight |
| $a_{30}$ | 3.01 | $\theta_p$ apical, pre weight |
| $a_{31}$ | 2.44 | $\theta_p$ apical, post weight |
| $\delta_{IP3}$ | 2.92 | IP3 bolus per vesicle |
| $\tau_{IP3}$ | 331.65 ms | IP3 decay time constant |
| $V_{CICR}$ | 119.19 /mM/ms | IP3R conductance (absorbs ReLU gain) |
| $V_{SERCA}$ | 1.99 mM/ms | SERCA max pump rate |
| $\tau_{extrusion}$ | 152.32 ms | Cytosolic CICR clearance |
| $\tau_{eff}$ | 131.03 ms | Effective calcium time constant |
