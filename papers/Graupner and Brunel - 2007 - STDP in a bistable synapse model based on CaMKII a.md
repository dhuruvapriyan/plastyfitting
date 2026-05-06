OPEN ACCESS Freely available online
PLOS COMPUTATIONAL BIOLOGY

# STDP in a Bistable Synapse Model Based on CaMKII and Associated Signaling Pathways

**Michael Graupner<sup>1,2,3*</sup>, Nicolas Brunel<sup>1,2</sup>**

**1** Université Paris Descartes, Laboratoire de Neurophysique et Physiologie, Paris, France, **2** CNRS, UMR 8119, Paris, France, **3** Max-Planck-Institut für Physik komplexer Systeme, Dresden, Germany

**The calcium/calmodulin-dependent protein kinase II (CaMKII) plays a key role in the induction of long-term postsynaptic modifications following calcium entry. Experiments suggest that these long-term synaptic changes are all-or-none switch-like events between discrete states. The biochemical network involving CaMKII and its regulating protein signaling cascade has been hypothesized to durably maintain the evoked synaptic state in the form of a bistable switch. However, it is still unclear whether experimental LTP/LTD protocols lead to corresponding transitions between the two states in realistic models of such a network. We present a detailed biochemical model of the CaMKII autophosphorylation and the protein signaling cascade governing the CaMKII dephosphorylation. As previously shown, two stable states of the CaMKII phosphorylation level exist at resting intracellular calcium concentration, and high calcium transients can switch the system from the weakly phosphorylated (DOWN) to the highly phosphorylated (UP) state of the CaMKII (similar to a LTP event). We show here that increased CaMKII dephosphorylation activity at intermediate Ca<sup>2+</sup> concentrations can lead to switching from the UP to the DOWN state (similar to a LTD event). This can be achieved if protein phosphatase activity promoting CaMKII dephosphorylation activates at lower Ca<sup>2+</sup> levels than kinase activity. Finally, it is shown that the CaMKII system can qualitatively reproduce results of plasticity outcomes in response to spike-timing dependent plasticity (STDP) and presynaptic stimulation protocols. This shows that the CaMKII protein network can account for both induction, through LTP/LTD-like transitions, and storage, due to its bistability, of synaptic changes.**

**Citation:** Graupner M, Brunel N (2007) STDP in a bistable synapse model based on CaMKII and associated signaling pathways. PLoS Comput Biol 3(11): e221. doi:10.1371/journal.pcbi.0030221

### Introduction

Synaptic plasticity is thought to underlie learning and memory, but the mechanisms by which changes in synaptic efficacy are induced and maintained over time are still unclear. Numerous experiments have shown how synaptic efficacy can be increased (long-term potentiation, LTP) or decreased (long-term depression, LTD) by spike timing of presynaptic and postsynaptic neurons [1,2], presynaptic firing rate [3,4], or presynaptic firing paired with postsynaptic holding potential [5]. These experiments have led to phenomenological models that capture one or several of these aspects [6–14]. However, these models tell us nothing about the biochemical mechanisms of induction and maintenance of synaptic changes. The question of the mechanisms at the biochemical level has been addressed by another line of research work originating from early work by Lisman (1985) [15]. Models at the biochemical level describe enzymatic reactions of proteins in the postsynaptic density (PSD) [15–19]. These proteins form a network with positive feedback loops that can potentially provide a synapse with several stable states—two, in the simplest case—providing a means to maintain the evoked changes. Hence, synapses in such models are similar to binary switches, exhibiting two stable states, an UP state with high efficacy, and a DOWN state with low efficacy. The idea of binary synapses is supported by recent experiments on CA3-CA1 synapses [20–22].

One of the proposed positive feedback loops involves the calcium/calmodulin-dependent protein kinase II (CaMKII) kinase-phosphatase system [15–19]. CaMKII activation is governed by Ca<sup>2+</sup>/calmodulin binding and is prolonged beyond fast-decaying calcium transients by its autophosphorylation [23]. Autophosphorylation of CaMKII at the residue theronine-286 in the autoregulatory domain (Thr<sup>286</sup>) occurs after calcium/calmodulin binding and enables the enzyme to remain autonomously active after dissociation of calcium/calmodulin [24] (see Materials and Methods). In turn, as long as CaMKII stays activated it is reversibly translocated to a postsynaptic density (PSD)-bound state where it interacts with multiple LTP-related partners structurally organizing protein anchoring assemblies and therefore potentially delivering $\alpha$-amino-3-hydroxyl-5-methyl-4-isoxazole-propionate acid (AMPA) receptors to the cell surface [23,25–28]. The direct phosphorylation of the AMPA receptor GluR1 subunit by active CaMKII enhances AMPA channel function [29,30]. The network involving CaMKII is particularly

**Editor:** Karl J. Friston, University College London, United Kingdom

**Received** April 30, 2007; **Accepted** September 26, 2007; **Published** November 30, 2007

A previous version of this article appeared as an Early Online Release on September 26, 2007 (doi:10.1371/journal.pcbi.0030221.eor).

**Copyright:** © 2007 Graupner and Brunel. This is an open-access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

**Abbreviations:** AMPA, $\alpha$-amino-3-hydroxyl-5-methyl-4-isoxazole-propionate acid; CaMKII, calcium/calmodulin-dependent protein kinase II; cAMP, cyclic adenosine monophosphate; LTD, long-term depression; LTP, long-term potentiation; PKA, protein kinase A; PP1, protein phosphatase 1; PSD, postsynaptic density; STDP, spike-timing dependent plasticity

\* To whom correspondence should be addressed. E-mail: michael.graupner@univ-paris5.fr

PLoS Computational Biology | www.ploscompbiol.org | 2299 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

> ### Author Summary
>
> Learning and memory have been hypothesized to occur thanks to synaptic modifications. The efficacy of synaptic transmission has been shown to change as a function of correlated activity between presynaptic and postsynaptic neurons. Long-lasting synaptic modifications can occur in both directions (long-term potentiation (LTP) and long-term depression (LTD)). Recent experiments suggest that these synaptic changes are all-or-none switch-like changes. This would mean that only two stable states of synaptic transmission efficacy exist, i.e., a low state, or "switched off", and a high state, or "switched on". LTP would correspond to switching on the synapse and LTD to switching off. We propose a realistic biochemical model of protein–protein interactions which exhibits two stable states. We then investigate conditions under which the model exhibits transitions between the two stable states. We show that experimental stimulation protocols known to evoke LTP and LTD lead to corresponding transitions in the model. This work supports the idea that the investigated intracellular protein network has a role in both induction and storage of synaptic changes, and hence in learning and memory storage.

appealing in terms of learning and memory maintenance since N-methyl-D-aspartate receptor (NMDA-R)-dependent LTP requires calcium/calmodulin activation of CaMKII, potentially expressed by the phosphorylation level or the number of AMPA receptors, or both [19,27,28,31–33]. However, the role of CaMKII beyond LTP induction remains controversial [34–36]. Finally, there is experimental evidence for the involvement of proteins associated with CaMKII activity (cyclic adenosine monophosphate (cAMP)–regulated protein kinase A (PKA), protein phosphatase 1 (PP1), and calcineurin) in LTP and LTD [37–40]. We emphasize that multiple mechanisms supporting LTP/LTD induction and expression are likely to be present in synapses of different regions—we focus here on synapses for which the above statements have been shown to apply, e.g., the CA3-CA1 Schaffer collateral synapse (see review by Cooke and Bliss [41]).

Modeling studies have shown that a system including CaMKII and associated pathways could be bistable in a range of calcium concentrations including the resting level—a necessary requirement for the maintenance of long-term changes [15,17,18,42]. In such models, the two states correspond to two stable phosphorylation levels of the CaMKII protein for a given calcium concentration, i.e., a weakly (DOWN) and a highly phosphorylated state (UP). A transition from the DOWN to the UP state which could underlie long-term potentiation (LTP) can be induced by a sufficiently large and prolonged increase in calcium concentration. However, the opposite transition which could underlie depotentiation or LTD only occurs under unrealistic conditions, for example decrease of calcium concentration below resting level. Furthermore, it has not been considered how these biochemical network models behave in response to calcium transients evoked by experimental protocols that are known to induce synaptic plasticity such as STDP, which has been shown to rely on kinase (CaMKII) and phosphatase (calcineurin) activation [43]. Rubin et al. reproduce experimental results on STDP using a model detector system which qualitatively resembles the protein network influencing CaMKII, but this model does not exhibit bistability [44]. Other studies on biochemical signal transduction pathways including CaMKII showed that the AMPA receptor activity can reproduce bidirectional synaptic plasticity as a function of calcium [45,46]. However, realistic stimulation protocols were not investigated in these models, and again they do not show bistability.

In this paper, we consider a realistic model of protein interactions associated with CaMKII autophosphorylation through calcium/calmodulin and dephosphorylation by protein phosphatase 1 in the PSD. We first study the steady-state phosphorylation properties of CaMKII with respect to calcium and changing levels of PP1 activity. Conditions are elaborated for which the system allows for "LTP" and "LTD" transitions in reasonable ranges of calcium concentrations. We then demonstrate the ability of the CaMKII system to perform LTP- or LTD-like transitions in response to STDP stimulation protocols. We expose the CaMKII system to calcium transients evoked by pairs of presynaptic and postsynaptic spikes with a given time lag and show that short positive time lags evoke transitions from the DOWN to the UP state and short negative time lags lead to transitions from the UP to the DOWN state. We demonstrate furthermore that the CaMKII model qualitatively reproduces experimental plasticity outcomes for presynaptic stimulation protocols. Finally, we consider the transition behavior in response to purely presynaptic or postsynaptic spike-pair stimulation protocols.

## Results

We investigate in this paper a realistic model for the protein network of the postsynaptic density, focusing on the pathways affecting the phosphorylation dynamics of CaMKII localized in the PSD. The model describes the calcium/calmodulin-dependent autophosphorylation of CaMKII. Phosphorylation of a CaMKII subunit by its neighboring subunit requires calcium/calmodulin to bind to the substrate subunit. The catalytic subunit is active if bound to Ca<sup>2+</sup>/calmodulin, or phosphorylated (see Figure 1A–1E). Dephosphorylation of phosphorylated CaMKII subunits by PP1 in the PSD is implemented according to the Michaelis-Menten scheme. We also take into account how calcium/calmodulin influences PP1 activity via a protein signaling cascade. PP1 is inhibited by phosphorylated inhibitor 1 (I1). The phosphorylation level of inhibitor 1 is in turn controlled by the balance between a pathway phosphorylating I1 (through cAMP–PKA) and a pathway dephosphorylating I1 (through calcineurin). Therefore, calcineurin activation by calcium/calmodulin increases PP1 activity, while calcium/calmodulin-dependent activation of the cAMP–PKA pathway decreases PP1 activity (Figure 1F). Finally, we model postsynaptic calcium and postsynaptic membrane potential dynamics induced by presynaptic and postsynaptic spikes in order to investigate the effects of spike-induced calcium transients on the dynamics of the system. Details of the model can be found in the Materials and Methods section.

### Bistability of the CaMKII system with Constant PP1 Activity

In this and the following section we investigate how the steady-state values of the total concentration of phosphorylated CaMKII subunits, $S_{active}$, depend on the concentration of calcium and the dephosphorylation activity. We also study how the steady-state behavior changes with the number of

PLoS Computational Biology | www.ploscompbiol.org 2300 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

```mermaid
graph TD
    subgraph "Figure 1. Schematic Representation of Calmodulin Binding to a CaMKII Subunit, Intersubunit Phosphorylation Steps, and the Protein Signaling Cascade"
        subgraph "A-E: CaMKII Subunit Reactions"
            A[Dephosphorylated Subunit + C] -- "k<sub>5</sub> / k<sub>-5</sub>" --> A_bound[C-bound Subunit]
            B[Phosphorylated Subunit + C] -- "k<sub>9</sub> / k<sub>-9</sub>" --> B_bound[C-bound Phosphorylated Subunit]
            
            C_step[Initiation: cat. sub. + sub. sub. both C-bound] -- "k<sub>6</sub>" --> C_result[Phosphorylated Subunit]
            D_step[cat. sub. Phosphorylated & C-bound + sub. sub. C-bound] -- "k<sub>7</sub>" --> D_result[Phosphorylated Subunit]
            E_step[cat. sub. Phosphorylated + sub. sub. C-bound] -- "k<sub>8</sub>" --> E_result[Phosphorylated Subunit]
        end

        subgraph "F: Protein Signaling Cascade"
            C_signal((C)) -- blue line --> CaMKII[CaMKII]
            C_signal -- red dashed line --> cAMP_PKA[cAMP-PKA]
            cAMP_PKA -- red dashed line o --> Inhibitor1[Inhibitor 1 phosphorylated]
            Inhibitor1 -- red dashed line | --> PP1[PP1]
            C_signal -- red dotted line --> Calcineurin[Calcineurin]
            Calcineurin -- red dotted line o --> Inhibitor1
            PP1 -- red line | --> CaMKII
        end
    end
```

**Figure 1. Schematic Representation of Calmodulin Binding to a CaMKII Subunit, Intersubunit Phosphorylation Steps, and the Protein Signaling Cascade**
(A–E) Show calcium/calmodulin binding and phosphorylation reactions of a ring of six functionally coupled subunits of the CaMKII holoenzyme. A gray subunit stands for dephosphorylated and a green subunit surrounded by a dotted line for phosphorylated. (A,B) The calcium/calmodulin complex—C, blue circle—can bind to a dephosphorylated (A) or a phosphorylated subunit (B) with dissociation constants $K_5 = k_{-5} / k_5$ or $K_9 = k_{-9} / k_9$, respectively. Note that the calmodulin binding (A,B) and the autophosphorylation steps (shown in C–E) are assumed to take place independently of the phosphorylation state of other subunits in the ring (here depicted as dephosphorylated, i.e., in gray). Subunits shown in dotted gray can be either dephosphorylated or phosphorylated.
(C–E) The three possible intersubunit phosphorylation steps: in all three cases, the catalytic subunit and the substrate subunit are labeled with cat. and sub., respectively. Unlabeled subunits are depicted as dephosphorylated, but the three phosphorylation steps are assumed to proceed independently of their phosphorylation state.
(C) Initiation step: calmodulin has to bind to the two interacting subunits, i.e., to the substrate and the catalyst, in order to phosphorylate the substrate subunit at Thr<sup>286</sup> (shown in green and surrounded by a dotted line).
(D) Calmodulin is bound to the phosphorylated catalyst and the subunit to be phosphorylated.
(E) The phosphorylated subunit stays active as catalyst after calmodulin dissociation and phosphorylates the substrate subunit bound with calmodulin. $k_6$, $k_7$, and $k_8$ denote the respective autophosphorylation rates of the three steps described above.
(F) Protein signaling cascade governing PP1 activity. Interactions shown with a circle at the end of a line indicate stimulation—whereas lines ending with a bar stand for inhibition of target activity. Calcium/calmodulin—C, blue circle—directly phosphorylates CaMKII (blue line). Furthermore, the dephosphorylation of CaMKII by protein phosphatase 1 (PP1) is indirectly controlled by calcium/calmodulin via a protein signaling cascade (red lines). Calcium/calmodulin-directed phosphorylation of inhibitor 1 via cyclic-AMP and PKA increases CaMKII activity by inhibiting PP1. This CaMKII stimulating pathway is depicted in red dashed lines. On the contrary, activation of calcineurin activates PP1 by dephosphorylating inhibitor 1, which in turn leads to increased CaMKII dephosphorylation. This pathway, shown in red dotted lines, decreases CaMKII activity.
doi:10.1371/journal.pcbi.0030221.g001

interacting subunits in the cluster. We start by exploring Ca<sup>2+</sup>/calmodulin-stimulated autophosphorylation of CaMKII at a fixed dephosphorylation activity. This will allow us later to better understand how the parameters of the signaling cascade controlling dephosphorylation activity affect the phosphorylation behavior of CaMKII. To do this, we set the PP1 dephosphorylation activity to a constant, independent of the calcium concentration (this is equivalent to removing the red lines in Figure 1F except for the interaction between CaMKII and PP1). The PP1 dephosphorylation activity is the product of $k_{12}$, the maximal dephosphorylation rate, and $D$, the free PP1 concentration (see Equation 6).

Figure 2A shows the steady-state concentration of phosphorylated CaMKII subunits as a function of the calcium concentration for 2, 4, 6, and 8 functionally connected subunits in the CaMKII cluster. The graphs show that in all cases there exists a range of calcium concentration for which the system is bistable (region between the diamond and the circle in the case of the six-subunit model). In the bistable region, three steady-states are present. The top and the bottom steady-states (depicted by the thick full lines) are stable, whereas the intermediate one (dashed thin lines) is unstable. The branch of unstable steady-states separates the basins of attraction of the highly and the weakly phosphorylated stable steady-states. This means that the system will converge to the UP state if it is initially above this line, while it will converge to the DOWN state if it is below this line. As in other studies on CaMKII bistability, the bistable phosphorylation behavior emerges from the combination of strong cooperativity of CaMKII autophosphorylation and the saturation of the per-subunit dephosphorylation rate, $k_{10}$ (see Equation 6 in Materials and Methods), at high phosphorylation levels [15,17,18]. This saturation arises from the Michaelis-Menten approach employed to describe dephosphorylation, which is valid if the enzyme (PP1) is present in small amounts compared to the substrate (phosphorylated subunits). This is plausible since the CaMKII protein is localized at high concentrations in the PSD [24,27,47].

Figure 2A demonstrates that the increasing saturation of the per-subunit dephosphorylation rate with increasing number of interacting subunits in the holoenzyme ring plays a crucial role in the extent of the bistable region. Whereas the difference between the two- and the four-subunit model is very pronounced, increasing further the number of subunits has less and less impact on the size of the bistable region—it still increases substantially when this number goes from four to six, but there is almost no noticeable difference between the six- and the eight-subunit model. The effect of the number of subunits on the extent of the bistable region is mainly due to an increase in the range of stability of the UP state with increasing subunit number, since the stronger saturation of the per-subunit dephosphorylation rate becomes apparent in the highly phosphorylated state only (see Equation 6). On the other hand, the range of stability of the DOWN state is essentially unaffected by the number of subunits, since it is mostly controlled by the balance between the dephosphorylation rate and the probability of the initiation step to occur. Interestingly, experimental data

PLoS Computational Biology | www.ploscompbiol.org 2301 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

indicate that the number of functionally coupled subunits in a CaMKII holoenzyme ring is six [48–50], which could be a good compromise between having both a relatively small number of subunits and a large bistability range. In the following, we consider exclusively a model with six subunits.

How the location and the extent of the bistable region changes with respect to the PP1 dephosphorylation activity is shown in Figure 2B. The curves depict the boundaries of the bistable region for the six-subunit model in the PP1 activity—calcium concentration plane, for three values of the total calmodulin concentration, $CaM_0$ (indicated by the three different colors). For each value of $CaM_0$, the colored area shows the bistable region in which the UP and the DOWN states coexist. Above the colored area, only the DOWN state is present, while below that area only the UP state is present. The resting calcium concentration can be included in the bistable region, provided the PP1 activity is chosen accordingly (e.g., $(k_{12} \cdot D) = 6.648\ \mu M/s$ for $CaM_0 = 0.1\ \mu M$ in Figure 2B).

### "LTD Window" in a Model with Ca-Dependent PP1 Activity via Protein Signaling Cascade Including PKA and Calcineurin

The right-hand boundary of the bistable region in Figure 2A corresponds to a down-to-up switching threshold: if the calcium concentration increases persistently above this level, the CaMKII will converge from a weakly phosphorylated to a highly phosphorylated state (down-to-up switching). Hence, we define the range above this right-hand bifurcation point "LTP window". It corresponds to high calcium concentrations, consistent with experimental data on the range of calcium concentrations leading to LTP.

The available experimental data also suggest that (i) at resting calcium concentrations, no transitions should occur (both UP and DOWN states should be stable), (ii) for intermediate calcium concentrations (higher than resting concentration, but lower than the down-to-up switching threshold), LTD transitions should occur. This would happen if the UP state was no longer stable in an intermediate range of calcium concentrations—in such a scenario, the UP state would be stable in two disconnected regions, one around resting calcium concentration, and the other one at high calcium concentrations. The region where the UP state would not be stable could be called "LTD window" since the system would exhibit LTD (up-to-down switching) whenever the calcium concentration stays in that region for a sufficiently long time, i.e., the CaMKII would converge from a highly phosphorylated state to a weakly phosphorylated state in this range of calcium. The scenario depicted in Figure 2 seems at odds, however, with this picture.

How can the steady-state picture of Figure 2A be modified to obtain such an LTD window? A possible scenario is to take into account the protein signaling cascade governing PP1 dephosphorylation activity in a calcium/calmodulin-dependent manner (see Figure 1F). In this way, the active concentration of PP1, $D$, changes with calcium, and the region of bistability is no longer defined by a horizontal line in Figure 2B. Rather, the location and extent of the LTD and the LTP windows are given by the intersections of the curve describing how the steady-state PP1 activity changes with calcium concentration with the curves specifying the location

**A**
[The image shows a line graph (Figure 2A) plotting the concentration of phosphorylated CaMKII subunits $S_{active}$ ($\mu M$) against calcium concentration $Ca$ ($\mu M$). It shows curves for different numbers of subunits (2, 4, 6, 8). Solid lines represent stable steady-states, and dashed lines represent unstable steady-states. A vertical line marks the resting calcium concentration at 0.1 $\mu M$.]

**B**
[The image shows a log-linear graph (Figure 2B) plotting PP1 activity $k_{12} \cdot D$ ($\mu M/s$) against calcium concentration $Ca$ ($\mu M$). Three shaded regions represent bistability for different calmodulin concentrations: $CaM_0 = 1\ \mu M$ (green), $CaM_0 = 0.1\ \mu M$ (red), and $CaM_0 = 0.01\ \mu M$ (blue). A diamond and a circle mark the boundaries for the six-subunit case at $CaM_0 = 0.1\ \mu M$.]

**Figure 2. CaMKII Bistability Properties with Constant PP1 Activity**
(A) Bistability of the steady-state concentration of phosphorylated CaMKII subunits as a function of calcium concentration, for various numbers of subunits: the steady-state concentration of phosphorylated CaMKII subunits, $S_{active}$, is shown for different numbers of subunits in an interacting cluster (subunit number indicated). Dashed thin lines characterize unstable steady-states and full lines show stable steady-states. The left-hand calcium boundary of the bistable region for the six-subunit model is indicated by the diamond and the right-hand boundary by the circle. In all cases, the total concentration of subunits is $200\ \mu M$, i.e., $CaMKII_0 = 50\ \mu M$ for the two-subunit case, $CaMKII_0 = 25\ \mu M$ for the four-subunit case, $CaMKII_0 = 16.67\ \mu M$ for the six-subunit, and $CaMKII_0 = 12.5\ \mu M$ for the eight-subunit case. The total level of calmodulin, $CaM_0$, is $0.1\ \mu M$ for all cases. The vertical full line shows the position of the calcium resting concentration in (A) and (B). The PP1 activity is kept constant at $(k_{12} \cdot D) = 6.648\ \mu M/s$.
(B) Boundaries of the bistable region in the PP1 activity—calcium concentration plane for different levels of calmodulin with the six-subunit model: lines of any given color depict the location of the left-hand (upper line) and the right-hand (lower line) boundaries of the bistable region with respect to the PP1 activity. Shaded areas between both boundaries mark the regions of bistability in the PP1 activity—calcium plane. Different colors correspond to different levels of calmodulin as indicated in the panel. The diamond illustrates the position with respect to calcium of the left-hand boundary ($Ca = 0.091\ \mu M$) and the circle of the right-hand boundary ($Ca = 0.129\ \mu M$) of the bistable region for parameters as for the six-subunit case in (A) ($CaM_0 = 0.1\ \mu M$, $(k_{12} \cdot D) = 6.648\ \mu M/s$). See Table 1 for other parameters.
doi:10.1371/journal.pcbi.0030221.g002

PLoS Computational Biology | www.ploscompbiol.org 2302 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

![Figure 3](https://example.com/figure3.png)

**Figure 3. Ca<sup>2+</sup>-Dependent PP1 Activity via Protein Signaling Cascade and Phosphorylated CaMKII Subunit Concentration Steady-States**

(A) Phosphorylation and dephosphorylation rate of inhibitor 1 as functions of the calcium concentration: the red line shows the calcineurin activity dephosphorylating inhibitor 1, $v_{CaN}(C)$, as a function of calcium, and the blue line shows the PKA activity leading to phosphorylation of I1, $v_{PKA}(C)$. Both rates are given by Equation 9, in which the calcium/calmodulin complex concentration, $C$, is in equilibrium with the present calcium concentration. The total calmodulin concentration is $CaM_0 = 0.1\ \mu M$. The vertical thin full line depicts the position of the calcium resting concentration in all panels.

(B) Bistable region, PP1 activity, and I1P steady-state concentration in the PP1–Ca<sup>2+</sup> plane: the full lines correspond to a total CaMKII concentration of $CaMKII_0 = 16.67\ \mu M$, whereas dashed lines correspond to $CaMKII_0 = 8.33\ \mu M$. The full red lines mark the positions of the left-hand (line above) and right-hand (line below) boundaries of the bistable region (red shaded region between both lines) with respect to the PP1 activity for $CaM_0 = 0.1\ \mu M$. The purple and the green full lines show the calcium-dependent steady-state of the PP1 activity, $(k_{12} \cdot D)$, and I1P concentration, $I$, respectively, given by the rates shown in (A). The numbered points 1, 2, 3, and 4 at the intersections of the PP1 steady-state with the full red lines give the locations of the boundaries (saddle-node bifurcation points) of the bistable regions. The dashed red and purple lines depict the boundaries of the bistable region and the steady-state PP1 activity, respectively, for the case $CaMKII_0 = 8.33\ \mu M$ (see text).

(C) Steady-states of the phosphorylated CaMKII subunit concentration ($S_{active}$) as a function of calcium: full lines characterize stable steady-states whereas dashed lines mark unstable steady-states. Steady-states for $CaMKII_0 = 16.67\ \mu M$ are shown in blue and for $CaMKII_0 = 8.33\ \mu M$ in green. Red shaded areas depict regions of bistability for the $CaMKII_0 = 16.67\ \mu M$ case, i.e., three steady-states (two stable and one unstable steady-state) exist for a given calcium concentration. Bifurcation points are numbered as in (B). The cross (diamond) marks the position of the UP (DOWN) state at resting calcium concentration for the $CaMKII_0 = 16.67\ \mu M$ case (see Table 1 and text for parameters, $k_{CaN} = 18\ 1/s$).
doi:10.1371/journal.pcbi.0030221.g003

of the left- and the right-hand boundary of the bistable region in the PP1 activity–calcium concentration plane.

Figure 3B shows an example in which the steady-state PP1 activity ($k_{12} \cdot D$) versus calcium concentration curve (purple line) intersects the bifurcation lines (red lines) four times, such that an LTD window emerges in a range of intermediate calcium concentrations. As Figure 3B shows, this can be obtained whenever the PP1 activity has a sufficiently large peak at some intermediate calcium concentrations. This peak has to be such that in a range of calcium concentrations, PP1 activity is above both bifurcation lines (region between intersection points 2 and 3 in Figure 3B). As discussed above, only the DOWN state is stable in this region. This PP1 peak is in turn obtained due to PKA activating at higher calcium concentration than calcineurin, since the balance between calcineurin and PKA activity determines the level of PP1 inhibition via inhibitor 1 (see Figures 1F and 3A). Hence, the peak in steady-state PP1 concentration at intermediate calcium concentrations is due to a relative increase in calcineurin activity with respect to PKA activity in this range (compare Figure 3A and 3B). To include the calcium resting concentration, $Ca_0$ (marked by the vertical thin line in Figure 3), in a region of bistability, PP1 activity at $Ca_0$ has to reside in between the bifurcation lines. The fourth intersection point defines the down-to-up switching threshold, i.e., the left-hand boundary of the LTP window (see point 4 in Figure 3B and 3C). The range of bistability between points marked 3 and 4 in Figure 3C emerges from the declining PP1 activity (purple line in Figure 3B) crossing the ascending range of bistability (red shaded area in Figure 3B). These opposing trends lead to a narrow range of bistability at high calcium concentrations in the example presented here since the intersections of both define the borders of the bistable region.

In practice, the location of these four intersection points can be chosen by adjusting parameters describing the calcium/calmodulin-dependent activation of PKA and calcineurin activity (see Materials and Methods section for more details). We can obtain four such parameters (PKA base and maximal activity $k^0_{PKA}$ and $k_{PKA}$, respectively, the PKA half activity concentration $K_{PKA}$ and the PKA Hill coefficient $n_{PKA}$ (see Equation 9)) by simultaneously solving four equations, i.e., one for each of the four intersection points 1, 2, 3, and 4 at $Ca = 0.09, 0.22, 0.36,$ and $0.37\ \mu M$ (see Figure 3B). Figure 3A displays the resulting instantaneous calcium/calmodulin-dependent phosphorylation $v_{PKA}$ and dephosphorylation $v_{CaN}$ rates of inhibitor 1 (see Equation 23 in Materials and Methods) leading to the steady-state PP1 concentration scenario shown in Figure 3B by the full purple line. The parameters obtained through this procedure can be found in

PLoS Computational Biology | www.ploscompbiol.org | 2303 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

# Table 1. Parameters of the Model

<table>
  <tbody>
    <tr>
        <td>Parameter</td>
        <td colspan="2">Definition</td>
        <td>Value</td>
        <td>Range: [X–Y]</td>
        <td>Unit</td>
        <td>Comment and/or Reference</td>
    </tr>
    <tr>
        <th>A, parameters determined by experiments</th>
        <th>K₁</th>
        <th>Dissociation constant to calcium binding to calmodulin</th>
        <th>0.1</th>
        <th>—</th>
        <th>µM</th>
        <th>[100]</th>
    </tr>
    <tr>
        <th></th>
        <th>K₂</th>
        <th>Dissociation constant to calcium binding to calmodulin</th>
        <th>0.025</th>
        <th>—</th>
        <th>µM</th>
        <th>[100]</th>
    </tr>
    <tr>
        <th></th>
        <th>K₃</th>
        <th>Dissociation constant to calcium binding to calmodulin</th>
        <th>0.32</th>
        <th>—</th>
        <th>µM</th>
        <th>[100]</th>
    </tr>
    <tr>
        <th></th>
        <th>K₄</th>
        <th>Dissociation constant to calcium binding to calmodulin</th>
        <th>0.4</th>
        <th>—</th>
        <th>µM</th>
        <th>[100]</th>
    </tr>
    <tr>
        <th></th>
        <th>CaMKII₀</th>
        <th>Total CaMKII concentration</th>
        <th>8.33, 16.67</th>
        <th>—</th>
        <th>µM</th>
        <th>[73]</th>
    </tr>
    <tr>
        <th></th>
        <th>K₅</th>
        <th>Dissociation constant between dephosphorylated subunit and C</th>
        <th>0.1</th>
        <th>—</th>
        <th>µM</th>
        <th>[102]</th>
    </tr>
    <tr>
        <th></th>
        <th>K₉</th>
        <th>Dissociation constant between phosphorylated subunit and C</th>
        <th>1 · 10⁻⁴</th>
        <th>—</th>
        <th>µM</th>
        <th>[123]</th>
    </tr>
    <tr>
        <th></th>
        <th>k₁₃</th>
        <th>I1P, PP1 association rate</th>
        <th>500</th>
        <th>—</th>
        <th>1/(s · µM)</th>
        <th>[124,125]</th>
    </tr>
    <tr>
        <th></th>
        <th>k₋₁₃</th>
        <th>I1P, PP1 dissociation rate</th>
        <th>0.1</th>
        <th>—</th>
        <th>1/s</th>
        <th>[124,125]</th>
    </tr>
    <tr>
        <th></th>
        <th>k_CaN</th>
        <th>Calcineurin half activation concentration</th>
        <th>0.053</th>
        <th>—</th>
        <th>µM</th>
        <th>[113]</th>
    </tr>
    <tr>
        <th></th>
        <th>n_CaN</th>
        <th>Calcineurin Hill coefficient</th>
        <th>3</th>
        <th>—</th>
        <th></th>
        <th>[113]</th>
    </tr>
    <tr>
        <th>B, Parameters determined by constraints imposed by the model</th>
        <th>k⁰_PKA</th>
        <th>PKA base activity</th>
        <th>0.00359</th>
        <th>[0.0031–0.0041]</th>
        <th>1/s</th>
        <th>Constrains the location of the ‘‘LTD’’ and ‘‘LTP’’ windows in the steady-state scenario</th>
    </tr>
    <tr>
        <th></th>
        <th>k_PKA</th>
        <th>Maximum Ca²⁺/calcium-dependent PKA activity</th>
        <th>100</th>
        <th>[2–804]</th>
        <th>1/s</th>
        <th>Constrains the location of the ‘‘LTD’’ and ‘‘LTP’’ windows in the steady-state scenario</th>
    </tr>
    <tr>
        <th></th>
        <th>K_PKA</th>
        <th>PKA half activity concentration</th>
        <th>0.11</th>
        <th>[0.085–0.189]</th>
        <th>µM</th>
        <th>Constrains the location of the ‘‘LTD’’ and ‘‘LTP’’ windows in the steady-state scenario</th>
    </tr>
    <tr>
        <th></th>
        <th>n_PKA</th>
        <th>PKA Hill coefficient</th>
        <th>8</th>
        <th>[6.8–14.9]</th>
        <th></th>
        <th>Constrains the location of the ‘‘LTD’’ and ‘‘LTP’’ windows in the steady-state scenario</th>
    </tr>
    <tr>
        <th></th>
        <th>k_CaN</th>
        <th>Maximum Ca²⁺/CaM-dependent calcineurin activity</th>
        <th>18, 20</th>
        <th>[16.6–18.1]</th>
        <th>1/s</th>
        <th>Changes the PP1 level during stimulation</th>
    </tr>
    <tr>
        <th></th>
        <th>k₁₂</th>
        <th>Maximal dephosphorylation rate</th>
        <th>6,000</th>
        <th>See text</th>
        <th>1/s</th>
        <th>Changes velocity of the PP1 dynamics by scaling with R and Q</th>
    </tr>
    <tr>
        <th></th>
        <th>D₀</th>
        <th>Total PP1 concentration</th>
        <th>0.2</th>
        <th></th>
        <th>µM</th>
        <th>Changes velocity of the PP1 dynamics by scaling with R</th>
    </tr>
    <tr>
        <th></th>
        <th>k₆</th>
        <th>Probability of phosphorylation step shown in Figure 1C</th>
        <th>6</th>
        <th>See text</th>
        <th>1/s</th>
        <th>Changes the velocity of autophosphorylation by scaling with Q</th>
    </tr>
    <tr>
        <th></th>
        <th>k₇</th>
        <th>Probability of phosphorylation step shown in Figure 1D</th>
        <th>6</th>
        <th>See text</th>
        <th>1/s</th>
        <th>Changes the velocity of autophosphorylation by scaling with Q</th>
    </tr>
    <tr>
        <th></th>
        <th>k₈</th>
        <th>Probability of phosphorylation step shown in Figure 1E</th>
        <th>6</th>
        <th>See text</th>
        <th>1/s</th>
        <th>Changes the velocity of autophosphorylation by scaling with Q</th>
    </tr>
    <tr>
        <th>C, other parameters</th>
        <th>CaM₀</th>
        <th>Total calmodulin concentration</th>
        <th>0.1</th>
        <th>—</th>
        <th>µM</th>
        <th>See text.</th>
    </tr>
    <tr>
        <th></th>
        <th>K_M</th>
        <th>Michaelis constant of dephosphorylation</th>
        <th>0.4</th>
        <th>—</th>
        <th>µM</th>
        <th>As in [7]</th>
    </tr>
    <tr>
        <th></th>
        <th>k⁰_CaN</th>
        <th>Calcineurin base activity</th>
        <th>0.1</th>
        <th>—</th>
        <th>1/s</th>
        <th>Arbitrarily chosen, changing this value does not affect the steady-state results.</th>
    </tr>
    <tr>
        <th></th>
        <th>I₀</th>
        <th>Total I1 concentration</th>
        <th>1</th>
        <th>—</th>
        <th>µM</th>
        <th>Arbitrarily chosen, changing this value does not affect the steady-state results.</th>
    </tr>
  </tbody>
</table>

doi:10.1371/journal.pcbi.0030221.t001

Table 1B). Table 1B also shows the ranges of values of each parameter for which the above-described behavior is qualitatively observed. These ranges are obtained varying each parameter while keeping the remaining three constant. This shows that the system is relatively robust to parameter changes. It reacts most sensitively to changes in $k^0_{PKA}$, whose value can be varied by about 14% in both directions, while the other parameters can be varied over a range of about 100% and even ~800% for $k_{PKA}$. Note that the choice of the maximal calcineurin activity, which basically controls the height of the PP1 peak shown in Figure 3B, depends also on constraints discussed in the following section.

The system is also robust to changes in the total CaMKII concentration, as shown in Figure 3C where we compare the bifurcation diagrams for $CaMKII_0 = 16.67\ \mu M$ (blue line) and $8.33\ \mu M$ (green line) provided the PP1 activity is rescaled accordingly by using $k^0_{PKA} = 0.007\ 1/s$ in the latter case (other parameters remain unchanged). Note that both values of $CaMKII_0$ cover a range of CaMKII concentration that encompasses experimental estimates ($\sim 10\ \mu M$) for the PSD [24,27,47]. The dashed lines in Figure 3B show the position of the bistable region (dashed red lines) and the steady-state PP1 activity (dashed purple line) in the PP1 activity-calcium plane for $CaMKII_0 = 8.33\ \mu M$ and $k^0_{PKA} = 0.007\ 1/s$ (see Table 1A and 1B for other parameters).

To summarize the results so far, the model behavior is such that: (i) the calcium resting concentration is included in a region of bistability, giving rise to two stable steady-states—

PLoS Computational Biology | www.ploscompbiol.org | 2304 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

DOWN and UP—at resting conditions (marked by the diamond and the cross in Figure 3C, respectively); (ii) in a region of intermediate calcium concentrations, only a weakly phosphorylated steady-state exists (the LTD window, between filled circles marked **2** and **3**); (iii) conversely, at high calcium concentrations, only the highly phosphorylated steady-state is stable (the LTP window, beyond filled circle marked **4**). This scenario is now qualitatively consistent with experimental data. Note that, in contrast with Zhabotinsky (2000), our model does not require an unrealistic high phosphorylated inhibitor 1 concentration at resting calcium concentration to have a stable highly phosphorylated CaMKII state (compare green line in Figure 3B and [17] at this concentration).

### Dynamic Response of the Model to the STDP Protocol
Up to this point, we have investigated the steady-states of the CaMKII kinase-phosphatase system as a function of the intracellular calcium concentration. In experimental conditions, however, synaptic modifications are evoked by calcium transients resulting from experimental stimulation protocols inducing synaptic plasticity. Hence, the occurrence of transitions between weakly and highly phosphorylated states in the model needs to be examined in response to such calcium dynamics. Here, we explore in which conditions the spike-timing dependent plasticity (STDP) protocol as well as presynaptic or postsynaptic stimulation protocols alone induce such transitions.

For the STDP protocol, we use a standard repetitive stimulation protocol (60 pairs at 1 Hz, see experiments by Bi and Poo [2]). Each stimulation pair consists of a presynaptic spike at time $t_{pre}$ and a back-propagating postsynaptic action potential occurring at time $t_{post} = t_{pre} + \Delta t$. In experimental conditions, LTD is evoked for short negative $\Delta t$s, while LTP is evoked for short positive $\Delta t$s [1,2,51–53].

### STDP Protocol Stimulation with Deterministic Calcium Dynamics
Figure 4 shows the time course of calcium concentration transients evoked by one pair of a presynaptic spike and a back-propagating action potential (BPAP) at different $\Delta t$s. An isolated postsynaptic spike generates a calcium transient of amplitude $\Delta Ca_{post}$, due to opening of calcium channels induced by the depolarization caused by the BPAP. Likewise, an isolated presynaptic spike generates another calcium transient of amplitude $\Delta Ca_{pre}$, due to NMDA channel opening. Below, we will vary systematically the size of $\Delta Ca_{pre}$, keeping the ratio constant, $\Delta Ca_{post} / \Delta Ca_{pre} = 2$ [54]. See Materials and Methods for details of the model.

What happens when presynaptic and postsynaptic spikes are sufficiently close together so that their respective calcium transients overlap?

**For short negative time differences, $\Delta t < 0$ ms (post before pre).** The decaying phase of the fast BPAP-evoked calcium signal overlaps in time with the long-lasting calcium transient mediated by NMDA receptors (see Figure 4 for the $\Delta t = -10$ ms case). Though this temporal overlap has only a weak effect on the integral of the calcium transient induced by the pair of spikes compared to a case in which both transients do not interact at all (large positive and large negative $\Delta t$), the time spent by the system in different intervals of calcium concentration does change significantly with $\Delta t$. This feature largely contributes to the fact that LTD can potentially be observed at short negative $\Delta t$s only (see below).

**For positive time differences, $\Delta t > 0$ ms (pre before post).** The strong depolarization by the BPAP increases drastically the voltage-dependent NMDA-R mediated calcium current, leading to a supralinear superposition of the two contributions. The ratio between the calcium peak amplitude at $\Delta t = 10$ ms and the linear sum of individual presynaptically and postsynaptically evoked calcium transients is about 1.6, consistent with experimental data [55]. This supralinearity explains to a large extent the occurrence of LTP at short positive $\Delta t$ and also prevents LTD transitions at large positive $\Delta t$ protocols (see discussion below). The repetitive presentation (60 times at 1 Hz) of the presynaptic and postsynaptic spike pair produces repetitively the calcium transient shown in Figure 4 for a few examples of $\Delta t$.

### LTP and LTD-Like Transitions as a Function of $\Delta t$
When the parameters of the model are chosen accordingly, the model reproduces qualitatively the experimental results in response to the STDP stimulation protocol: (i) short positive $\Delta t$ stimulation protocols move CaMKII from the weakly phosphorylated state to the highly phosphorylated state. Starting from the UP state, no transition occurs. (ii) A system at rest at the UP state is switched to the DOWN state by short negative $\Delta t$ protocols, whereas the same protocol does not evoke transitions from the DOWN to the UP state. (iii) Large positive and negative $\Delta t$s do not evoke transitions between the DOWN and the UP states. We show in Figures 5–7 (red lines) the behavior of the model for parameters shown in Table 1A–1C, with $k_{CaN} = 18$ 1/s and $\Delta Ca_{pre} = 0.17$ $\mu$M.

Figure 5 shows the dynamics of the system for the whole stimulation protocol, and until the system has reached the final steady-state except for Figure 5A and 5B, which depicts the time course of the calcium concentration for one spike pair presentation only. Figure 5C and 5D shows the time

[The image shows a line graph of calcium concentration $Ca$ ($\mu$M) versus time (ms). Several curves are plotted for different time lags $\Delta t$: -100 ms (red), -10 ms (green), 0 ms (blue), +25 ms (orange), +100 ms (purple), and +200 ms (black). The baseline calcium concentration is 0.1 $\mu$M. The peaks for isolated presynaptic ($\Delta Ca_{pre} = 0.17$ $\mu$M) and postsynaptic ($\Delta Ca_{post} = 0.34$ $\mu$M) responses are indicated. The graph shows that for small positive $\Delta t$ (e.g., 0, +25 ms), the calcium peak is significantly higher than for other time lags.]

**Figure 4. Calcium Dynamics Evoked by a Pair of a Presynaptic and a Postsynaptic Spike Occurring at Different Time Lags, $\Delta t$**
Temporal evolution of the intracellular calcium concentration generated by the model in response to a presynaptic spike at $t_{pre} = 200$ ms and an additional postsynaptic spike at $t_{post} = t_{pre} + \Delta t$, where $\Delta t$ is indicated at the corresponding curve in ms. The calcium amplitudes of the isolated presynaptic ($\Delta Ca_{pre} = 0.17$ $\mu$M) and postsynaptic ($\Delta Ca_{post} = 0.34$ $\mu$M) responses are indicated in the panel. See Table 2 for parameters.
doi:10.1371/journal.pcbi.0030221.g004

PLoS Computational Biology | www.ploscompbiol.org | 2305 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

# LTP
## A, Calcium
The chart shows calcium concentration $Ca$ ($\mu$M) over time (ms) for two protocols:
- **+15 ms (red line):** Shows a sharp peak reaching approximately 0.85 $\mu$M around 220 ms, followed by a decay.
- **+100 ms (green line):** Shows a smaller initial rise, followed by a sharp peak reaching approximately 0.6 $\mu$M at 300 ms.

## C, PP1 activity
The chart shows PP1 activity $k_{12} \cdot D$ ($\mu$M/s) over time (min) with an inset showing time (s).
- **+15 ms (red line):** Activity rises and plateaus around 10 $\mu$M/s during stimulation, then decays.
- **+100 ms (green line):** Activity rises higher, plateauing around 15-16 $\mu$M/s during stimulation, then decays.
- The inset shows the step-like increases in activity during the first few seconds of stimulation.

## E, Phosphorylated CaMKII
The chart shows the concentration of phos. CaMKII subunits $S_{active}$ ($\mu$M) over time (min) with an inset showing time (s).
- **+15 ms (red line):** The system transitions from the DOWN state (~0 $\mu$M) to the UP state (~150 $\mu$M) after stimulation.
- **+100 ms (green line):** The system remains in the DOWN state (~0 $\mu$M) after a transient increase.
- The inset shows the initial accumulation of phosphorylated subunits during the first 3 seconds.

# LTD
## B, Calcium
The chart shows calcium concentration $Ca$ ($\mu$M) over time (ms) for two protocols:
- **-50 ms (blue line):** Shows a peak reaching approximately 0.45 $\mu$M at 150 ms, followed by a second lower peak.
- **-10 ms (orange line):** Shows a peak reaching approximately 0.45 $\mu$M at 200 ms.

## D, PP1 activity
The chart shows PP1 activity $k_{12} \cdot D$ ($\mu$M/s) over time (min) with an inset showing time (s).
- **-10 ms (orange line):** Activity rises and plateaus at a higher level (~17 $\mu$M/s) than the -50 ms protocol.
- **-50 ms (blue line):** Activity rises and plateaus around 15 $\mu$M/s.
- The inset shows the step-like increases in activity during the first few seconds of stimulation.

## F, Phosphorylated CaMKII
The chart shows the concentration of phos. CaMKII subunits $S_{active}$ ($\mu$M) over time (min) with an inset showing time (s).
- **-50 ms (blue line):** The system starts in the UP state (~150 $\mu$M), undergoes a transient decrease during stimulation, but returns to the UP state.
- **-10 ms (orange line):** The system starts in the UP state (~150 $\mu$M) and transitions to the DOWN state (~0 $\mu$M) after stimulation.
- The inset shows the small fluctuations in the UP state during the first 3 seconds of stimulation.

**Figure 5. Time Course of Calcium, PP1 Activity, and Phosphorylated CaMKII Subunit Concentration During STDP Stimulation Protocols with $\Delta t = -50$, $-10$, $15$, and $100$ ms**
(A,C,E) Show the dynamics during STDP stimulation protocols with $\Delta t = 15$ ms (red lines) and $\Delta t = 100$ ms (green lines), with the system initially in the DOWN state.
(B,D,F) The CaMKII system is initially in the UP state and is subjected to stimulation protocols with $\Delta t = -50$ ms (blue lines) and $\Delta t = -10$ ms (orange lines).
Note that the time course of calcium dynamics (A,B) is shown for one spike-pair presentation only. The PP1 and $S_{active}$ time course (C–F) is shown for the

PLoS Computational Biology | www.ploscompbiol.org 2306 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

full-stimulation protocol with 60 spike-pair presentations at 1 Hz and until the system converges to the final steady-state. The insets in panel (C–F) display the dynamics on a shorter time scale (first three spike pairs in the stimulation protocol). In this figure: $k_{CaN} = 18$ 1/s, $\Delta Ca_{pre} = 0.17$ $\mu$M. The PP1 steady-state activity is 7.21 $\mu$M/s at $Ca_0 = 0.1$ $\mu$M (compare Figure 3B).
doi:10.1371/journal.pcbi.0030221.g005

course of active PP1, while Figure 5E and 5F shows the dynamics of phosphorylated CaMKII subunit concentration. The left column shows the dynamics of the system when it is initially in the DOWN state (low concentration of phosphorylated CaMKII subunits) for two representative time lags ($\Delta t = 15$ ms and 100 ms), while the right column shows the dynamics when it is initially in the UP state, again for two representative time lags ($\Delta t = -50$ and $-10$ ms).

To understand why the system exhibits transitions in specific ranges of $\Delta t$, it is first crucial to examine how PP1 activation depends on $\Delta t$. For the value of $\Delta Ca_{pre}$ chosen here, PP1 activation is the largest at short negative time differences, since such values of $\Delta t$ maximize the time spent by the system in the range of calcium concentrations close to PP1 peak activation (see Figure 3B). On the other hand, PP1 activation is minimal for short positive time lags since Ca goes transiently to high concentrations and spends a short time at intermediate values. Let us now focus on the situation in which the system is initially in the DOWN state. During the stimulation protocol, two situations can arise. For short positive time differences (as, for example, the 15 ms case shown in Figure 5), the increase in PP1 activity is low and insufficient to counterbalance the large increase in the concentration of phosphorylated CaMKII subunits, since high calcium transients strongly favor the autophosphorylation process which outweighs the low dephosphorylation activity. Hence, the system reaches a high phosphorylation level during the stimulation protocol and converges gradually toward its equilibrium value in the UP state thereafter. On the other hand, for negative and large positive time lags, the increase in PP1 activity is large enough to counterbalance the calcium/calmodulin triggered autophosphorylation, i.e., CaMKII stays dephosphorylated and remains in the DOWN state (see for example the 100 ms case in Figure 5).

When the system is initially in the UP state, the concentration of phosphorylated CaMKII subunits again depends on the competition between dephosphorylation by PP1 and autophosphorylation progress during the protocol. Again, we have two possible outcomes of the protocol: either the PP1 concentration becomes large enough such that the system gets sufficiently dephosphorylated and moves in the basin of attraction of the DOWN state during the stimulation protocol (this occurs for example for the $-10$ ms case shown in Figure 5); or it is not large enough and autophosphorylation prevails, i.e., the system remains in the basin of attraction of the UP state. For the parameter set used in Figure 5, this happens for large negative and positive time lags.

Another way of visualizing the dynamics during and after the STDP protocol consists in plotting the trajectory of the system in the concentration of phosphorylated CaMKII subunits $S_{active}$–PP1 activity plane. This is done for several values of $\Delta t$ in Figure 6A and 6B. The DOWN and the UP stable steady-states of the CaMKII phosphorylation level are shown by the diamond and the cross, respectively (located at the intersections of the $S_{active}$ and $(k_{12} \cdot D)$ nullclines). In Figure 6A the system is initially in the DOWN state, whereas it is initially in the UP state in Figure 6B. In both Figure 6A and 6B, the end of the stimulation protocol corresponds to the point at which PP1 and $S_{active}$ stop to oscillate, and there is a sharp turn of the trajectories in the plane. In this plane, the separatrix (dotted black line) marks the boundary between the basins of attraction of both stable steady-states. Depending on the position of the system at the end of the stimulation protocol relative to this separatrix, the system relaxes either to the UP or the DOWN state. The separatrix is obtained by adjusting numerically $\Delta t$ to be at the boundary between the regions in which a transition to the UP (respectively, DOWN) state occurs or not.

The outcomes of the deterministic STDP protocols for $\Delta t$ values from $-100$ to 150 ms are summarized in Figure 7A (red line). We consider a large population of independent synapses submitted to the same protocol, in which initially half of the synapses are in the DOWN state and the other half in the UP state. Figure 7A shows the relative change in the fraction of synapses in the UP state as a function of $\Delta t$ ($+1$ means all synapses initially in DOWN where switched to UP; 0 means no change; $-1$ means all synapses in UP have switched to DOWN). There is a range of values of $\Delta t$ (from 10 to 16 ms) for which all synapses initially in the DOWN state switch to the UP state (LTP). LTD, or up-to-down transitions of the synapses initially in the UP state, is observed in a range of $\Delta t$ values from $-14$ to $-2$ ms (see red line in Figure 7A).

### STDP Protocol Stimulation with Stochastic Calcium Dynamics

The CaMKII kinase-phosphatase system in the PSD is composed of a few molecules only ($\sim 30$ CaMKII holoenzymes [56]), hence stochastic fluctuations potentially play an important role (see [57]). The CaMKII system is also exposed to fluctuating calcium transients stemming from stochastic neurotransmitter release, stochastic channel opening, and the stochastic nature of neurotransmitter as well as calcium diffusion [54,58]. It is therefore necessary to investigate the dynamic behavior of the CaMKII system in the presence of noise. Here we choose for simplicity to introduce fluctuations in calcium transients exclusively.

Two sources of noise are introduced in the calcium dynamics simulations: (i) the NMDA receptor maximum conductance is drawn at random at the occurrence of each presynaptic spike, and (ii) the maximum conductance of the voltage-dependent calcium channel is drawn at random at the occurrence of each postsynaptic spike. Both conductances are drawn from binomial distributions similar to those measured in experiments [58,59] (see Materials and Methods for more details). Some examples of noisy calcium transients are shown in the inset of Figure 7A (dashed lines; the full line depicts the average transient).

Again, we consider a large population of independent synapses exposed to stochastic stimulation protocols. $N = 300$ independent synapses are simulated, 150 initially in the DOWN and 150 in the UP state. Applying the stimulation protocol leads to stochastic transitions between UP and DOWN states. Figure 7A shows the relative change in the

PLoS Computational Biology | www.ploscompbiol.org 2307 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

conductances results in a variability of calcium transients around the mean transients. The consequence of this variability in calcium transients is that, while the shape of the PP1 level versus $\Delta t$ is qualitatively unchanged, the PP1 level reached during stimulation protocols is significantly reduced. This is due to the fact that the variability in calcium transients decreases the time spent by the system at calcium concentrations which maximize PP1 buildup. Hence, the probability that the PP1 level is high enough to make the system switch to the DOWN state becomes small for $k_{CaN} = 18$ 1/s (the value used in deterministic simulations). Consequently, the up-to-down transition probability at short negative time lags is low and LTD is effectively absent in this case (see green line with squares in Figure 7A). However, the LTD probability becomes larger as $k_{CaN}$ is increased. Figure 7A shows an up-to-down switching probability of about 0.93 for short negative time lags with $k_{CaN} = 20$ 1/s (at $\Delta t = -10$ ms). It also shows that for this value of $k_{CaN}$ there exists a small but finite probability of eliciting LTD transitions for large positive and large negative time lags, due to variability in calcium transients. The range of values of $k_{CaN}$ for which the LTD probability for short negative $\Delta t$ is larger than 0.5 AND the LTD probability for large positive $\Delta t$ is smaller than 0.5 is 19 1/s $< k_{CaN} < 20$ 1/s. To summarize, down-to-up transitions occur robustly for a large range of parameters at short positive time lags. On the contrary, the range of short negative values of $\Delta t$ for which UP to DOWN switches are observed is less robust to noise (see Discussion).

### Effect of Phosphatase Inhibitors

The role of protein phosphatases in synaptic plasticity has been investigated through the application of phosphatase inhibitors during the presentation of stimulation protocols inducing synaptic changes [21,37,43,60]. These experiments have shown that phosphatase inhibitors prevent LTD while sparing LTP. We investigate the effect of phosphatase inhibitors in our model by gradually reducing the dephosphorylation activity of PP1 and study the changes in the steady-states of the phosphorylated CaMKII subunit concentration and in the transition behavior.

Since the steady-state PP1 concentration is given by $D_{steady-state} = D_0 / (1 + (I_0 k_{13} v_{PKA}) / (k_{-13} v_{CaN}))$ (see Materials and Methods), scaling down $D_0$ corresponds to a reduction of the steady-state PP1 activity given by the purple line in Figure 3B. Consequently, the intersections between the boundaries of the bistable region (given by the red lines in Figure 3B) and the PP1 activity change. In other words, the locations and ranges of the LTD and the LTP windows change as a function of the level of PP1 inhibition. Scaling down the total PP1 concentration leads to a diminution of the size of the LTD window and to the emergence of a second LTP window at low calcium concentrations (see the 80% case shown by the green line in Figure 8A and 8B). Decreasing further protein phosphatase strength makes the LTD window disappear and a large LTP window emerges starting at low calcium concentrations (see 60% and 40% cases in Figure 8A and 8B). Finally, reducing the PP1 concentration below $\sim 40\%$ results in a loss of the stability of the DOWN state at resting calcium concentrations, leaving the UP state as the only stable steady-state for all calcium concentrations.

Figure 8C shows how LTP/LTD transitions are affected by reduced total PP1 concentration, when the model is exposed

[The image contains two phase-plane plots, A and B, showing the dynamics of the system in the PP1 activity ($k_{12} \cdot D$) vs. phosphorylated CaMKII subunit concentration ($S_{active}$) space.]

**Figure 6. Dynamics of the System in the PP1 Activity–Phosphorylated CaMKII Subunit Concentration Phase-Plane**
(A,B) Show the trajectories in the $(k_{12} \cdot D) - S_{active}$ phase-plane during STDP stimulation protocols with $\Delta t = -50$ ms (red), $\Delta t = -10$ ms (green), $\Delta t = 15$ ms (blue), and $\Delta t = 100$ ms (cyan), and until the system reaches the final steady-state.
(A) System initially in the DOWN state (diamond). (B) System initially in the UP state (cross). The black arrows indicate the direction of motion of the system along the trajectory for some examples. The full and the dashed black line depict the $S_{active}$ and the $(k_{12} \cdot D)$ nullclines at resting conditions ($Ca_0 = 0.1 \mu M$), respectively. Therefore, the intersections of both nullclines mark the positions of the steady-states of the system: two stable—the DOWN (diamond) and the UP (cross) state—and one unstable at $S_{active} \approx 56.8 \mu M$. Note that the PP1 activity nullcline (dashed black line) is independent of $S_{active}$ (see Equations 24 and 25), whereas the $S_{active}$ nullcline (full black line) is dependent on $(k_{12} \cdot D)$ and $S_{active}$ (see Equations 7–20). The separatrix, separating the basins of attraction of the two stable steady-states is shown as a dotted black line (see Table 1 for parameters, $k_{CaN} = 18$ 1/s, $\Delta Ca_{pre} = 0.17 \mu M$).
doi:10.1371/journal.pcbi.0030221.g006

fraction of synapses in the UP state as a function of $\Delta t$, for $k_{CaN} = 18$ 1/s and 20 1/s. For example, a relative change of $-0.8$ for $\Delta t = -15$ ms ($k_{CaN} = 20$ 1/s case) means that 120 of the synapses in the UP state (out of the 150) switched to the DOWN state in response to this protocol, while none of the 150 synapses in the DOWN state experienced a down-to-up transition during the $\Delta t = -15$ ms stimulation.

The variability in maximum NMDA and CaL current

PLoS Computational Biology | www.ploscompbiol.org 2308 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

PP1 concentration to 60% and more results in up-to-down transitions for all time differences $\Delta t$ (see red line in Figure 8C).

### Presynaptic or Postsynaptic Stimulation at Different Frequencies

Experiments on STDP show that presynaptic or postsynaptic spikes alone at the same stimulation frequency (1 Hz) do not evoke any plasticity (see for example [55]). To check the behavior of the model in this situation, we expose the CaMKII system to either 60 presynaptic or postsynaptic spikes of different frequencies and show the transitions results for the deterministic calcium transient case in Figure 7B by red lines. 60 presynaptic spikes alone do not evoke any transitions at low frequencies (1–3 Hz). For presynaptic stimulations in the range 4–18 Hz, up-to-down transitions occur, and for frequencies equal and larger to 19 Hz the CaMKII system is switched from the DOWN to the UP state (see full red line in Figure 7B). The transition outcomes change dramatically if stimulation occurs exclusively with postsynaptic spikes. 60 postsynaptic spikes do not evoke transitions up to a stimulation frequency of 84 Hz (see dashed red line in Figure 7B). Above 85 Hz, transitions from DOWN to UP occur. Note that the spike pairs during the STDP spike-pair stimulation protocol employed above are presented at a frequency of 1 Hz only, i.e., presynaptic or postsynaptic spikes alone at this frequency do not evoke transitions, consistent with experiments [55]. Note also that the model does not incorporate frequency-dependent attenuation of EPSPs and BPAPs. Attenuation of BPAPs at high frequencies could prohibit down-to-up transitions in the post protocol at any frequency.

We also expose the CaMKII system to fluctuating calcium transients evoked by presynaptic or postsynaptic frequency stimulations. The implementation of calcium transient noise is exactly as for STDP spike-pair protocols above. The average relative changes in the fraction of synapses in the UP state for these stimulations are shown for varying frequencies in Figure 7B for $k_{CaN} = 18\text{ 1/s}$ and $20\text{ 1/s}$ ($N = 300$ synapses). Presynaptic stimulations at frequencies between $\sim 2$ and $\sim 16$ Hz evoke a net increase of synapses in the DOWN state, while stimulation above $\sim 16$ Hz lead to LTP transitions. Again, no up-to-down transitions are observed with postsynaptic stimulation alone, while stimulation frequencies above $\sim 50$ Hz yield a net increase of the number of synapses in the UP state.

### Presynaptic or Postsynaptic Spike-Pair Stimulation at Different Time Differences

Another simple generalization of the STDP protocol consists in exposing the system to purely presynaptic spike pairs, or purely postsynaptic spike pairs. Spike pairs with a fixed inter-spike interval $\Delta t$ are presented 60 times at varying frequencies. We investigate the transition behavior of the model for varying inter-spike intervals and for different presentation frequencies. This is a protocol for which plasticity outcomes have, to our knowledge, not yet been characterized.

Presynaptic spike pairs lead to up-to-down transitions for all values of $\Delta t$ at a frequency of $f = 1\text{ Hz}$, consistent with the fact that presynaptic stimulation of single spikes at 2 Hz evokes such transitions (see Figure 7B). On the other hand,

[The image contains two line graphs, A and B, showing the relative change in the fraction of synapses in the UP state.]

**Figure 7. Synaptic Modifications in Response to STDP, Purely Presynaptic, or Purely Postsynaptic Stimulation Protocols**
(A) The relative change in the fraction of synapses in the UP state in response to deterministic (red line) and stochastic (green and blue lines with symbols) STDP stimulation protocols is shown as a function of $\Delta t$. 0 means no net change in number of synapses in the UP state, positive relative change means a net increase of synapses in the UP state, and a negative relative change means a net increase of synapses in the DOWN state. Stochastic stimulation results are shown for $k_{CaN} = 18\text{ 1/s}$ (green line with squares, same value as in deterministic case) and $k_{CaN} = 20\text{ 1/s}$ (blue line with circles). The inset shows some example calcium transients (dashed lines) of the stochastic stimulation protocol evoked by a spike pair with $\Delta t = -100\text{ ms}$ and $t_{pre} = 200\text{ ms}$. The average calcium transient is shown by the full line and is the same as the red curve in Figure 4. ($\Delta Ca_{pre} = 0.17\text{ }\mu\text{M}$.)
(B) Relative change in the fraction of synapses in the UP state after purely presynaptic, or purely postsynaptic stimulation protocols, as a function of frequency: the stimulation protocol consists of 60 presynaptic (full lines) or postsynaptic spikes (dashed lines) at a given frequency. Red lines: deterministic stimulation, $k_{CaN} = 18\text{ 1/s}$. Green lines with diamonds or squares: stochastic presynaptic or postsynaptic stimulations, respectively, with $k_{CaN} = 18\text{ 1/s}$. Blue line with triangles or circles: stochastic presynaptic or postsynaptic stimulations, respectively, with $k_{CaN} = 20\text{ 1/s}$. In all panels, the stochastic stimulation results are averaged over $N = 300$ synapses.
doi:10.1371/journal.pcbi.0030221.g007

to the STDP stimulation protocol with noisy calcium transients. Consistent with experiments, reducing the PP1 concentration by 20% leads to a loss of LTD transitions (see green line in Figure 8C), while increasing the range of $\Delta t$ for which LTP transitions are observed. Further reduction of the

PLoS Computational Biology | www.ploscompbiol.org | 2309 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

**A**
[The image shows a line graph of PP1 activity $k_{12}D$ ($\mu$M/s) versus calcium concentration $Ca$ ($\mu$M). Multiple curves represent different levels of PP1 inhibition: 100% (blue), 80% (green), 60% (red), 40% (orange), and 20% (purple). A shaded region indicates the "bistable range". A vertical black line marks the resting calcium concentration at 0.1 $\mu$M.]

**B**
[The image shows a graph of the relative concentration of phosphorylated CaMKII subunits $S_{active}$ versus calcium concentration $Ca$ ($\mu$M). Solid lines represent stable steady-states and dashed lines represent unstable steady-states for different PP1 inhibition levels (100%, 80%, 60%, 40%). The UP and DOWN states are labeled at the resting calcium concentration. $CaMKII_0 = 16.67$ $\mu$M.]

**C**
[The image shows a graph of the relative change in fraction of synapses in UP state versus time difference $\Delta t$ (ms). Three data series are shown: 100% (blue circles), 80% (green squares), and 60% (red diamonds). The blue curve shows a characteristic STDP shape with a dip (LTD) and a peak (LTP).]

**Figure 8. Phosphorylated CaMKII Subunit Concentration Steady-States and Transition Outcomes in the Presence of PP1 Inhibitors**
(A) PP1 activity as a function of calcium concentration, for different levels of inhibition (indicated close to the corresponding curves). For example, in the 80% case, the total PP1 concentration is $0.8 D_0$, i.e., 0.16 $\mu$M. The 100% curve corresponds to no PP1 inhibition (same curve as in Figure 3B). Vertical black line: calcium resting concentration, $Ca_0 = 0.1$ $\mu$M.
(B) Steady-states of the phosphorylated CaMKII subunit concentration versus Calcium for different levels of PP1 inhibition (same colors as in (A)). Full lines: stable steady-states. Dashed lines: unstable steady-states. The positions of the UP and the DOWN state at calcium resting conditions are shown for the 100% PP1 concentration case by the cross and the diamond, respectively. $CaMKII_0 = 16.67$ $\mu$M, $k_{CaN} = 18$ 1/s, see Table 1 for other parameters.
(C) Transition results in response to the STDP stimulation protocol evoking noisy calcium transients in the presence of PP1 inhibitors. The average relative changes in the fraction of synapses in the UP state for three different total PP1 concentrations is shown (same colors as in (A) and (B)). The blue line is the same as the blue line in Figure 7A for $D_0 = 0.2$ $\mu$M as given in Table 1, i.e., the 100% case ($N = 300$, see Table 1 for other parameters).
doi:10.1371/journal.pcbi.0030221.g008

purely postsynaptic spike pairs evoke down-to-up transitions in a very narrow range of values of $\Delta t$ (from 3 to 8 ms) at $f = 1$ Hz. In other words, postsynaptic spike pairs have to be presented sufficiently closely in time for the phosphorylation changes to sum up, so that the system converges to the UP state. Decreasing the spike-pair presentation frequency $f$ leads to transitions in narrower ranges of $\Delta t$ for the presynaptic and the postsynaptic protocol (e.g., $f = 0.5$ Hz; presynaptic spike pairs with $0 < \Delta t \lesssim 300$ ms lead to up-to-down transitions, postsynaptic spike pairs with $3 \lesssim \Delta t \lesssim 6$ ms evoke down-to-up transitions). At $f = 0.1$ Hz, there are no longer any transitions in the purely presynaptic stimulation protocol and only a small down-to-up transition probability exists for postsynaptic spike pairs ($3 \lesssim \Delta t \lesssim 4$ ms; unpublished data).

The difference in transition outcomes between presynaptic and postsynaptic spike-pair stimulations can be understood by inspecting the calcium transients evoked by both stimulation protocols. The maximum calcium amplitude reached by pairs of postsynaptic spikes is much larger than the calcium amplitude evoked by presynaptic spikes (maximum amplitude for $\Delta t = 10$ ms is $\sim 0.45$ $\mu$M for presynaptic spike pairs and $\sim 0.7$ $\mu$M for postsynaptic spike pairs with the parameters given in Table 2). On the other hand, pairs of presynaptic spikes evoke calcium transients which last much longer than postsynaptic pairs of spikes (compare the different time scales in Figure 4). The high calcium transients evoked by postsynaptic spike pairs strongly activate the cAMP–PKA pathway and therefore suppress PP1 activity. This suppression, together with the strong CaMKII autophosphorylation due to high calcium concentrations, leads to down-to-up transitions in response to closely spaced postsynaptic spike pairs. Even single postsynaptically evoked calcium transients reach calcium levels sufficiently high to activate the cAMP–PKA pathway. This explains why purely postsynaptic stimulation at varying frequencies does not go through an LTD range (see Figure 7B and compare the small PP1 buildup in the inset in Figure 5D in response to the postsynaptically evoked calcium transient in the $\Delta t = -50$ ms protocol). In contrast, the long-lasting calcium transients evoked by purely presynaptic spike pairs make the system spend a lot of time in calcium ranges maximizing PP1 buildup. This leads to strong dephosphorylation of CaMKII by PP1 which cannot be counterbalanced by moderate autophosphorylation evoked by intermediate calcium levels. Hence up-to-down transitions are evoked for closely spaced presynaptic spike pairs. Again, this explains also why ongoing presynaptic stimulation at different frequencies evokes LTD at low presentation frequencies before the calcium transients are adding up sufficiently to activate the cAMP–PKA pathway and evoke strong autophosphorylation (this happens above $f \approx 16$ Hz in Figure 7B).

PLoS Computational Biology | www.ploscompbiol.org 2310 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

## Effect of Parameter Changes on the Behavior of the Model

The model with the parameter set discussed until this point reproduces qualitatively experimentally observed transition outcomes of the STDP protocol. We now discuss how changing parameters affect the transition behavior. Numerical investigations of the model show that two characteristics of the CaMKII system dynamics are crucial: (i) how the level of PP1 activity at the end of the stimulation protocol depends on $\Delta t$; (ii) the time course of autophosphorylation and dephosphorylation of CaMKII, and of PP1 buildup, influences the number of spike pairs during the stimulation protocol necessary to evoke down-to-up- or up-to-down transitions. We focus here for the sake of simplicity on deterministic calcium transients.

### Effect of the Amplitude of Calcium Transients and of Calcineurin Activity on the PP1 Activity Level

We have shown above that a peak in steady-state PP1 activity at moderate calcium concentrations occurs if the cAMP–PKA pathway activates at higher calcium concentrations than the calcineurin pathway (see Figure 3A). Here, we address the question of how the dynamics of the PP1 activity during the stimulation protocol changes as a function of the balance between the activation of both pathways. Changing the NMDA-R mediated calcium amplitude $\Delta Ca_{pre}$ and the BPAP evoked calcium response, keeping their ratio constant ($\Delta Ca_{post} = 2 \cdot \Delta Ca_{pre}$), and also keeping the parameters of the protein signaling cascade constant, allows us to change the balance between the activation of both pathways and to get an insight into what controls the dependence of the PP1 activity level on $\Delta t$. Figure 9A and 9C show the PP1 activity level immediately after the presentation of one and 60 spike pairs, respectively, as a function of $\Delta t$, for different values of $\Delta Ca_{pre}$. Note that the dependence of PP1 activity with respect to $\Delta t$ after the presentation of one spike pair (Figure 9A) is qualitatively preserved after the entire stimulation protocol of 60 spike-pair presentations (Figure 9C).

Figure 9B represents the change in PP1 activity induced by a single spike pair, computed from Equation 34, as well as contributions of the PKA and calcineurin pathways to this change. The dashed lines in Figure 9B show the contribution of the cAMP–PKA pathway to the change in PP1 activity (second term in the integral of Equation 34). This contribution is negative, since this pathway decreases PP1 activity. Due to the high half activation calcium concentration of $v_{PKA}(C)$ (see blue line in Figure 3A), the cAMP–PKA pathway is sensitive to high calcium elevations only. Hence, the negative contribution of this pathway increases drastically when the calcium amplitude $\Delta Ca_{pre}$ increases, since the calcium transients spend more time in the range of cAMP–PKA activation. In response to the supralinear superposition of the NMDA-R and the BPAP evoked currents at short positive time differences, this pathway ensures a low level of PP1 activity in this range.

The dotted lines in Figure 9B show the contribution of the calcineurin pathway to the change in PP1 activity. This contribution is positive, since this pathway increases PP1 activity. The calcineurin pathway activates at lower calcium concentrations than the PKA pathway (see red line in Figure 3A; integral of the first part of Equation 34), and therefore this pathway is sensitive to the time spent by the system at intermediate and high calcium levels. This calcineurin contribution starts to increase at negative time differences (when calcium transients induced by pre- and post-synaptic spikes start to interact), reaches a peak close to $\Delta t = 0$, and then decreases slowly with $\Delta t$.

The sum of the two contributions yields the net change in PP1 activity (full lines of Figure 9B). For $\Delta Ca_{pre} = 0.17\ \mu\text{M}$, the value chosen in the rest of the paper, the PP1 change versus $\Delta t$ curve shows first a peak at negative $\Delta t$ (due to increase in calcineurin activity in this range), followed by a trough at positive $\Delta t$ (due to the strong increase in PKA activity in this range). There is a secondary peak of PP1 change at larger values of $\Delta t$ ($\sim 100\ \text{ms}$) because calcineurin activity decays more slowly with $\Delta t$ than PKA activity. However, this peak is smaller than the peak at negative $\Delta t$, which explains why LTD is observed at short negative $\Delta t$ but not large positive ones.

Changing the size of the calcium transients potentially changes qualitatively the shape of this curve because it affects the time spent by the system in different calcium concentration ranges. For example, decreasing the size of the calcium transients weakens considerably the PKA pathway, leading to an increase in PP1 activity for negative as well as positive values of $\Delta t$. On the other hand, increasing the calcium transients leads to a strengthening of the PKA pathway relative to the calcineurin pathway, leading to a much smaller peak in the PP1 change curve at short negative $\Delta t$. This peak eventually vanishes for large enough $\Delta Ca_{pre} \geq 0.4\ \mu\text{M}$ (unpublished data).

Since transitions are a result of an unbalance between autophosphorylation and dephosphorylation mediated by PP1, the $\Delta t$ range for which transitions are evoked or prevented can therefore be controlled by means of the calcium amplitude. If the calcium amplitude is decreased in the model, no transitions are observed any more (e.g., for $\Delta Ca_{pre} = 0.15\ \mu\text{M}$). On the other hand, increasing the calcium amplitude extends the $\Delta t$ range for which up-to-down and down-to-up transitions are evoked ($\Delta Ca_{pre} = 0.18\ \mu\text{M}$, LTD range: $[-21 \dots -3]\ \text{ms}$ and LTP range: $[3 \dots 33]\ \text{ms}$; unpublished data). These predictions could be checked experimentally by changing the external calcium concentration and therefore changing the calcium influx evoked by presynaptic and postsynaptic spikes.

To summarize, there exists a range of $\Delta Ca_{pre}$ for which the PP1 level at the end of the stimulation protocol as a function of $\Delta t$ exhibits a maximum for short negative $\Delta t$s and is low enough to be outweighed by autophosphorylation for short positive $\Delta t$s. This is a requirement for a system to exhibit LTD-like transitions at short negative time intervals only, and LTP-like transitions at short positive time intervals only. However, these qualitative features of the PP1 activation versus $\Delta t$ curve are not sufficient to ensure that STDP protocol stimulations with short negative time lags lead to transitions from the UP to the DOWN state only. In addition, (i) the absolute level of PP1 activity for short negative $\Delta t$ stimulations must be high enough to evoke up-to-down transitions; (ii) at the same time, the total PP1 level has to be low enough such that for large negative and large positive time lag stimulations the system remains in the UP state and that for short positive $\Delta t$ protocols autophosphorylation prevails over dephosphorylation leading to down-to-up transitions. These two criteria can be met by changing the maximal calcium/calmodulin-dependent calcineurin activity $k_{\text{CaN}}$, which changes the amplitude of the peak of the PP1 vs

PLoS Computational Biology | www.ploscompbiol.org 2311 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

**A**
[The image shows a line graph titled "PP1 activity after a single pair $k_{12} \cdot D$ ($\mu$M/s)" vs "time difference $\Delta t$ (ms)". There are three curves:
- Red curve: $Ca = 0.3 \mu$M. It shows a sharp dip at $\Delta t = 0$ and a slow recovery.
- Green curve: $Ca = 0.17 \mu$M. It shows a smaller dip and recovery.
- Blue curve: $Ca = 0.1 \mu$M. It shows a very slight dip and recovery.
Dashed lines represent numerical integration of Equations 24 and 25, while full lines represent the approximate solution.]

**B**
[The image shows a line graph titled "contributions to PP1 changes $k_{12} \delta D$ ($\mu$M/s)" vs "time difference $\Delta t$ (ms)".
- Legend:
  - Solid line: change in PP1 activity
  - Dotted line: calcineurin pathway
  - Dashed line: cAMP-PKA pathway
The graph shows these contributions for three different calcium amplitudes (Red, Green, Blue) as in panel A.]

**C**
[The image shows a line graph titled "PP1 activity after 60 pairs $k_{12} \cdot D$ ($\mu$M/s)" vs "time difference $\Delta t$ (ms)".
- Red curve: $Ca = 0.3 \mu$M.
- Green curve: $Ca = 0.17 \mu$M.
- Blue curve: $Ca = 0.1 \mu$M.
The graph shows the cumulative PP1 activity after 60 spike pairs, showing more pronounced changes compared to panel A.]

**Figure 9. PP1 Activity Level After the Presentation of One or 60 Spike Pairs as a Function of $\Delta t$ and the Amplitude of Calcium Transients**
(A) PP1 level after the presentation of one spike pair as a function of $\Delta t$. Dashed lines: numerical integration of Equations 24 and 25. Full lines: approximate solution ($k_{12} (D^* + \delta D(t))$) from numerical integration of Equation 34. The three different colors correspond to three different calcium amplitudes $\Delta Ca_{pre}$ as marked in the panel.
(B) The increase or decrease of PP1 activity induced by a single spike pair (full lines) as well as the contributions from the calcineurin pathway (dotted lines) and the cAMP–PKA pathway (dashed lines) to this change in PP1 activity as a function of $\Delta t$: the three colors correspond to the same three different calcium amplitudes $\Delta Ca_{pre}$ as marked in (A). The dotted and dashed lines are obtained from numerical integration of the first and the second terms in Equation 34, respectively. The full lines depict the sum of both contributions and represent $k_{12} \cdot \delta D(t)$ after the presentation of the first pair of spikes (see Materials and Methods).
(C) Total level of PP1 activity after the *whole* stimulation protocol from numerical integration of Equations 24 and 25, as a function of $\Delta t$: The different colors correspond again to the three different calcium amplitudes as in (A) and (B) (see Table 1 for parameters, $k_{CaN} = 18$ 1/s).
doi:10.1371/journal.pcbi.0030221.g009

$Ca^{2+}$ steady-state curve at moderate calcium concentrations (purple lines in Figure 3B). Consequently, this parameter allows us to control the PP1 level attained during the stimulation protocol for all $\Delta t$s. In particular, the range $16.6 \le k_{CaN} \le 18.1$ 1/s fulfills the two requirements above (Figures 5 and 6 and the red as well as the green lines in Figure 7 use $k_{CaN} = 18$ 1/s).

### Effect of Kinetics of Autophosphorylation and Dephosphorylation on the Number of Spike-Pair Presentations Needed for Transitions
The autophosphorylation rates $k_6$, $k_7$, $k_8$, the maximal dephosphorylation rate $k_{12}$, and the total PP1 concentration $D_0$ determine the velocity of autophosphorylation as well as dephosphorylation dynamics of CaMKII and the dynamics of the PP1 response during exposure to the STDP protocol. We introduce scaling parameters $R$ and $Q$ such that varying $R$ and $Q$ does not change the steady-state behavior of the CaMKII system (see Figure 3B and 3C) nor the maximum PP1 activity reached during the stimulation but only the dynamics of the system. Both scaling parameters are varied extensively in order to investigate their impact on the transition behavior of the model, i.e., $0.002 \le R \le 2$ and $0.083 \le Q \le 1.67$.

$R$ is chosen such as to control the dephosphorylation kinetics, while leaving the PP1 activity, given by the product ($k_{12} \cdot D$), constant. This leaves the steady-state behavior intact since it depends on this product only. Hence, in the following simulations, $k_{12}$ and $D_0$ are replaced by $k'_{12} = k_{12} \cdot R$ and $D'_0 = D_0/R$, where $k_{12}$ and $D_0$ are the "control" parameters listed in Table 1B. $R$ controls how fast the dephosphorylation dynamics responds to calcium transients, since the PP1 buildup during the presentation of the stimulation protocol and the decay dynamics thereafter depend on the value of $D$ but not on $k_{12}$ (see Equation 31 in Materials and Methods). Figure 10A shows the PP1 activity time course for three different values of $R$ during and after the STDP stimulation protocol with $\Delta t = 15$ ms, i.e., for $R = 0.002, 0.078$, and $1$.

$Q$ scales the autophosphorylation rates $k_6$, $k_7$, and $k_8$ together with the maximal dephosphorylation rate $k_{12}$ as $k'_x = k_x \cdot Q$ (with $x = 6, 7, 8$, and $12$), where all the rates $k_x$ take the values listed in Table 1B. This corresponds to a rescaling of the $y$-axis in Figure 3B, i.e., the points of intersection between the bistable range (red shaded areas) and the PP1 activity (purple lines) are kept fixed and therefore the steady-state concentration of phosphorylated CaMKII subunits (Figure 3C) is left unchanged. We illustrate the impact of changes in $Q$ on the dynamics of $S_{active}$ in Figure 10B for three different $Q$s and $R = 1$ as well as three different $R$s and $Q = 1$. Since the temporal evolution of $S_{active}$ is a result of the competing autophosphorylation and dephosphorylation progress, the choice of both scaling parameters influences $S_{active}$ dynamics (see Figure 10B). Note that $R = 1$ and $Q = 1$ is used

PLoS Computational Biology | www.ploscompbiol.org | 2312 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

# A, PP1 activity
[The image shows a graph of PP1 activity $k_{12} \cdot D$ ($\mu$M/s) versus time (min). Three curves are shown for different values of R:
- R=1 (red line): Rapidly rises to a steady-state oscillation around 10.2 $\mu$M/s within 0.5 min, then decays after 1 min.
- R=0.078 (green line): Rises more slowly, reaching the same steady-state oscillation by 1 min, then decays.
- R=0.002 (blue line): Rises linearly throughout the 1 min stimulation period, reaching only about 8.5 $\mu$M/s, and decays very slowly.]

# B, Phosphorylated CaMKII
[The image shows a graph of the concentration of phosphorylated CaMKII subunits $S_{active}$ ($\mu$M) versus time (min). Five curves are shown for different R and Q values:
- R=1, Q=1.5 (magenta): Rapid rise to ~170 $\mu$M, then stabilizes around 150 $\mu$M.
- R=0.078, Q=1 (green): Slower rise, reaching ~160 $\mu$M, then stabilizes around 150 $\mu$M.
- R=0.002, Q=1 (orange): Slowest rise, reaching ~140 $\mu$M, then stabilizes around 140 $\mu$M.
- R=1, Q=1 (red): Rapid rise to ~140 $\mu$M, then stabilizes around 140 $\mu$M.
- R=1, Q=0.083 (blue): Remains near 0 $\mu$M throughout.]

# C, varying R; Q=1
[The image shows a plot of time difference $\Delta t$ (ms) versus the number of spike-pair presentations. It maps regions of synaptic state transitions:
- UP-to-DOWN, -1: Large red shaded region for $\Delta t$ between ~25ms and 200ms, and a smaller blue/red/green shaded region for negative $\Delta t$ between ~-10ms and -40ms.
- DOWN-to-UP, +1: Small diagonally striped regions for $\Delta t$ between ~0ms and 25ms.
- no change, 0: White regions.
- Specific R values are labeled: R = 0.078 (red), R = 1 (green), R = 2 (blue).]

# D, varying Q; R=1
[The image shows a similar plot to C, but varying Q:
- UP-to-DOWN, -1: Large red shaded region for positive $\Delta t$ and a blue/green/red shaded region for negative $\Delta t$.
- DOWN-to-UP, +1: Diagonally striped regions for small positive $\Delta t$.
- Specific Q values are labeled: Q = 1.67 (red), Q = 1 (green), Q = 0.83 (blue).]

**Figure 10. Impact of the Kinetics of PP1 Activity and CaMKII Phosphorylation on the Number of Spike-Pair Presentations Leading to Transitions**
(A,B) Time course of PP1 activity ($k_{12} \cdot D$) (A) and phosphorylated CaMKII subunit concentration (B) during and after the STDP stimulation protocol for different values of $R$ and $Q$ (deterministic stimulation protocol, 60 spike pairs with $\Delta t = 15$ ms). The buildup of PP1 activity ($k_{12} \cdot D$) is shown in (A) for $R = 0.002$ (blue line), $R = 0.078$ (green line) and $R = 1$ (red line; same curve as in Figure 5C). $Q$ has no impact on PP1 dynamics. The time course of phosphorylated CaMKII subunit concentration is depicted in (B) for five sets of values of $R$ and $Q$ (indicated close to the corresponding curves).
(C,D) The impact of the number of spike-pair presentations and of $R$ and $Q$ on the $\Delta t$ ranges for which transitions occur is depicted for STDP stimulation protocols evoking deterministic calcium transients. White region: no change (relative change in fraction of synapses in the UP state is 0); diagonally striped regions: down-to-up transitions (relative change in fraction of synapses in the UP state is +1); shaded regions: up-to-down transitions (relative change in fraction of synapses in the UP state is -1). In each case, down-to-up and up-to-down transition regions in the same color correspond to the same choice of $R$ (C) or $Q$ (D). All the cases in (C) use $Q = 1$ paired with: $R = 0.078$: red regions; $R = 1$: green regions; and $R = 2$: blue region. In (D): blue regions: $Q = 0.83$ (no down-to-up transitions); green regions: $Q = 1$; and red regions: $Q = 1.67$. $R = 1$ is utilized in all cases in (D), i.e., the green regions in (C) and (D) are identical (see Table 1 for parameters, $k_{CaN} = 18$ 1/s, $\Delta Ca_{pre} = 0.17$ $\mu$M).
doi:10.1371/journal.pcbi.0030221.g010

everywhere in this paper, except for the results discussed in this section and shown in Figure 10.

Increasing $R$ accelerates the convergence of PP1 toward a steady-state oscillation. In Figure 10A, this happens after $\sim 20$ and $\sim 30$ spike-pair presentations for $R = 1$ and $R = 0.078$, respectively. This constant value is not attained during the 60 s stimulation protocol with $R = 0.002$ at all. Reaching such a steady-state behavior is needed for the system to be robust to changes in the number of spike-pair presentations. Indeed, if the PP1 activity is still in the raising phase at the end of the stimulation protocol (as in the case $R = 0.002$, see blue line in Figure 10A), then more spike-pair presentations would lead to a higher PP1 level and therefore to up-to-down transitions for a drastically wider range of $\Delta t$ values if the system is initially in the UP state. On the other hand, less spike-pair presentations would not give rise to any transitions at all. Figure 10C and 10D give an insight on how the $\Delta t$ range for which transitions occur depends on $R$, $Q$, and the number of spike-pair presentations. For $R = 1$ and $Q = 1$ ($k_{CaN} = 18$ 1/s, $\Delta Ca_{pre} = 0.17$ $\mu$M), the range of $\Delta t$ values evoking down-to-up transitions saturates beyond 50 spike-pair presentations, whereas the range resulting in up-to-down transitions becomes essentially insensitive to the number of spike-pair presentations beyond $\sim 150$ spike pairs (see green regions in Figure 10C and 10D; both depict the same results). Increasing $R$ further does not lead to any significant changes in the

PLoS Computational Biology | www.ploscompbiol.org | 2313 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

ranges of transitions compared to the case R = 1 (compare green and blue regions in Figure 10C for R = 1 and R = 2, respectively, and see Materials and Methods). Decreasing R slows down the convergence toward a stable range of time lags evolving down-to-up or up-to-down transitions (compare R = 1, green regions, and R = 0.078, red regions, cases in Figure 10C). This is due to the slower convergence of PP1 activity to its oscillatory behavior around a constant value during the stimulation protocol (see Figure 10A). The examples in Figure 10A show furthermore that the smaller R, the slower the decay of PP1 activity after the stimulation protocol. When R = 0.078 the PP1 dephosphorylation activity decays after the stimulation protocol so slow that large positive time lag stimulations evoke transitions from UP to DOWN (see upper red shaded region in Figure 10C, Q = 1, k<sub>m5</sub> = 18 μM, k<sub>cat</sub> = 0.17 μM). Up-to-down transitions at large positive time lags appear in the range up until 200 spike-pair presentations for R ≤ 0.202.

Similar arguments hold for the scaling parameter Q that controls CaMKII autophosphorylation and dephosphorylation dynamics. Indeed, the degree of CaMKII subunit phosphorylation should also reach a steady oscillation around a constant value during the presentation of spike pairs, for the robustness of bistable behavior (see the R = 1.Q = 1.5 and R = 1.Q = 0.083 cases in Figure 10B). The larger the Q, the faster autophosphorylation (through an increase of k<sub>s</sub> and k<sub>s</sub>; Figure 1C–1E) and dephosphorylation (through an increase of k<sub>1a</sub>, Equation 6) proceed. Therefore, less spike-pair presentations are required to evoke transitions and Δt ranges leading to down-to-up or up-to-down transitions saturate at smaller numbers of spike-pair presentations (see blue line in Figure 10B). For Q = 0.083, no transitions are observed at all in the range from 1 to 200 spike-pair presentations (see blue line in Figure 10B), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For LTP ≤ 0 (k<sub>cat</sub> = 0.083 case in Figure 10B). Down-to-up transitions appear after 60 spike-pair presentations for Q ≤ 0.6 (k<sub>cat</sub> = 18 μM, R = 1), whereas up-to-down transitions occur at lower Q values (see red regions for R = 1.Q = 0.083 case in Figure 10B). For Lwhich is supported by experimental studies reporting a prolonged localization of CaMKII at its postsynaptic site following LTP stimulation [25,62,63].

Experimentalists have reported several types of synaptic increase or decrease. For example, LTD decrease of efficacy from "basal" strength) and depotentiation (decrease of efficacy after potentiation) have often been considered as two distinct processes. Some of the differences between the two can be reconciled in our model by considering that a "basal" condition is likely to be a mix of synapses in the UP and DOWN state. Hence, a LTD protocol will decrease synaptic strength by provoking up-to-down transitions in some synapses that were initially in the UP state. On the other hand, in depotentiation protocols the initial conditions are different, because a larger fraction of the synapses are in the UP state. However, some studies indicate that depotentiation and LTD might operate through different molecular mechanisms [64–66]. A more complex model than the one proposed here would be needed to account for these experimental data.

## LTP/LTD Transitions at Fixed Calcium Concentration

As in previous models, our model exhibits LTP for high enough calcium concentrations. Unlike previous models, however, it possesses an "LTD window", where the system makes a transition from the highly phosphorylated to the weakly phosphorylated state, under plausible conditions. There are three requirements for LTP and LTD transitions to occur at realistic calcium concentrations in our model (see Figure 3C).

(I) The steady-state concentration of phosphorylated CaMKII subunits has to exhibit a bistable behavior, i.e., a highly and a weakly phosphorylated state should coexist in a range of calcium. This is the case if phosphatase activity saturates at high CaMKII phosphorylation levels. In turn, this is ensured if the phosphatase is present in small amounts compared to CaMKII, which itself is enriched at high concentrations in the PSD [24,27,47]. We show that bistability is a property of the CaMKII system which is robust to variations in the number of interacting subunits (see Figure 2A).

(II) The phosphatase activity at resting conditions has to allow for two stable CaMKII phosphorylation states. Bistability at C<sub>0</sub> is a robust property of the model over a large range of values of most of the protein signaling cascade parameters (see Table 1B). However, this requirement constrains the calcium-independent activities of the calcineurin and the cAMP-PKA pathways and is the reason why the model presented here is sensitive to changes in PKA base activity k<sup>b</sup><sub>k<sub>s</sub></sub>, i.e., varying k<sup>b</sup><sub>k<sub>s</sub></sub> than 14% from the value given in Table 1B leads to a loss of bistability at resting conditions.

(III) The "LTD window" emerges from an elevated phosphatase activity in the range of intermediate calcium concentrations. There are two possible realizations of the cAMP-PKA pathway for such an "LTD window" to arise: (i) if the PKA activity is assumed to be calcium-independent, the PP1 activity curve (purple lines in Figure 3B) would show a Hill-function-like behavior. However, a CaMKII versus calcium bifurcation diagram qualitatively similar to Figure 3C could still be obtained. How such a scenario would affect


PLoS Computational Biology | www.ploscompbiol.org

STDP in a Bistable Synapse Model

the behavior of the system in response to the STDP protocol is still to be clarified. (ii) If the cAMP–PKA pathway is calcium/calmodulin-dependent as chosen here (see also [61]), the PP1 activity can be coupled to the calcium concentration such that a peak emerges at intermediate calcium concentrations. Several lines of experimental evidence support the inclusion of such a calcium-dependent cAMP–PKA pathway which promotes LTP by blocking phosphatases in the model: the induction of hippocampal LTP is blocked by inhibiting cAMP-dependent protein kinase A or inhibition of postsynaptic kinases in general and is facilitated in a PKA-dependent manner by inhibiting calcineurin [38,60,67]; a rapid increase in PKA activity accompanies the early phase of LTP in afferent fibers between hippocampus and prefrontal cortex [68]; calcium-stimulable forms of cAMP exist which indirectly control PKA activity [69]. For the CaMKII system to exhibit the "LTD window" with a calcium/calmodulin-dependent cAMP–PKA pathway, the model predicts that the cAMP–PKA pathway should activate at higher calcium concentrations compared to the calcineurin pathway, as this is required for the peak of phosphatase activity to emerge.

Another way to assess the coupling of the protein signaling cascades to PP1 activity and to CaMKII is to check what the model predicts if we block different parts of the pathways and compare it to experimental results. We can implement the blockade of the calcineurin or the cAMP–PKA pathways in the model by removing the calcium/calmodulin-dependence of the calcineurin or the cAMP–PKA pathways, since inhibitor 1 is also dephosphorylated by the calcium-independent protein phosphatase 2A [70,71] and phosphorylated by the calcium-independent protein kinase G [72]. Blocking the calcium/calmodulin-dependent part of the calcineurin pathway (i.e., $k_{\text{CaN}} = 0$) leads to facilitation of LTP, and the reverse transition (LTD) is prevented. On the contrary, blocking the calcium/calmodulin-dependent part of the PKA pathway (i.e., $k_{\text{PKA}} = 0$) facilitates LTD and prevents LTP. Transitions in either one of both directions can be evoked since bistability at resting conditions is preserved in both cases. All these model predictions are consistent with experimental assays inhibiting either the calcineurin [37,43,60] or the cAMP–PKA pathway [38]. If either the calcineurin or the cAMP–PKA pathways are completely abolished in the model, i.e., both the calcium-independent and the calcium-dependent parts are suppressed (i.e., $k^0_{\text{CaN}} = k_{\text{CaN}} = 0$ or $k^0_{\text{PKA}} = k_{\text{PKA}} = 0$), the system becomes locked in the UP or the DOWN state, respectively. Under these conditions, bistability is not present at resting calcium concentrations, i.e., no transitions can be evoked in a stable fashion. This also means a change in basal synaptic transmission since all synapses in the system will converge to one of the two stable states. Along the lines of the argumentation above, this situation would correspond to a scenario in which all proteins de- or phosphorylating inhibitor 1 are inhibited. Inhibiting completely protein phosphatase 1 activity, i.e., setting PP1 activity to zero, results in locking the system to the UP state for all calcium concentrations in our model. However, other calcium-independent phosphatases such as protein phosphatase 2A and 2C are known to dephosphorylate CaMKII [73]. Adding such phosphatases to the model would lead to bistability even in the absence of PP1. Such a scenario would be consistent with experiments which have shown that LTD but not LTP requires the activation of PP1 [37,39,74]. Our model indeed predicts a progressive diminution of the LTD window and an enlargement of the LTP window as a function of PP1 inhibition. In response to the STDP protocol, LTD disappears first when phosphatase activity is decreased as suggested by experimental results [67]. Reducing the phosphatase activity further results in down-to-up transitions for all $\Delta t$s before the stable DOWN state disappears if the total PP1 concentration is reduced below 40%.

In addition to the "LTD window" at intermediate calcium concentrations, our model possesses a second region of bistability between the "LTD window" and the "LTP window" (see region between points 3 and 4 in Figure 3C). This region is not present in previous models and can be seen as a region of no changes. Starting from the DOWN or the UP state, calcium elevations to this range do not evoke any transition. A similar region of calcium concentrations in between LTP and LTD calcium levels leading to no plasticity is found experimentally by Cho et al. and discussed by Lisman as "no man's land" [75,76].

## LTP/LTD Transitions in Response to STDP Protocols

We have shown that the model can qualitatively reproduce plasticity outcomes in response to the STDP protocol. In our model as in previous models [9,10,77,78], the only signal driving synaptic changes is the dynamics of the calcium concentration, consistent with current experimental data [3,55,79–82]. However, previous modeling studies that use either the maximum amplitude of the calcium signal or simple readout mechanisms of the entire calcium dynamics reproduce only partially STDP results [9,10,77,78]. In particular, it has proven difficult to prevent the appearance of a second LTD range at large positive $\Delta t$s. Shouval and Kalantzis show that stochastic properties of synaptic transmission can markedly reduce the LTD magnitude in this range [83]. Karmarkar et al. hypothesize that two functionally distinct calcium pools trigger different readout mechanisms for LTP and LTD in order to overcome this difficulty [9]. Here, we show that the compound calcium signal from VDCCs and NMDA-Rs combined with a complex readout mechanism is sufficient to account for experimental STDP data; in other words, the two calcium influxes do not have to be separated. This is due to the highly cooperative CaMKII autophosphorylation and the protein signaling cascade influencing PP1 activity, which provide a strongly nonlinear detector system, which is sensitive enough to translate differences in the time course of the calcium concentration into observed plasticity outcomes. Finally, CaMKII phosphorylation level changes need to sum over several pairs of spikes in order to observe LTP- or LTD-like transitions, as suggested by experiments on STDP [4,43,84–86]. These changes combine in a highly nonlinear fashion in our model, going beyond simple summation of pairwise interactions. In particular, a minimal number of spike pairs is needed to observe any plasticity, as shown in Figure 10. This number depends on the kinetics of autophosphorylation and dephosphorylation dynamics in the model. Froemke et al. (visual cortex slices) and Wittenberg and Wang (hippocampal slices) showed that LTP (causal spike pairings) requires only a few spike-pair presentations whereas the appearance of LTD (anti-causal pairings) requires a longer period of stimulation (~100 spike pairings) [87,88]. Figure 10C and 10D show the faster saturation of the time lag

PLoS Computational Biology | www.ploscompbiol.org 2315 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

range evoking LTP compared to the one evoking LTD, consistent with those experimental results. Interestingly, Wittenberg and Wang see a second LTD range at large positive time differences emerging after sufficiently long stimulation (70–100 spike pairings; compare second LTD window at positive $\Delta t$ emerging at high spike-pair presentation numbers in Figure 10C and 10D for ($R = 0.078, Q = 1$) and ($R = 1, Q = 1.67$), respectively).

Rubin et al. recently proposed a model based on pathways resembling the CaMKII kinase-phosphatase system, which reproduces experimental STDP outcomes but does not exhibit bistability [44]. In that model, high, short-lasting calcium levels evoke LTP, low and prolonged calcium elevations above a certain threshold evoke LTD, and intermediate calcium levels act like a "Veto" preventing LTD induction. The durations for which their detector system has to be exposed to respective calcium levels are consistent with our findings. The competition between the PP1 buildup level and the autophosphorylation progress implements naturally the concept of the veto in our model. This balance between PP1 activity and autophosphorylation changes with $\Delta t$ and defines the transition outcome: (short negative $\Delta t$s) high PP1 accumulation and intermediate autophosphorylation of CaMKII evoked by linear interactions of the calcium influxes lead to LTD; (short positive $\Delta t$s) low PP1 activity together with strong autophosphorylation of CaMKII as a result of supralinear calcium summations produce LTP; (all other cases) intermediate PP1 concentrations and weak to intermediate autophosphorylation arouse no changes. In particular, the stronger cAMP–PKA pathway activation due to higher calcium elevations for large positive $\Delta t$ protocols can be seen as a realistic veto preventing LTD transitions to occur in this range. The differential activation of competing pathways at different calcium levels receives further support by recent experimental studies [89].

We observe a larger extent of the range of $\Delta t$ values evoking LTD compared to the LTP range for $R = 1$ and $Q = 1$ in the noiseless case (see Figure 10D, green regions). In our model, the LTD range can be either larger, or smaller, than the LTP range, depending on various parameters such as noise, $R$, and $Q$. For large noise levels, the LTD range is generally smaller than the LTP range, while experimental data seems to indicate the opposite trend (compare blue line in Figure 7A and [2,4,85]). Investigating extensively how the parameters of the system change the extent of the LTP and LTD ranges goes beyond the scope of this study. In any case, the range of $\Delta t$s leading to up-to-down transitions cannot be extended beyond the range of interaction between both calcium influxes. Hence, we expect the LTD range to become larger if this interaction is extended, e.g., due to nonlinear buffer dynamics [44] or the recruitment of additional protein signaling cascades [55,75,90]. Furthermore, BPAP attenuation and broadening has been shown experimentally to affect the STDP results and could change the balance between the ranges of time lags evoking LTP and LTD in our model [87,91]. Our model predicts that the range of time lags evoking LTP in response to the STDP protocol can be increased by amplifying PKA activity. On the other hand, increasing the strength of the calcineurin pathway shifts down horizontally the entire STDP curve (Figure 7A), leading to LTD transitions at all $\Delta t$s (unpublished data).

Our model also reproduces qualitatively experimental transition results evoked by a purely presynaptic stimulation protocol [3]. Low stimulation frequencies evoke LTD and high frequencies LTP with a transition from LTD to LTP at 16–17 Hz in our simulations (compare with [3] where the transition happens at around 10 Hz but 900 presynaptic spikes are presented, instead of 60 here), for the same parameters that fit qualitatively the STDP data. Our model furthermore predicts that postsynaptic frequency stimulation evokes LTP at frequencies above 50 Hz (see Figure 7B). Interestingly, this type of stimulation does not evoke transitions from UP to DOWN at any frequency. However, we expect this form of plasticity to be strongly dependent on the extent and the time course of BPAP amplitude suppression. We also exposed the CaMKII system to purely presynaptic or postsynaptic spike-pair stimulation protocols. Since presynaptic spikes evoke long-lasting calcium transients and postsynaptic spikes high but fast-decaying calcium elevations, the $\Delta t$ ranges for which transitions can be observed in the two cases are markedly different. In particular, our model predicts pairs of postsynaptic spikes should elicit down-to-up transitions only if spikes are very closely spaced, and only when the frequency of the pair is large enough ($3 \le \Delta t \le 8$ ms for $f = 1$ Hz). In the case of presynaptic spike pairs occurring at 1 Hz, the model predicts depression or up-to-down transitions for all values of $\Delta t$. The $\Delta t$ ranges for which transitions occur become smaller if the presentation frequency of the spike pairs is reduced (presynaptic spike pairs at 0.5 Hz: down-to-up transitions for $\Delta t < 300$ ms; postsynaptic spike pairs at 0.5 Hz: up-to-down transitions for $3 \le \Delta t \le 6$ ms). Nevian and Sakmann found that three postsynaptic spikes at 50 Hz, repeated 60 times at 0.1 Hz, do not evoke any synaptic changes [55]. We find a similar outcome in our model, but predict that an increase in frequency and/or decrease in the burst inter-spike interval should lead to potentiation. This prediction is, however, again sensitive to the extent of summation in calcium in between spikes. If the calcium transients evoked by the back-propagating action potentials do not accumulate, but the second BPAP evokes a calcium transient with the same amplitude as the first one, no down-to-up transitions are observed in the model. We have also investigated the transition behavior of the CaMKII system in response to spike triplets [43], and our model reproduces qualitatively such data provided short-term depression (STD) is added, as in [92] (unpublished data).

In conclusion, our model possesses two stable states of CaMKII activation, which could represent the core mechanism of binary synaptic strength maintenance. We furthermore show that it is possible to reproduce qualitatively experimental STDP results on LTP- and LTD-like transitions. These two results taken together suggest that the CaMKII-associated protein network could account for storage and induction of synaptic changes. Our model therefore predicts that the CaMKII protein also plays a major role in LTD, namely that CaMKII gets dephosphorylated during LTD induction. Experiments addressing the role of CaMKII in LTD provide controversial results. Sajikumar et al. showed that LTD in hippocampal CA1 neurons is blocked by CaMKII inhibition during induction but the application of the CaMKII inhibitor (KN-62) after the induction had no impact on LTD [93]. In other experiments, LTD has been shown to occur in the presence of CaMKII inhibitors during LTD

PLoS Computational Biology | www.ploscompbiol.org 2316 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

induction in hippocampal cultures and slices [21,43]. Such inhibitors bind to CaMKII and block its activation by calmodulin (inhibitor KN-62, which is known not to inhibit the autophosphorylated kinase; used in [43]) or interact with the ATP-binding site of CaMKII (K252a used in [21]) [24]. In the presence of each of both inhibitors, CaMKII can still get dephosphorylated by PP1. We predict that LTD will no longer occur if the CaMKII–phosphatase interaction is disrupted. However, LTD experiments on the hippocampus, the somatosensory cortex, as well as the perirhinal cortex of rats suggest that the metabotropic glutamate receptor (mGluR) pathway is also involved in LTD [55,75,90,94]. The biochemical cascades emerging from mGluR activation could in principle make the occurrence of LTD transitions more robust. The negative coupling of group II mGluRs with the cAMP–PKA pathway [95–97] is consistent with the idea presented here, that LTD requires a shift in kinase–phosphatase balance in favor of phosphatases. Overall, we suggest the dynamics of the global calcium time course play a crucial role for the sign of synaptic changes alongside the crosstalk between signaling cascades that include the one considered here.

# Materials and Methods

**Model of the CaMKII system with constant PP1 activity.** *Calcium binding to calmodulin.* Calmodulin contains four calcium binding sites, two at the C- and two at the N-terminal domain. Calcium binding happens in a cooperative manner in each one of these pairs [98]. The following scheme describes the macroscopic binding of calcium to calmodulin, i.e., we take into account the number of bound calcium ions only, regardless of the occupied microscopic binding sites:

$$M \xrightleftharpoons[-Ca^{2+}]{+Ca^{2+} K_1} C_1 \xrightleftharpoons[-Ca^{2+}]{+Ca^{2+} K_2} C_2 \xrightleftharpoons[-Ca^{2+}]{+Ca^{2+} K_3} C_3 \xrightleftharpoons[-Ca^{2+}]{+Ca^{2+} K_4} C. \quad (1)$$

Here M is free intracellular calmodulin and $C_i$ ($i = 1,2,3,4$) with $C_4 \equiv C$ denotes the calcium/calmodulin complex with $i$ bound calcium ions. Calmodulin target proteins including CaMKII are partially activated by calmodulin with two, three, or four calcium ions bound. However, CaMKII autophosphorylation rates induced by calmodulin bound with two or three calcium ions are much smaller than with calmodulin bound with four calcium ions [99]. Hence, we consider for simplicity in the model that only calmodulin bound with four calcium ions is able to phosphorylate CaMKII. Since the binding of calcium by calmodulin is fast (with binding rates of the order $\sim 1000 \text{ }(\mu\text{M/s})^{-1}$ [100,101]), we assume reaction Equation 1 to be in equilibrium with the calcium concentration. The macroscopic dissociation constants of successive calcium binding are taken from Linse et al. (see Table 1A for parameters [100]). The total concentration of calmodulin is $CaM_0 = M + C_1 + C_2 + C_3 + C$. Experimental studies suggest that the total available level of calmodulin in neurons is $CaM_0 \approx 10 \text{ }\mu\text{M}$ [98,102,103]. Here, we use a smaller value due to the vast number of target proteins of calmodulin besides CaMKII, and the sequestration of calmodulin by neurogranin in spines under resting conditions (see Table 1 and [102,104]). For simplicity, we do not consider the dynamics of calmodulin sequestration by neurogranin, which has been suggested to provide calmodulin during LTP protocols [105]. Assuming a calmodulin bath is an effective way to implement dissociation of calmodulin–neurogranin complexes, which provides calmodulin to the PSD during autophosphorylation and phosphatase/kinase activation. Italic style symbols in this manuscript refer to concentrations of the respective element or protein.

*Autophosphorylation of CaMKII.* The calcium/calmodulin-dependent protein kinase II (CaMKII) holoenzyme has 12 domains, grouped into two clusters each with six functionally coupled subunits [48,49]. CaMKII is activated by $Ca^{2+}$/calmodulin binding to its subunits. $Ca^{2+}$/calmodulin binding to adjacent subunits in the subunit ring stimulates intersubunit autophosphorylation at the residue threonine-286 in the autoregulatory domain ($Thr^{286}$). Autophosphorylation increases CaMKII affinity for $Ca^{2+}$/calmodulin and prolongs activation beyond dissociation of $Ca^{2+}$/calmodulin. In turn, as long as CaMKII stays activated it can bind to the NMDA-R and phosphorylate exogenous substrates [24,49]. For simplicity, some aspects of CaMKII function are not accounted for in the model. Any differences between the CaMKII$\alpha$ and -$\beta$ isoforms are not considered. The binding of calcium/calmodulin and protein phosphatase 1 to a subunit is assumed to be independent of the state of neighboring subunits. The autophosphorylation at $Thr^{305}$ and $Thr^{306}$ is not considered.

CaMKII autophosphorylation is an intersubunit process during which one subunit acts as substrate and the neighboring subunit as catalyst. For autophosphorylation to take place, calmodulin has to be bound to the substrate subunit [49]. Autophosphorylation at $Thr^{286}$ or binding of calmodulin each disable the autoinhibitory domain, therefore the catalytic subunit can be in one of the following states: (i) bound with calmodulin, (ii) phosphorylated and bound with calmodulin, or (iii) phosphorylated only (for an illustration see Figure 1C–1E) [27].

The chemical reaction schemes in Figure 1A–1E show schematically how binding of calmodulin and autophosphorylation is represented in the model. Reactions in 1A and 1B show calcium/calmodulin complex binding to dephosphorylated- or phosphorylated subunits, respectively. Autophosphorylation steps where the catalytic subunit is bound with calcium/calmodulin, phosphorylated and bound with calcium/calmodulin, or phosphorylated only are illustrated in Figure 1C, 1D, and 1E, respectively. The intersubunit autophosphorylation is likely to be a directed interaction in the ring and is here assumed to proceed in a single direction [27].

For the autophosphorylation steps depicted in Figure 1C, 1D, and 1E to occur, the substrate subunit must bind the calcium/calmodulin complex C. Let $\gamma$ be the probability that a dephosphorylated subunit S binds with C, i.e., $\gamma = SC / (S + SC)$ (SC stands for a dephosphorylated subunit bound with C); and $\gamma^*$ the probability that a subunit phosphorylated at $Thr^{286}$, $S^*$, binds with C, i.e., $\gamma^* = S^*C / (S^* + S^*C)$ ($S^*C$ stands for a phosphorylated subunit bound with C). Assuming reactions in Figure 1A and 1B to be in equilibrium and using the Law of Mass Action yields $SC = S \cdot C / K_5$ and $S^*C = S^* \cdot C / K_9$, respectively, where $K_5 = k_{-5} / k_5$ and $K_9 = k_{-9} / k_9$ are the dissociation constants of reactions shown in Figure 1A and 1B, respectively. These assumptions lead to:

$$\gamma = \frac{SC}{S + SC} = \frac{C}{K_5 + C}, \quad (2)$$

$$\gamma^* = \frac{S^*C}{S^* + S^*C} = \frac{C}{K_9 + C}. \quad (3)$$

The probability that reaction in Figure 1C takes place in a unit time between two subunits in the single direction is $k_6 \gamma^2$. Correspondingly, the probability for reaction in Figure 1D to occur in a unit time is $k_7 \gamma \gamma^*$ and for reaction in Figure 1E to occur is $k_8 \gamma (1 - \gamma^*)$. This probabilistic description of autophosphorylation allows us to describe a given subunit by two possible states only, i.e., whether or not a subunit is phosphorylated at $Thr^{286}$. Note that with a six-subunit ring this yields 14 macroscopic distinguishable activation states (see below). This ansatz does not account for calmodulin consumption during the process of autophosphorylation, assuming a bath of calmodulin. Similar approaches have been used in the investigations of Okamoto and Ichikawa as well as Zhabotinsky in CaMKII models exhibiting bistability [17,18], and in other studies describing in detail CaMKII autophosphorylation, but do not exhibit bistability [106–109].

*Dephosphorylation of CaMKII.* PP1 is the only protein phosphatase that dephosphorylates CaMKII associated with the postsynaptic densities [73]. The dephosphorylation of a free, phosphorylated subunit, and a phosphorylated subunit bound with the calcium/calmodulin complex are described according to the Michaelis-Menten scheme:

$$S^* + D \xrightleftharpoons[k_{-11}]{k_{11}} S^* \cdot D \xrightarrow{k_{12}} S + D, \quad (4)$$

$$S^*C + D \xrightleftharpoons[k_{-11}]{k_{11}} S^*C \cdot D \xrightarrow{k_{12}} SC + D, \quad (5)$$

where $D$ denotes the concentration of active PP1. Note that dephosphorylation happens independently whether a subunit is bound with the calcium/calmodulin complex or not. Assuming the $(S^* \cdot D)$ and $(S^*C \cdot D)$ formations are at equilibrium, i.e., $\frac{d}{dt}(S^* \cdot D) \approx \frac{d}{dt}(S^*C \cdot D) \approx 0$, we can use the standard Michaelis-Menten equation [110] to obtain the per-subunit rate of dephosphorylation, $k_{10}$,

PLoS Computational Biology | www.ploscompbiol.org | 2317 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

$$k_{10} = \frac{k_{12}D}{K_M + S_{active}},$$ (6)

where the Michaelis constant $K_M$ is given by $K_M = (k_{-11} + k_{12}) / k_{11}$ and $S_{active}$ is the total concentration of phosphorylated CaMKII subunits, $S_{active} = \sum_{i=0}^{n=13} m_i \cdot S_i$, where $m_i$ is the number of phosphorylated subunits of the macroscopic activation state $i$ (see below for more details). The per-subunit rate of dephosphorylation, $k_{10}$, is proportional to the amount of available phosphatase, $D$. The dephosphorylation rate per subunit declines if a lot of subunits are phosphorylated and the phosphatase activity remains constant, i.e., if $S_{active}$ is high and $D$ constant. This saturation of $k_{10}$ leads to the bistable behavior of the CaMKII phosphorylation level (see "Bistability of the CaMKII system with constant PP1 activity" section).

Applying the Law of Mass Action and taking into account the geometry of the CaMKII six-subunit ring, its autophosphorylation and dephosphorylation by PP1 is described by the following system of coupled, ordinary differential equations for concentrations of CaMKII with different numbers of phosphorylated subunits

$$\dot{S}_0 = \dot{S}_{000000} = -6k_6\gamma^2 S_0 + k_{10}S_1,$$ (7)

$$\dot{S}_1 = \dot{S}_{100000} = 6k_6\gamma^2 S_0 - 4k_6\gamma^2 S_1 - k_7\gamma S_1 - k_{10}S_1 + 2k_{10}(S_2 + S_3 + S_4),$$ (8)

$$\begin{aligned} \dot{S}_2 = \dot{S}_{110000} = & \ k_6\gamma^2 S + k_7\gamma S_1 - 3k_6\gamma^2 S_2 - k_7\gamma S_2 - 2k_{10}S_2 \\ & + k_{10}(2S_5 + S_6 + S_7), \end{aligned}$$ (9)

$$\begin{aligned} \dot{S}_3 = \dot{S}_{101000} = & \ 2k_6\gamma^2 S_1 - 2k_6\gamma^2 S_3 - 2k_7\gamma S_3 - 2k_{10}S_3 \\ & + k_{10}(S_5 + S_6 + S_7 + 3S_8), \end{aligned}$$ (10)

$$\dot{S}_4 = \dot{S}_{100100} = k_6\gamma^2 S_1 - 2k_6\gamma^2 S_4 - 2k_7\gamma S_4 - 2k_{10}S_4 + k_{10}(S_6 + S_7),$$ (11)

$$\begin{aligned} \dot{S}_5 = \dot{S}_{111000} = & \ k_6\gamma^2 S_2 + k_7\gamma(S_2 + S_3) - 2k_6\gamma^2 S_5 - k_7\gamma S_5 - 3k_{10}S_5 \\ & + k_{10}(2S_9 + S_{10}), \end{aligned}$$ (12)

$$\begin{aligned} \dot{S}_6 = \dot{S}_{110100} = & \ k_6\gamma^2(S_2 + S_3) + 2k_7\gamma S_4 - k_6\gamma^2 S_6 - 2k_7\gamma S_6 - 3k_{10}S_6 \\ & + k_{10}(S_9 + S_{10} + 2S_{11}), \end{aligned}$$ (13)

$$\begin{aligned} \dot{S}_7 = \dot{S}_{110010} = & \ k_6\gamma^2(S_2 + S_4) + k_7\gamma S_3 - k_6\gamma^2 S_7 - 2k_7\gamma S_7 - 3k_{10}S_7 \\ & + k_{10}(S_9 + S_{10} + 2S_{11}), \end{aligned}$$ (14)

$$\dot{S}_8 = \dot{S}_{101010} = k_6\gamma^2 S_3 - 3k_7\gamma S_8 - 3k_{10}S_8 + k_{10}S_{10},$$ (15)

$$\begin{aligned} \dot{S}_9 = \dot{S}_{111100} = & \ k_6\gamma^2 S_5 + k_7\gamma(S_5 + S_6 + S_7) - k_6\gamma^2 S_9 - k_7\gamma S_9 \\ & - 4k_{10}S_9 + 2k_{10}S_{12}, \end{aligned}$$ (16)

$$\begin{aligned} \dot{S}_{10} = \dot{S}_{111010} = & \ k_6\gamma^2(S_5 + S_6) + k_7\gamma(S_7 + 3S_8) - 2k_7\gamma S_{10} \\ & - 4k_{10}S_{10} + 2k_{10}S_{12}, \end{aligned}$$ (17)

$$\dot{S}_{11} = \dot{S}_{110110} = k_6\gamma^2 S_7 + k_7\gamma S_6 - 2k_7\gamma S_{11} - 4k_{10}S_{11} + k_{10}S_{12},$$ (18)

$$\begin{aligned} \dot{S}_{12} = \dot{S}_{111110} = & \ k_6\gamma^2 S_9 + k_7\gamma(S_9 + 2S_{10} + 2S_{11}) - k_7\gamma S_{12} \\ & - 5k_{10}S_{12} + 6k_{10}S_{13}, \end{aligned}$$ (19)

$$\dot{S}_{13} = \dot{S}_{111111} = k_7\gamma S_{12} - 6k_{10}S_{13}.$$ (20)

Here $S_i$ refers to the concentration of the 14 ($i = 0, \dots, 13$) macroscopic distinguishable activation states of the CaMKII protein. The subscript in the second column shows the geometrical order of Thr<sup>286</sup> phosphorylated sites in the CaMKII ring, 1 refers to a phosphorylated subunit, 0 to a dephosphorylated subunit. Attention should be drawn to the fact that, for example, $S_5$, $S_6$, $S_7$, and $S_8$, all have three phosphorylated subunits, i.e., all of them have the same macroscopic level of activation, i.e., $m_5 = m_6 = m_7 = m_8 = 3$. However, in terms of symmetry all four have to be distinguished since at $S_5$ the phosphorylated sites are adjoined, $S_{111000}$, whereas at $S_8$ they are separated by a dephosphorylated subunit, $S_{101010}$, for example. With regard to this difference, the propagation of autophosphorylation is different for both, the phosphorylation step shown in Figure 1C can occur on two pairs of subunits at $S_5$ but cannot occur at $S_8$ at all. Taking into account that the different autophosphorylation steps, depicted in Figure 1C–1E, happen with different probabilities leads to differing occurrences of the activation states $S_i$ (with $i = 0 \dots 13$). Note that we used the fact $k_7 = k_8$ and simplified $k_7\gamma\gamma^* + k_8\gamma(1 - \gamma^*)$ to $k_7\gamma$. $k_{10}$ is the per-subunit rate of dephosphorylation (see above).

$\sum_{i=0}^n \dot{S}_i = 0$, with $n = 13$ for 14 macroscopic distinguishable activation states of the six-subunit CaMKII ring, yields the CaMKII protein mass conservation, $2CaMKII_0 = \sum_{i=0}^n S_i$. $2CaMKII_0$ gives the total concentration of functionally independent CaMKII clusters of six subunits and $CaMKII_0$ the total CaMKII concentration since one holoenzyme comprises two six-subunit rings. Note that the number of macroscopic distinguishable activation states is 3, 6, 14, and 36 for the two, four, six, and eight subunit models, respectively.

**Model with Ca-dependent PP1 activity via protein signaling cascade including PKA and calcineurin** The dephosphorylation activity of PP1 is indirectly governed by calcium/calmodulin via inhibitor 1 (I1), i.e., phosphorylated inhibitor 1 inhibits PP1 [111,112]. Inhibitor 1 itself is phosphorylated by cyclic AMP-dependent protein kinase A (PKA) and protein kinase G and dephosphorylated by the phosphatase calcineurin and protein phosphatase 2A [37,38,60,70–72]. A simple realization of this protein signaling cascade is given by

```mermaid
graph LR
    IG((IG)) -- "vPKA(C)" --> I((I))
    I -- "vCaN(C)" --> IG
``` (21)

```mermaid
graph LR
    I_D[I + D] -- "k13" --> DI((DI))
    DI -- "k-13" --> I_D
``` (22)

where $I_G$ refers to dephosphorylated I1, $I$ denotes phosphorylated inhibitor 1 (I1P), $D$ is free PP1, and $D_I$ stands for inhibited PP1, i.e., PP1 bound with phosphorylated inhibitor 1. See Figure 1F for a scheme of the protein signaling cascade.

The balance between inhibitor 1 phosphorylation ($v_{PKA}$)- and dephosphorylation rate ($v_{CaN}$) is calcium/calmodulin-dependent. The enzymatic activity of calcineurin can be described by a Hill equation [113]. The PKA activity is also known to be calcium/calmodulin-dependent via cyclic AMP [69], but there is no data characterizing this dependency. We chose to describe both by a Hill equation

$$v_X(C) = k_x^0 + \frac{k_x}{1 + (\frac{K_x}{C})^{n_x}}, \quad x = \text{CaN, PKA}$$ (23)

with a calcium/calmodulin-independent base activity ($k_x^0$) which also accounts for protein kinase G phosphorylation ($x = \text{PKA}$) and protein phosphatase 2A dephosphorylation ($x = \text{CaN}$). $k_x$ is the maximal, calcium/calmodulin-dependent activity, $K_x$ the half activity concentration, and $n_x$ denotes the Hill coefficient.

Applying the Law of Mass Action and taking into account protein phosphatase 1 conservation yields

$$\frac{dI}{dt} = -k_{13}ID + k_{-13}(D_0 - D) - v_{CaN}(C)I + v_{PKA}(C)I_0,$$ (24)

$$\frac{dD}{dt} = -k_{13}ID + k_{-13}(D_0 - D),$$ (25)

where $I_0$ and $D_0 = D + D_I$ refer to the total I1 and PP1 concentration, respectively. The concentration of dephosphorylated, free inhibitor 1, $I_0$, is treated like a bath, assuming a rapid exchange between the PSD with the spine volume and between the spine and the parent dendrite, as in [17,57]. Therefore, inhibitor 1 is not conserved in Equation 24 due to this bath assumption.

**Approximation of the PP1 activity level after presentation of one spike pair.** As can be seen in Figure 5C and 5D, the change in PP1 activity, as well as the change in I1P concentration (unpublished data), during the presentation of one spike pair is small. We therefore separate both variables into two terms, a constant value and a small time-dependent change, i.e., $D(t) \to D^* + \epsilon \delta D(t)$ and $I(t) \to I^* + \epsilon \delta I(t)$, where $D^*$ and $I^*$ are the values before the spike-pair presentation, and $\delta D(t)$ and $\delta I(t)$ describe the changes during the presentation. Since these small changes are exclusively driven by changes in $v_{CaN}(t)$ and $v_{PKA}(t)$, we consider the time-dependent part of both rates as small compared to $k_{13}$ and $k_{-13}$, i.e., $v_{CaN} \to k_{CaN}^0 + \epsilon \delta v_{CaN}(t)$ and $v_{PKA} \to k_{PKA}^0 + \epsilon \delta v_{PKA}(t)$. Inserting these expressions in Equations 24

PLoS Computational Biology | www.ploscompbiol.org 2318 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

and 25 yields at zero order in $\epsilon$ the steady-state values $D^*$ and $I^*$. The first-order equations in $\epsilon$ are

$$\frac{d\delta I}{dt} = -(k_{13}D^* + v_{CaN})\delta I - (k_{13}I^* + k_{-13})\delta D - \delta v_{CaN}I^* + \delta v_{PKA}I_0, \tag{26}$$

$$\frac{d\delta D}{dt} = -k_{13}D^*\delta I - (k_{13}I^* + k_{-13})\delta D. \tag{27}$$

The Eigenvalues of the homogeneous system of Equations 26 and 27 are

$$\lambda_{\pm} = -(v_{CaN} + k_{13}D^* + k_{13}I^* + k_{-13})/2 \tag{28}$$

$$\pm \sqrt{(v_{CaN} + k_{13}D^* + k_{13}I^* + k_{-13})^2/4 - v_{CaN}(k_{13}I^* + k_{-13})}. \tag{29}$$

Since $v_{CaN}$ is much smaller than $k_{13}D^*$, $k_{13}I^*$, or $k_{-13}$, we expand the two Eigenvalues around $v_{CaN}$. This yields a fast and a slow Eigenvalue since $\lambda_+$ is zero at leading order. The Eigenvalues become

$$\lambda^{(fast)} \equiv \lambda_{fast} \approx -(k_{13}(D^* + I^*) + k_{-13}), \tag{30}$$

$$\lambda^{(slow)} \equiv \lambda_{slow} \approx -\frac{(k_{13}I^* + k_{-13})v_{CaN}}{k_{13}(D^* + I^*) + k_{-13}}. \tag{31}$$

With the initial conditions $\delta D(0) = \delta I(0) = 0$, the solution for the inhomogeneous system (Equations 26 and 27) becomes

$$\delta I(t) = A_1 e^{\lambda_{fast}t} \int_0^t e^{-\lambda_{fast}s} S(\tau) d\tau + B_1 e^{\lambda_{slow}t} \int_0^t e^{-\lambda_{slow}s} S(\tau) d\tau, \tag{32}$$

$$\delta D(t) = A_2 e^{\lambda_{fast}t} \int_0^t e^{-\lambda_{fast}s} S(\tau) d\tau + B_2 e^{\lambda_{slow}t} \int_0^t e^{-\lambda_{slow}s} S(\tau) d\tau, \tag{33}$$

with $A_1 = -(k_{13}D^* + v_{CaN} + \lambda_{slow}) / (\lambda_{fast} - \lambda_{slow})$, $B_1 = (k_{13}D^* + v_{CaN} + \lambda_{fast}) / (\lambda_{fast} - \lambda_{slow})$, $A_2 = -k_{13}D^* / (\lambda_{fast} - \lambda_{slow})$, and $B_2 = k_{13}D^* / (\lambda_{fast} - \lambda_{slow})$. $S(\tau)$ is the inhomogeneous part in Equation 26, i.e., $S(\tau) = -\delta v_{CaN}(\tau)I^* + \delta v_{PKA}(\tau)I_0$. The first term in Equations 32 and 33 describes the fast dynamics of both variables and allows $D$ and $I$ to follow on a fast time scale the calcium transient. After the spike-pair presentation, this term decays rapidly with the time constant $\lambda_{fast}$. The second term determines the slow dynamics of the system and therefore gives rise to a slow buildup, which decays after the spike-pair presentation with the slow time constant $\lambda_{slow}$. Since $(\lambda_{slow} \cdot t)$ is small at the scale of single presentations, we obtain for the slow dynamics

$$\delta D(t) = \frac{k_{13}D^*}{\lambda_{fast} - \lambda_{slow}} \int_0^t (-\delta v_{CaN}(\tau)I^* + \delta v_{PKA}(\tau)I_0) d\tau. \tag{34}$$

$\delta D(t)$ is shown in Figure 9B as a measure for the slowly decaying PP1 buildup after the presentation of one spike pair. $D^* + \delta D(t)$ is compared with the PP1 activity obtained from numerical integration of Equations 24 and 25 after one spike pair in Figure 9A. Note that the product of $\delta D$, $D^*$, and $D$ with $k_{12}$ is shown in Figure 9A and 9B.

In the section "STDP protocol with deterministic calcium transients," we point out that an increase in $R$ beyond the value of 1 does not significantly affect the dynamics of the PP1 response, which is basically determined by $\lambda_+^{(slow)}$ (see paragraph above). This can be understood by considering $\lambda_+^{(slow)}$ (Equation 31), if $D^* \ll I^*$, $k_{-13}/k_{13}$, its denominator, will be controlled by $k_{13}I^*$ and $k_{-13}$ only, and changes in $D^*$ will have no impact on the PP1 dynamics.

**Synaptic activity and postsynaptic calcium signaling.** To investigate how the model behaves when realistic calcium transients are applied to it, we use the following model for postsynaptic calcium and postsynaptic membrane potential dynamics. We focus on a single spine compartment, and do not simulate the backpropagation of the action potential from its initiation site to the spine. Instead, we model the action potential dynamics directly at the spine.

*Postsynaptic membrane potential.* The postsynaptic membrane potential is modeled using the Hodgkin-Huxley formalism in a single compartment. The reference volume for the membrane potential and the calcium dynamics model is taken to be a postsynaptic spine ($V_{spine} \approx 1 \mu\text{m}^3$, $r_{spine} \approx 0.5 \mu\text{m}$). The dynamics of the membrane potential $V$ follows the differential equation

$$C_m \frac{dV}{dt} = -I_L - I_{Na} - I_K - I_{NMDA} - I_{CaL} - I_{AMPA} + I_{stim} \tag{35}$$

where $C_m$ is the whole cell capacitance of 0.1 nF, $I_x$ ($x = \text{L, Na, K, NMDA, CaL, AMPA}$) are the ionic currents listed below. An action potential is evoked by a 1 ms depolarizing pulse current $I_{stim}$ of 3 nA.

*Postsynaptic calcium dynamics.* The model of the calcium dynamics involves the two main sources of postsynaptic calcium influx in the spine: NMDA receptors (NMDA-R) and voltage-dependent calcium channels (VDCC) [114]. Extrusion, diffusion, and slow buffering is accounted for by a single exponential decay, yielding the following equation for the time course of the intracellular calcium concentration

$$\tau_{Ca} \frac{dCa}{dt} = -(Ca - Ca_0) + \tau_{Ca} \zeta (\beta_{NMDA} I_{NMDA} + \beta_{CaL} I_{CaL}) \tag{36}$$

where $Ca$ is the free, intracellular calcium concentration, $\tau_{Ca} = 12$ ms refers to the single exponential time constant of the passive decay process [54], $Ca_0$ is the calcium resting concentration, and $\zeta = 2.59 \cdot 10^4 \text{ m}^2 \text{ }\mu\text{M/C}$ converts the ion currents into concentration changes per time for a spine of volume $\approx 1 \mu\text{m}^3$. $I_x$ ($x = \text{NMDA, CaL}$) are the ionic currents listed below. Scaling parameters $\beta_{NMDA} = 1/1000$ and $\beta_{CaL} = 1/100$ take into account both the immediate uptake of calcium by intracellular buffers ($\approx 99\%$, [115]) and the fact that only about $\approx 10\%$ of the NMDA-mediated current is carried by calcium ions (see [116,117] and below).

*Noisy calcium transients.* To investigate stochastic effects, we add two possible noise sources to the model: (i) NMDA receptor maximum conductance drawn at random at each presynaptic spike and (ii) maximum conductance of the voltage-dependent calcium channel drawn at random at each postsynaptic spike. Both conductances are drawn from binomial distributions characterized by the total number of channels $N_{tot}$ and the opening probability per channel $p_o$. Each presynaptic or postsynaptic spike gives rise to an integer number, $n_o$, of NMDA or CaL channel openings, respectively. We assume that the channels open independently of each other. The single channel conductance $g_{single}$ is chosen so that the mean calcium amplitudes are as stated above. To account for the stochasticity of calcium ions influx, Gaussian noise with zero mean and a variance scaled with $n_o$ is added to ($n_o \cdot g_{single}$). The parameters of the NMDA and CaL maximum conductance distributions are adjusted such that they fit the experimental findings of single spine measurements by Mainen et al. and Sabatini and Svoboda, respectively [58,59] (see Table 2 for parameters).

*Ion currents dynamics.* The description and the parameters of the ionic currents are taken from Poirazi et al. ($I_{CaL}$) as well as Purvis and Butera ($I_{Na}, I_K$) [118,119].

LEAK CURRENT: The leak current is given by
$$I_L = g_L(V - E_L) \tag{37}$$
where $g_L$ is the leak conductance. The leak potential is adjusted such that the resting potential is -70 mV.

The ionic currents listed here have the general form $I_{ionic} = gy(V - E_{ionic})$. $E_{ionic}$ is the reversal potential for the respective ions carried, $g$ refers to the maximum conductance of each current, and $y$ is the product of one or more gating variables. $y$ determines the dynamics of the ion currents regulated by voltage-dependent activation and inactivation variables which are described according to

$$\frac{dx}{dt} = (x_\infty(V) - x) / \tau_x(V), \tag{38}$$

$$x_\infty(V) = \frac{1}{1 + \exp((V - \theta_x) / \sigma_x)}, \tag{39}$$

$$\tau_x(V) = \frac{A}{\exp((V - \theta_1) / \sigma_1) + \exp((V - \theta_2) / \sigma_2)} + B. \tag{40}$$

Here $x_\infty(V)$ is the steady-state voltage-dependent (in)activation function of $x$, and $\tau_x(V)$ is the voltage-dependent time constant. In terms of this formalism, the respective ion currents are given by:

SODIUM CURRENT:
$$I_{Na} = g_{Na} m_{Na}^3 h_{Na} (V - E_{Na}), \tag{41}$$

$$m_{\infty Na}(V) = \frac{1}{1 + e^{-(V+36)/8.5}}, \tau_{mNa} = 0.1 \text{ ms} \tag{42}$$

PLoS Computational Biology | www.ploscompbiol.org | 2319 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

**Table 2. Further Parameters Used in the Model**

<table>
  <thead>
    <tr>
        <th>Parameter</th>
        <th>Definition</th>
        <th>Value</th>
        <th>Reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>E<sub>L</sub></td>
        <td>Leak reversal potential</td>
        <td>-68.0331 mV</td>
        <td>See text.</td>
    </tr>
    <tr>
        <td>E<sub>Na</sub></td>
        <td>Sodium reversal potential</td>
        <td>60 mV</td>
        <td>[119]</td>
    </tr>
    <tr>
        <td>E<sub>K</sub></td>
        <td>Potassium reversal potential</td>
        <td>-80 mV</td>
        <td>[119]</td>
    </tr>
    <tr>
        <td>E<sub>AMPA</sub></td>
        <td>AMPA-mediated current reversal potential</td>
        <td>0 mV</td>
        <td>[120]</td>
    </tr>
    <tr>
        <td>E<sub>NMDA</sub></td>
        <td>Total NMDA-mediated current reversal potential</td>
        <td>0 mV</td>
        <td>[120]</td>
    </tr>
    <tr>
        <td>E<sub>Ca</sub></td>
        <td>Calcium reversal potential</td>
        <td>140 mV</td>
        <td>[118]</td>
    </tr>
    <tr>
        <td>g<sub>L</sub></td>
        <td>Maximum leak conductance</td>
        <td>0.005 μS</td>
        <td>[119]</td>
    </tr>
    <tr>
        <td>g<sub>Na</sub></td>
        <td>Maximum sodium conductance</td>
        <td>0.7 μS</td>
        <td>[119]</td>
    </tr>
    <tr>
        <td>g<sub>K</sub></td>
        <td>Maximum potassium conductance</td>
        <td>1.3 μS</td>
        <td>[119]</td>
    </tr>
    <tr>
        <td>g<sub>AMPA</sub></td>
        <td>Maximum AMPA current conductance</td>
        <td>0.0195 μS</td>
        <td>See text..</td>
    </tr>
    <tr>
        <td>g<sub>NMDA</sub></td>
        <td>Maximum NMDA-R current conductance</td>
        <td>4.5 · 10<sup>-4</sup> μS</td>
        <td>For example, for ΔCa<sub>pre</sub> = 0.17 μM, see text.</td>
    </tr>
    <tr>
        <td>g<sub>CaL</sub></td>
        <td>Maximum CaL current conductance</td>
        <td>5.6 · 10<sup>-4</sup> μS</td>
        <td>For example, for ΔCa<sub>post</sub> = 0.34 μM, see [54].</td>
    </tr>
    <tr>
        <td>N<sub>NMDA tot</sub></td>
        <td>Total number of NMDA receptors</td>
        <td>20</td>
        <td>[126,127]</td>
    </tr>
    <tr>
        <td>p<sub>NMDA o</sub></td>
        <td>Single channel opening probability</td>
        <td>0.5</td>
        <td>[127]</td>
    </tr>
    <tr>
        <td>σ<sub>NMDA</sub></td>
        <td>SD of the Gaussian noise added</td>
        <td>3.3% of g<sub>NMDA</sub></td>
        <td>[58]</td>
    </tr>
    <tr>
        <td>N<sub>CaL tot</sub></td>
        <td>Total number of CaL channels</td>
        <td>5</td>
        <td>[59]</td>
    </tr>
    <tr>
        <td>p<sub>CaL o</sub></td>
        <td>Single channel opening probability</td>
        <td>0.52</td>
        <td>[59]</td>
    </tr>
    <tr>
        <td>σ<sub>CaL</sub></td>
        <td>SD of the Gaussian noise added</td>
        <td>10% of g<sub>CaL</sub></td>
        <td>[59]</td>
    </tr>
  </tbody>
</table>

doi:10.1371/journal.pcbi.0030221.t002

$$h_{\infty Na}(V) = \frac{1}{1 + e^{(V+44.1)/7}}, \tau_{hNa}(V) = \left( \frac{3.5}{e^{(V+35)/4} + e^{-(V+35)/25}} + 1 \right) \text{ ms.} \tag{43}$$

DELAYED-RECTIFIER POTASSIUM CURRENT:

$$I_K = g_K n^4 (V - E_K), \tag{44}$$

$$n_{\infty}(V) = \frac{1}{1 + e^{-(V+30)/25}}, \tau_n(V) = \left( \frac{2.5}{e^{(V+30)/40} + e^{-(V+30)/50}} + 0.01 \right) \text{ ms.} \tag{45}$$

VOLTAGE-DEPENDENT CALCIUM CURRENT (HIGH-VOLTAGE ACTIVATED L-TYPE):

$$I_{CaL} = g_{CaL} m_{CaL}^3 h_{CaL} (V - E_{Ca}), \tag{46}$$

$$m_{\infty CaL}(V) = \frac{1}{1 + e^{-(V+37)}}, \tau_{mCaL} = 3.6 \text{ ms,} \tag{47}$$

$$h_{\infty CaL}(V) = \frac{1}{1 + e^{(V+41)/0.5}}, \tau_{hCaL} = 29 \text{ ms.} \tag{48}$$

AMPA CURRENT: excitatory postsynaptic potentials are mainly mediated by the AMPA receptor current given by

$$I_{AMPA} = g_{AMPA} s_{AMPA}(t)(V - E_{AMPA}), \tag{49}$$

$$\dot{s}_{AMPA} = -s_{AMPA}/\tau_{AMPA} + \alpha_s x_{AMPA}(1 - s_{AMPA}), \tag{50}$$

$$\dot{x}_{AMPA} = -x_{AMPA}/\tau'_{AMPA} + \alpha_x \sum \delta(t - t_k), \tag{51}$$

with $\tau_{AMPA} = 2 \text{ ms}$, $\tau'_{AMPA} = 0.05 \text{ ms}$, $\alpha_s = 1 \text{ 1/ms}$, and $\alpha_x = 1$ (dimensionless) [120,121]. $s_{AMPA}$ is a single exponentially decaying gating variable with a finite rise time (the time-to-peak is $\approx 0.2 \text{ ms}$). At each occurrence of a presynaptic spike at time $t_k$, the variable $x_{AMPA}$ is increased by one (the sum on the right-hand side of Equation 51 goes over all presynaptic spikes occurring at times $t_k$).

NMDA CURRENT: the current mediated by the NMDA receptor is described by

$$I_{NMDA} = -g_{NMDA} s_{NMDA}(t) B(V) (V - E_{NMDA/Ca}), \tag{52}$$

where the voltage dependence of the magnesium block is given by

$$B(V) = \frac{1}{1 + \exp(-0.062V) \frac{[Mg^{2+}]}{3.57}}. \tag{53}$$

The voltage dependence is controlled by the extracellular magnesium concentration $[Mg^{2+}] = 1.0 \text{ mM}$ [122]. The dimensionless gating variable $s_{NMDA}$ obeys the same types of equations as $s$ and $x$ of the AMPA current (Equations 50 and 51, respectively) but with $\tau_{NMDA} = 80 \text{ ms}$ and $\tau'_{NMDA} = 2 \text{ ms}$ [121] (the time-to-peak is $\approx 8 \text{ ms}$).

The maximum leak conductance and the whole cell capacitance yield a membrane potential time constant $\tau_m$ of 20 ms, according to the equation $\tau_m = C_m / g_L$. The AMPA receptor conductance $g_{AMPA}$ is chosen such that a single presynaptic spike evokes a maximal depolarization of 1 mV at -70 mV. $g_{NMDA}$ and $g_{CaL}$ are chosen such that the amplitudes of the NMDA-R mediated and the action potential-evoked calcium transients in the spine are realized as stated in the text. The ratio of $\approx 2$ between the BPAP evoked calcium transient amplitude ($\Delta Ca_{post}$) and the NMDA-R mediated contribution ($\Delta Ca_{pre}$) is as measured by Sabatini et al. [54]. Note that the VDCC- and the NMDA-mediated calcium currents in the calcium dynamics (Equation 36) are multiplied by the scaling parameters $\beta_{NMDA}$ and $\beta_{CaL}$, which account for fast calcium buffering and for the fractional calcium current through NMDA-Rs of $\approx 10\%$. The calcium reversal potential, $E_{Ca}$, is used to describe the fractional calcium current through NMDAs in the calcium dynamics (Equation 36), whereas the reversal potential of the compound sodium, potassium, and calcium ion current, $E_{NMDA}$, mediated by NMDA-Rs, is employed in the voltage equation (Equation 35).

*Parameters of the model.* The model describing the interactions between proteins contains a large number of parameters (25). In some cases, we used experimentally determined values (see Table 1A for a list of those parameters). Other parameters are not (or poorly) determined experimentally. These parameters were varied systematically or were determined by the constraints we impose on the model (see Table 1B). Finally, a few parameters were set on the basis of previous modeling studies or set to an arbitrary value, in cases in which changing this value does not alter the results of the model (see Table 1C).

The calcium-dependent steady-state concentration of phosphorylated CaMKII subunits depends heavily on the choice of the parameters defining the PKA pathway ($k^0_{PKA}, k_{PKA}, K_{PKA}, n_{PKA}$). These parameters are adjusted in order to obtain the "LTD" and the "LTP window" at specific intervals of calcium concentration (see section "LTD window" in a model with Ca-dependent PP1 activity via protein signaling cascade including PKA and calcineurin). The maximal calcineurin activity $k_{CaN}$ is used to adjust the PP1 level evoked during the STDP stimulation protocol (see section "STDP protocol stimulation with stochastic calcium dynamics").

The total calmodulin concentration ($CaM_0$) is smaller than the value found in experimental studies, due to the reasons given above (see "Calcium binding to calmodulin" paragraph). $K_M$ is taken from the modeling study of [17]. Equations 24 and 25 give the steady-state PP1 concentration, $D_{steady-state} = D_0 / (1 + (I_0 k_{13} v_{PKA}) / (k_{-13} v_{CaN}))$. Hence, $D_{steady-state}$ depends on $I_0$, $v_{PKA}$, and $v_{CaN}$ through the single

PLoS Computational Biology | www.ploscompbiol.org 2320 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

variable $I_0 \frac{v_{PKA}}{v_{CaN}}$. This means that out of the five parameters $I_0$, $k_{CaN}^0$, $k_{CaN}$, $k_{PKA}^0$, and $k_{PKA}$, the steady-state PP1 concentration depends on three independent combinations of those parameters, e.g., $I_0 \frac{k_{PKA}^0}{k_{CaN}^0}$, $k_{PKA}/k_{PKA}^0$, and $k_{CaN}/k_{CaN}^0$. Thereby, two out of these five parameters can be set arbitrarily. The total I1 concentration and the calcineurin base activity, $I_0$ and $k_{CaN}^0$, are set to the values given in Table 1C and are kept constant throughout all investigations, while the remaining three parameters $k_{CaN}$, $k_{PKA}^0$, and $k_{PKA}$ are obtained by constraints imposed on the model (see Table 1B). On the other hand, the dynamics of the protein signaling cascade depends on all five parameters. We address this issue via the scaling parameter $R$ which influences the PP1 response dynamics (see "STDP protocol stimulation with stochastic calcium dynamics" section).

The parameters describing postsynaptic calcium and postsynaptic membrane potential dynamics are taken from previous modeling studies [59]. We systematically vary the calcium amplitudes evoked by a presynaptic ($\Delta Ca_{pre}$) and a postsynaptic spike ($\Delta Ca_{post}$), keeping their ratio constant, $\Delta Ca_{post} = 2 \cdot \Delta Ca_{pre}$ (see the section "Effect of Kinetics of Autophosphorylation and Dephosphorylation on the Number of Spike-Pair Presentations Needed for Transitions").

**Numerical methods.** We solve the system of coupled, ordinary differential equations with a fourth-order Runge-Kutta method with adaptive stepsize control. This has been implemented in a C++ program. We used XPPAUT by G. Bard Ermentrout (http://www.pitt.edu/~phase/) for the steady-state calculations of the CaMKII system.

### Acknowledgments

We thank David DiGregorio, Paul Miller, and Thomas Stephen Otis for very helpful comments and fruitful discussions. We are indebted to anonymous reviewers for their comments and suggestions which helped to improve the manuscript considerably.

**Author contributions.** NB and MG conceived and designed the model. MG performed the simulations and analyzed the data. MG and NB wrote the paper.

**Funding.** This work has been partly completed during support of MG by a French Government scholarship in conjunction with a DAAD supplement and an Eiffel Doctorate scholarship of the French Government. NB was supported by the French Ministry of Research, ACI Neurosciences intégratives et computationelles.

**Competing interests.** The authors have declared that no competing interests exist.

#### References

1. Markram H, Lübke J, Frotscher M, Sakmann B (1997) Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. Science 275: 213–215.
2. Bi G, Poo M (1998) Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. J Neurosci 18: 10464–10472.
3. Dudek S, Bear M (1992) Homosynaptic long-term depression in area CA1 of hippocampus and effects of N-methyl-D-aspartate receptor blockade. Proc Natl Acad Sci U S A 89: 4363–4367.
4. Sjöström P, Turrigiano G, Nelson S (2001) Rate, timing, and cooperativity jointly determine cortical synaptic plasticity. Neuron 32: 1149–1164.
5. Ngezahayo A, Schachner M, Artola A (2000) Synaptic activity modulates the induction of bidirectional synaptic changes in adult mouse hippocampus. J Neurosci 20: 2451–2458.
6. Gerstner W, Kempter R, van Hemmen JL, Wagner H (1996) A neuronal learning rule for sub-millisecond temporal coding. Nature 383: 76–81.
7. Song S, Miller K, Abbott L (2000) Competitive Hebbian learning through spike-timing–dependent synaptic plasticity. Nat Neurosci 3: 919–926.
8. Senn W, Markram H, Tsodyks M (2001) An algorithm for modifying neurotransmitter release probability based on pre- and postsynaptic spike timing. Neural Comput 13: 35–67.
9. Karmarkar UR, Najarian MT, Buonomano DV (2002) Mechanisms and significance of spike-timing dependent plasticity. Biol Cybern 87: 373–382.
10. Shouval HZ, Bear MF, Cooper LN (2002) A unified model of NMDA receptor-dependent bidirectional synaptic plasticity. Proc Natl Acad Sci U S A 99: 10831–10836.
11. Karbowski J, Ermentrout GB (2002) Synchrony arising from a balanced synaptic plasticity in a network of heterogeneous neural oscillators. Phys Rev E Stat Nonlin Soft Matter Phys 65: 031902.
12. Gerstner W, Kistler WM (2002) Mathematical formulations of hebbian learning. Biol Cybern 87: 404–415.
13. Abarbanel HDI, Gibb L, Huerta R, Rabinovich M (2003) Biophysical model of synaptic plasticity dynamics. Biol Cybern 89: 214–226.
14. Pfister JP, Gerstner W (2006) Triplets of spikes in a model of spike timing-dependent plasticity. J Neurosci 26: 9673–9682.
15. Lisman J (1985) A mechanism for memory storage insensitive to molecular turnover: a bistable autophosphorylating kinase. Proc Natl Acad Sci U S A 82: 3055–3057.
16. Bhalla U, Iyengar R (1999) Emergent properties of networks of biological signaling pathways. Science 283: 381–387.
17. Zhabotinsky AM (2000) Bistability in the Ca(2+)/calmodulin-dependent protein kinase-phosphatase system. Biophys J 79: 2211–2221.
18. Okamoto H, Ichikawa K (2000) Switching characteristics of a model for biochemical-reaction networks describing autophosphorylation versus dephosphorylation of Ca(2+)/calmodulin-dependent protein kinase II. Biol Cybern 82: 35–47.
19. Lisman J (2003) Long-term potentiation: outstanding questions and attempted synthesis. Philos Trans R Soc Lond B Biol Sci 358: 829–842.
20. Petersen C, Malenka R, Nicoll R, Hopfield J (1998) All-or-none potentiation at CA3-CA1 synapses. Proc Natl Acad Sci U S A 95: 4732–4737.
21. O’Connor DH, Wittenberg GM, Wang SSH (2005) Graded bidirectional synaptic plasticity is composed of switch-like unitary events. Proc Natl Acad Sci U S A 102: 9679–9684.
22. Bagal AA, Kao JPY, Tang CM, Thompson SM (2005) Long-term potentiation of exogenous glutamate responses at single dendritic spines. Proc Natl Acad Sci U S A 102: 14434–14439.
23. Fink CC, Meyer T (2002) Molecular mechanisms of CaMKII activation in neuronal plasticity. Curr Opin Neurobiol 12: 293–299.
24. Hanson PI, Schulman H (1992) Neuronal Ca(2+)/calmodulin-dependent protein kinase. Annu Rev Biochem 61: 559–601.
25. Shen K, Meyer T (1999) Dynamic control of CaMKII translocation and localization in hippocampal neurons by NMDA receptor stimulation. Science 284: 162–166.
26. Hayashi Y, Shi SH, Esteban JA, Piccini A, Poncer JC, et al. (2000) Driving AMPA receptors into synapses by LTP and CaMKII: requirement for GluR1 and PDZ domain interaction. Science 287: 2262–2267.
27. Lisman J, Schulman H, Cline H (2002) The molecular basis of CaMKII function in synaptic and behavioural memory. Nat Rev Neurosci 3: 175–190.
28. Colbran RJ (2004) Targeting of calcium/calmodulin-dependent protein kinase II. Biochem J 378: 1–16.
29. Mammen AL, Kameyama K, Roche KW, Huganir RL (1997) Phosphorylation of the alpha-amino-3-hydroxy-5-methylisoxazole4-propionic acid receptor GluR1 subunit by calcium/calmodulin-dependent kinase II. J Biol Chem 272: 32528–32533.
30. Derkach V, Barria A, Soderling T (1999) Ca(2+)/calmodulin-kinase II enhances channel conductance of alpha-amino-3-hydroxy-5-methyl-4-isoxazolepropionate type glutamate receptors. Proc Natl Acad Sci U S A 96: 3269–3274.
31. Giese K, Fedorov N, Filipkowski R, Silva A (1998) Autophosphorylation at Thr286 of the alpha calcium-calmodulin kinase II in LTP and learning. Science 279: 870–873.
32. Bliss TVP, Collingridge GL, Morris RGM (2003) Introduction. Long-term potentiation and structure of the issue. Philos Trans R Soc Lond B Biol Sci 358: 607–611.
33. Irvine EE, von Hertzen LSJ, Plattner F, Giese KP (2006) alphaCaMKII autophosphorylation: a fast track to memory. Trends Neurosci 29: 459–465.
34. Chen HX, Otmakhov N, Strack S, Colbran RJ, Lisman JE (2001) Is persistent activity of calcium/calmodulin-dependent kinase required for the maintenance of LTP? J Neurophysiol 85: 1368–1376.
35. Yang HW, Hu XD, Zhang HM, Xin WJ, Li MT, et al. (2004) Roles of CaMKII, PKA, and PKC in the induction and maintenance of LTP of C-fiber-evoked field potentials in rat spinal dorsal horn. J Neurophysiol 91: 1122–1133.
36. Lengyel I, Voss K, Cammarota M, Bradshaw K, Brent V, et al. (2004) Autonomous activity of CaMKII is only transiently increased following the induction of long-term potentiation in the rat hippocampus. Eur J Neurosci 20: 3063–3072.
37. Mulkey R, Endo S, Shenolikar S, Malenka R (1994) Involvement of a calcineurin/inhibitor-1 phosphatase cascade in hippocampal long-term depression. Nature 369: 486–488.
38. Blitzer RD, Connor JH, Brown GP, Wong T, Shenolikar S, et al. (1998) Gating of CaMKII by cAMP-regulated protein phosphatase activity during LTP. Science 280: 1940–1942.
39. Morishita W, Connor J, Xia H, Quinlan E, Shenolikar S, et al. (2001) Regulation of synaptic strength by protein phosphatase 1. Neuron 32: 1133–1148.
40. Morishita W, Marie H, Malenka RC (2005) Distinct triggering and expression mechanisms underlie LTD of AMPA and NMDA synaptic responses. Nat Neurosci 8: 1043–1050.
41. Cooke SF, Bliss TVP (2006) Plasticity in the human central nervous system. Brain 129: 1659–1673.
42. Hayer A, Bhalla US (2005) Molecular switches at the synapse emerge from receptor and kinase traffic. PLoS Comput Biol 1: 137–54. doi:10.1371/journal.pcbi.0010020

PLoS Computational Biology | www.ploscompbiol.org | 2321 | November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

43. Wang HX, Gerkin RC, Nauen DW, Bi GQ (2005) Coactivation and timing-dependent integration of synaptic potentiation and depression. Nat Neurosci 8: 187–193.
44. Rubin JE, Gerkin RC, Bi GQ, Chow CC (2005) Calcium time course as a signal for spike-timing-dependent plasticity. J Neurophysiol 93: 2600–2613.
45. D’Alcantara P, Schiffmann SN, Swillens S (2003) Bidirectional synaptic plasticity as a consequence of interdependent Ca(2+)-controlled phosphorylation and dephosphorylation pathways. Eur J Neurosci 17: 2521–2528.
46. Castellani GC, Quinlan EM, Bersani F, Cooper LN, Shouval HZ (2005) A model of bidirectional synaptic plasticity: from signaling network to channel conductance. Learn Mem 12: 423–432.
47. McNeill RB, Colbran RJ (1995) Interaction of autophosphorylated Ca<sup>2+</sup>/calmodulin-dependent protein kinase II with neuronal cytoskeletal proteins. characterization of binding to a 190-kDa postsynaptic density protein. J Biol Chem 270: 10043–10049.
48. Bradshaw JM, Hudmon A, Schulman H (2002) Chemical quenched flow kinetic studies indicate an intraholoenzyme autophosphorylation mechanism for Ca<sup>2+</sup>/calmodulin-dependent protein kinase II. J Biol Chem 277: 20991–20998.
49. Hudmon A, Schulman H (2002) Neuronal Ca(2+)/calmodulin-dependent protein kinase II: the role of structure and autoregulation in cellular function. Annu Rev Biochem 71: 473–510.
50. Rosenberg OS, Deindl S, Sung RJ, Nairn AC, Kuriyan J (2005) Structure of the autoinhibited kinase domain of CaMKII and SAXS analysis of the holoenzyme. Cell 123: 849–860.
51. Bell C, Han V, Sugawara Y, Grant K (1997) Synaptic plasticity in a cerebellum-like structure depends on temporal order. Nature 387: 278–281.
52. Magee J, Johnston D (1997) A synaptically controlled, associative signal for Hebbian plasticity in hippocampal neurons. Science 275: 209–213.
53. Debanne D, Gähwiler B, Thompson S (1998) Long-term synaptic plasticity between pairs of individual CA3 pyramidal cells in rat hippocampal slice cultures. J Physiol 507 (Part 1): 237–247.
54. Sabatini BL, Oertner TG, Svoboda K (2002) The life cycle of Ca(2+) ions in dendritic spines. Neuron 33: 439–452.
55. Nevian T, Sakmann B (2006) Spine Ca<sup>2+</sup> signaling in spike-timing-dependent plasticity. J Neurosci 26: 11001–11013.
56. Petersen JD, Chen X, Vinade L, Dosemeci A, Lisman JE, et al. (2003) Distribution of postsynaptic density (PSD)-95 and Ca<sup>2+</sup>/calmodulin-dependent protein kinase II at the PSD. J Neurosci 23: 11270–11278.
57. Miller P, Zhabotinsky AM, Lisman JE, Wang XJ (2005) The stability of a stochastic CaMKII switch: dependence on the number of enzyme molecules and protein turnover. PLoS Biol 3: e107. doi:10.1371/journal.pbio.0030107
58. Mainen Z, Malinow R, Svoboda K (1999) Synaptic calcium transients in single spines indicate that NMDA receptors are not saturated. Nature 399: 151–155.
59. Sabatini B, Svoboda K (2000) Analysis of calcium channels in single spines using optical fluctuation analysis. Nature 408: 589–593.
60. Malleret G, Haditsch U, Genoux D, Jones M, Bliss T, et al. (2001) Inducible and reversible enhancement of learning, memory, and long-term potentiation by genetic inhibition of calcineurin. Cell 104: 675–686.
61. Lisman J (1989) A mechanism for the Hebb and the anti-Hebb processes underlying learning and memory. Proc Natl Acad Sci U S A 86: 9574–9578.
62. Shen K, Teruel MN, Connor JH, Shenolikar S, Meyer T (2000) Molecular memory by reversible translocation of calcium/calmodulin-dependent protein kinase II. Nat Neurosci 3: 881–886.
63. Sharma K, Fong DK, Craig AM (2006) Postsynaptic protein mobility in dendritic spines: long-term regulation by synaptic NMDA receptor activation. Mol Cell Neurosci 31: 702–712.
64. Zhuo M, Zhang W, Son H, Mansuy I, Sobel RA, et al. (1999) A selective role of calcineurin A-alpha in synaptic depotentiation in hippocampus. Proc Natl Acad Sci U S A 96: 4650–4655.
65. Lee H, Barbarosie M, Kameyama K, Bear M, Huganir R (2000) Regulation of distinct AMPA receptor phosphorylation sites during bidirectional synaptic plasticity. Nature 405: 955–959.
66. Jouvenceau A, Billard JM, Haditsch U, Mansuy IM, Dutar P (2003) Different phosphatase-dependent mechanisms mediate long-term depression and depotentiation of long-term potentiation in mouse hippocampal CA1 area. Eur J Neurosci 18: 1279–1285.
67. O’Connor DH, Wittenberg GM, Wang SSH (2005) Dissection of bidirectional synaptic plasticity into saturable unidirectional processes. J Neurophysiol 94: 1565–1573.
68. Jay T, Gurden H, Yamaguchi T (1998) Rapid increase in PKA activity during long-term potentiation in the hippocampal afferent fibre system to the prefrontal cortex in vivo. Eur J Neurosci 10: 3302–3306.
69. Cooper D, Mons N, Karpen J (1995) Adenylyl cyclases and the interaction between calcium and cAMP signalling. Nature 374: 421–424.
70. Cohen P, Cohen PT (1989) Protein phosphatases come of age. J Biol Chem 264: 21435–21438.
71. Shenolikar S (1994) Protein serine/threonine phosphatases—new avenues for cell regulation. Annu Rev Cell Biol 10: 55–86.
72. Svenningsson P, Nishi A, Fisone G, Girault JA, Nairn AC, et al. (2004) DARPP-32: an integrator of neurotransmission. Annu Rev Pharmacol Toxicol 44: 269–296.
73. Strack S, Choi S, Lovinger DM, Colbran RJ (1997) Translocation of autophosphorylated calcium/calmodulin-dependent protein kinase II to the postsynaptic density. J Biol Chem 272: 13467–13470.
74. Huang C, Liang Y, Hsu K (2001) Characterization of the mechanism underlying the reversal of long term potentiation by low frequency stimulation at hippocampal CA1 synapses. J Biol Chem 276: 48108–48117.
75. Cho K, Aggleton JP, Brown MW, Bashir ZI (2001) An experimental test of the role of postsynaptic calcium levels in determining synaptic strength using perirhinal cortex of rat. J Physiol 532: 459–466.
76. Lisman J (2001) Three Ca(2+) levels affect plasticity differently: the LTP zone, the LTD zone and no man’s land. J Physiol 532: 285.
77. Kitajima T, Hara K (2000) A generalized Hebbian rule for activity-dependent synaptic modifications. Neural Netw 13: 445–454.
78. Abarbanel HDI, Talathi SS, Gibb L, Rabinovich M (2005) Synaptic plasticity with discrete state synapses. Phys Rev E Stat Nonlin Soft Matter Phys 72: 031914.
79. Yang S, Tang Y, Zucker R (1999) Selective induction of LTP and LTD by postsynaptic [Ca<sup>2+</sup>]i elevation. J Neurophysiol 81: 781–787.
80. Zucker RS (1999) Calcium- and activity-dependent synaptic plasticity. Curr Opin Neurobiol 9: 305–313.
81. Mizuno T, Kanazawa I, Sakurai M (2001) Differential induction of LTP and LTD is not determined solely by instantaneous calcium concentration: an essential involvement of a temporal factor. Eur J Neurosci 14: 701–708.
82. Ismailov I, Kalikulov D, Inoue T, Friedlander MJ (2004) The kinetic profile of intracellular calcium predicts long-term potentiation and long-term depression. J Neurosci 24: 9847–9861.
83. Shouval HZ, Kalantzis G (2005) Stochastic properties of synaptic transmission affect the shape of spike time-dependent plasticity curves. J Neurophysiol 93: 1069–1073.
84. Bi GQ, Wang HX (2002) Temporal asymmetry in spike timing-dependent synaptic plasticity. Physiol Behav 77: 551–555.
85. Froemke RC, Dan Y (2002) Spike-timing-dependent synaptic modification induced by natural spike trains. Nature 416: 433–438.
86. Froemke RC, Tsay IA, Raad M, Long JD, Dan Y (2006) Contribution of individual spikes in burst-induced long-term synaptic modification. J Neurophysiol 95: 1620–1629.
87. Froemke RC, Poo MM, Dan Y (2005) Spike-timing-dependent synaptic plasticity depends on dendritic location. Nature 434: 221–225.
88. Wittenberg GM, Wang SSH (2006) Malleability of spike-timing-dependent plasticity at the CA3-CA1 synapse. J Neurosci 26: 6610–6617.
89. Gerkin RC, Lau PM, Nauen DW, Wang YT, Bi GQ (2007) Modular competition driven by NMDA receptor subtypes in spike-timing-dependent plasticity. J Neurophysiol 97: 2851–2862.
90. Bender VA, Bender KJ, Brasier DJ, Feldman DE (2006) Two coincidence detectors for spike timing-dependent plasticity in somatosensory cortex. J Neurosci 26: 4166–4177.
91. Sjöström PJ, Häusser M (2006) A cooperative switch determines the sign of synaptic plasticity in distal dendrites of neocortical pyramidal neurons. Neuron 51: 227–238.
92. Cai Y, Gavornik JP, Cooper LN, Yeung LC, Shouval HZ (2007) Effect of stochastic synaptic and dendritic dynamics on synaptic plasticity in visual cortex and hippocampus. J Neurophysiol 97: 375–386.
93. Sajikumar S, Navakkode S, Frey JU (2007) Identification of compartment- and process-specific molecules required for "synaptic tagging" during long-term potentiation and long-term depression in hippocampal CA1. J Neurosci 27: 5068–5080.
94. Normann C, Peckys D, Schulze CH, Walden J, Jonas P, et al. (2000) Associative long-term depression in the hippocampus is dependent on postsynaptic N-type Ca<sup>2+</sup> channels. J Neurosci 20: 8290–8297.
95. Watkins J, Collingridge G (1994) Phenylglycine derivatives as antagonists of metabotropic glutamate receptors. Trends Pharmacol Sci 15: 333–342.
96. Nicoletti F, Bruno V, Copani A, Casabona G, Knöpfel T (1996) Metabotropic glutamate receptors: a new target for the therapy of neurodegenerative disorders? Trends Neurosci 19: 267–271.
97. Kemp N, Bashir ZI (2001) Long-term depression: a cascade of induction and expression mechanisms. Prog Neurobiol 65: 339–365.
98. Chin D, Means A (2000) Calmodulin: a prototypical calcium sensor. Trends Cell Biol 10: 322–328.
99. Shifman JM, Choi MH, Mihalas S, Mayo SL, Kennedy MB (2006) Ca(2+)/calmodulin-dependent protein kinase II (CaMKII) is activated by calmodulin with two bound calciums. Proc Natl Acad Sci U S A 103: 13968–13973.
100. Linse S, Helmersson A, Forsén S (1991) Calcium binding to calmodulin and its globular domains. J Biol Chem 266: 8050–8054.
101. Klee C (1988) Calmodulin. Amsterdam: Elsevier. pp. 35–56.
102. Meyer T, Hanson PI, Stryer L, Schulman H (1992) Calmodulin trapping by calcium-calmodulin-dependent protein kinase. Science 256: 1199–1202.
103. Persechini A, Stemmer PM, Ohashi I (1996) Localization of unique functional determinants in the calmodulin lobes to individual EF hands. J Biol Chem 271: 32217–32225.
104. Persechini A, Stemmer PM (2002) Calmodulin is a limiting factor in the cell. Trends Cardiovasc Med 12: 32–37.
105. Zhabotinsky AM, Camp RN, Epstein IR, Lisman JE (2006) Role of the

PLoS Computational Biology | www.ploscompbiol.org 2322 November 2007 | Volume 3 | Issue 11 | e221

STDP in a Bistable Synapse Model

neurogranin concentrated in spines in the induction of long-term potentiation. J Neurosci 26: 7337–7347.
106. Coomber C (1998) Site-selective autophosphorylation of Ca(2+)/calmodulin-dependent protein kinase II as a synaptic encoding mechanism. Neural Comput 10: 1653–1678.
107. Holmes WR (2000) Models of calmodulin trapping and CaM kinase II activation in a dendritic spine. J Comput Neurosci 8: 66–85.
108. Kubota Y, Bower JM (2001) Transient versus asymptotic dynamics of CaM kinase II: possible roles of phosphatase. J Comput Neurosci 11: 263–279.
109. Dupont G, Houart G, de Koninck P (2003) Sensitivity of CaM kinase II to the frequency of Ca(2+) oscillations: a simple model. Cell Calcium 34: 485–497.
110. Michaelis L, Menten M (1913) Die Kinetik der Invertinwirkung. Biochem Z 49: 333–369.
111. Oliver C, Shenolikar S (1998) Physiologic importance of protein phosphatase inhibitors. Front Biosci 3: D961–D972.
112. Munton RP, Vizi S, Mansuy IM (2004) The role of protein phosphatase-1 in the modulation of synaptic and structural plasticity. FEBS Lett 567: 121–128.
113. Stemmer P, Klee C (1994) Dual calcium ion regulation of calcineurin by calmodulin and calcineurin B. Biochemistry 33: 6859–6866.
114. Bollmann J, Helmchen F, Borst J, Sakmann B (1998) Postsynaptic Ca2+ influx mediated by three different pathways during synaptic transmission at a calyx-type synapse. J Neurosci 18: 10409–10419.
115. Helmchen F, Imoto K, Sakmann B (1996) Ca2+ buffering and action potential-evoked Ca(2+) signaling in dendrites of pyramidal neurons. Biophys J 70: 1069–1081.
116. Burnashev N, Zhou Z, Neher E, Sakmann B (1995) Fractional calcium currents through recombinant GluR channels of the NMDA, AMPA and kainate receptor subtypes. J Physiol 485 (Part 2): 403–418.
117. Schneggenburger R (1996) Simultaneous measurement of Ca(2+) influx and reversal potentials in recombinant N-methyl-D-aspartate receptor channels. Biophys J 70: 2165–2174.
118. Poirazi P, Brannon T, Mel BW (2003) Arithmetic of subthreshold synaptic summation in a model CA1 pyramidal cell. Neuron 37: 977–987.
119. Purvis LK, Butera RJ (2005) Ionic current model of a hypoglossal motoneuron. J Neurophysiol 93: 723–733.
120. Destexhe A, Mainen ZF, Sejnowski TJ (1998) Methods in neuronal modelling. Cambridge (Massachusetts): MIT Press.
121. Wang X (1999) Synaptic basis of cortical persistent activity: the importance of NMDA receptors to working memory. J Neurosci 19: 9587–9603.
122. Jahr C, Stevens C (1990) A quantitative description of NMDA receptor-channel kinetic behavior. J Neurosci 10: 1830–1837.
123. Koninck PD, Schulman H (1998) Sensitivity of CaM kinase II to the frequency of Ca(2+) oscillations. Science 279: 227–230.
124. Bhalla US, Iyengar R (2002) DOQCS database. http://doqcs.ncbs.res.in/
125. Endo S, Zhou X, Connor J, Wang B, Shenolikar S (1996) Multiple structural elements define the specificity of recombinant human inhibitor-1 as a protein phosphatase-1 inhibitor. Biochemistry 35: 5220–5228.
126. Kennedy M (2000) Signal-processing machines at the postsynaptic density. Science 290: 750–754.
127. Nimchinsky EA, Yasuda R, Oertner TG, Svoboda K (2004) The number of glutamate receptors opened by synaptic stimulation in single hippocampal spines. J Neurosci 24: 2054–2064.

PLoS Computational Biology | www.ploscompbiol.org 2323 November 2007 | Volume 3 | Issue 11 | e221