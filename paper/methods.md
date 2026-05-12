# Methods

## A. Problem Formulation

A motor-imagery (MI) trial is a multivariate EEG segment $X \in \mathbb{R}^{C \times N}$ with $C$ channels, $N$ samples, and class label $y \in \{1,\ldots,K\}$. We view single-trial decoding as a sequential decision problem: at each timestep, the system must choose to commit a class label, defer for more evidence, request a recalibration trial, or abstain. Let the augmented action set be $\mathcal{A} = \{1,\ldots,K\} \cup \{\text{DEFER}, \text{RECAL}, \text{ABSTAIN}\}$.

We discretize the trial into windows of length $L=1.0$~s with stride $\Delta=0.25$~s, yielding $T = \lceil(N/f_s - L)/\Delta\rceil + 1$ window indices per trial (e.g., $T=13$ for a 4~s trial at 250~Hz). The per-trial Markov decision process (MDP) is defined as follows:

**State.** $s_t$ is the concatenation of (i) a window encoding $\phi(X_t) \in \mathbb{R}^{D}$ from a frozen encoder $\phi$; (ii) the running mean and variance of $\phi(X_{1{:}t})$; (iii) elapsed time $t/(T-1)$; (iv) a recalibration flag $r_t \in \{0,1\}$; and (v) a $D_s$-dim subject embedding $e_{\text{subj}}$. Total $\dim(s_t) = 3D + 2 + D_s$.

**Transitions.** Deterministic given the trial recording. DEFER advances $t \mapsto t+1$. Commit and ABSTAIN terminate. RECAL pays a fixed time cost $\tau_r$ (advancing $t$ by $\lceil \tau_r/\Delta \rceil$ steps) in exchange for replacing $e_{\text{subj}}$ with a per-subject embedding refit from a held-out calibration buffer; $r_t \mapsto 1$ and RECAL becomes unavailable.

**Reward.** Let $\lambda$ be a per-second time-cost coefficient. Rewards are
$$
r_t = \begin{cases}
+R^+ - \lambda t \Delta & \text{correct commit} \\
-R^- - \lambda t \Delta & \text{wrong commit}\\
-\lambda \Delta & \text{DEFER} \\
-R_{\text{recal}} & \text{REQUEST\_RECAL} \\
-R_a - \lambda t \Delta & \text{ABSTAIN}
\end{cases}
$$
with $R^- > R_a > 0$ so that abstention is preferred to a likely-wrong commit. Defaults: $R^+ = 1, R^- = 5, R_a = 0.5, R_{\text{recal}} = 0.3, \lambda = 0.05/\text{s}$, swept in a sensitivity analysis (Table~III).

## B. Architecture

**Encoder.** We use the EEGNet backbone of \cite{Lawhern2018} (parameters: $F_1=8$, $D=2$, kernel lengths 64 and 16, dropout 0.25; $\sim$1.5K parameters), supervised-pretrained on training-trial 1-second windows. The encoder is frozen for the agent. (The intended foundation-model backbone is LaBraM \cite{LaBraM2024}; we report the EEGNet stand-in here and the LaBraM swap as Experiment~3.)

**Policy/critic networks.** A 3-layer MLP (hidden 256, ReLU) maps the $\sim$354-dim state to per-action Q-values. For risk-sensitive variants the critic instead outputs $M=31$ quantile values per action; the action-selection rule replaces the mean with $\mathrm{CVaR}_{\alpha}$, the average of the lowest $\lceil \alpha M \rceil$ quantiles.

**Subject embedding.** A fixed Gaussian random projection of dimension $D_s = 16$ applied to the mean encoder feature over a per-subject set of trials. The default embedding uses the subject's training trials; the post-recal embedding uses a held-out calibration buffer ($10\%$ of trials per class).

## C. Offline RL Training

We use Conservative Q-Learning \cite{Kumar2020CQL}. The training loss has three terms:

1. **Bellman residual** (Huber):  $\mathcal{L}_\text{TD} = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\big[\rho_\delta(Q_\theta(s,a) - y_{\text{tgt}})\big]$ with $y_{\text{tgt}} = r + \gamma (1-d) \max_{a'} Q_{\bar\theta}(s', a')$ and Polyak-averaged target network $\bar\theta$.
2. **Conservative penalty:** $\mathcal{L}_\text{CQL} = \alpha_\text{CQL}\big(\mathrm{logsumexp}_a Q_\theta(s,a) - Q_\theta(s,a_{\text{data}})\big)$.
3. **CMDP safety penalty:** with Lagrange multiplier $\beta = e^{\log\beta}$ and tolerance $\epsilon$, $\mathcal{L}_\text{CMDP} = \beta \cdot \mathbb{E}_{(s,a)\sim \mathcal{D}}\big[\mathbb{1}\{a \text{ is wrong commit}\}\, Q_\theta(s, a)\big]$ pulls down the value of state-action pairs whose data outcome was a wrong commit. The dual update on $\beta$ is $\nabla_{\log\beta} \mathcal{L} = -\beta(\bar w - \epsilon)$ where $\bar w$ is the empirical wrong-commit rate.

The risk-sensitive variant (CVaR-CQL) replaces Q with the quantile-Huber regression of QR-DQN \cite{Dabney2018QR-DQN} and uses $\mathrm{CVaR}_\alpha$ in policy evaluation, while keeping the mean for the Bellman target.

**Optimizer.** AdamW, $\text{lr}_{\text{critic}} = 3 \times 10^{-4}$, weight decay $10^{-4}$, gradient clip $1.0$, batch 256, polyak $\tau = 5\times 10^{-3}$, $\gamma=0.99$. We run 100K gradient steps in the full sweep and 10–20K in pilot experiments.

## D. Behavior Policies and Offline Buffer

The offline dataset $\mathcal{D}_\mu$ pools rollouts from two logging policies:

* **$\mu_1$ — fixed-window:** DEFER until a fixed step $T_{\text{fix}}$, then commit using a logistic-regression classifier on the running-mean encoder feature. We sweep $T_{\text{fix}} \in \{4, 8, 12\}$ strides. Action propensity is 1 (deterministic), suitable for FQE but degenerate for IS-based estimators.
* **$\mu_2$ — SPRT-stochastic:** at each step, DEFER unless the classifier's max-class probability exceeds an evidence threshold $\tau_\text{evid} = 0.55$ or $t = T-1$. Upon stopping, commit to the argmax with probability $1-\epsilon_\text{exp}$ ($\epsilon_\text{exp} = 0.10$) or uniformly otherwise. This yields a non-degenerate behavior log-probability $\log \mu_2(a|s)$ logged per step, enabling PDIS and DR.

## E. Off-Policy Evaluation

For a candidate policy $\pi$, we estimate $V(\pi) = \mathbb{E}_\pi[\sum_t \gamma^t r_t]$ from $\mathcal{D}_\mu$ alone using three estimators:

* **Per-Decision Importance Sampling (PDIS), weighted variant.** $\hat V_\text{PDIS} = \frac{1}{N}\sum_n \sum_t \gamma^t \tilde w_{n,t} r_{n,t}$ where $\tilde w_{n,t} = \prod_{t' \le t} \pi(a_{t'}|s_{t'})/\mu(a_{t'}|s_{t'})$ normalized per timestep across trajectories.
* **Fitted Q-Evaluation (FQE).** A separate Q-network $\hat Q_\text{FQE}$ trained per target policy via the Bellman equation under $\pi$: $\hat Q(s,a) \leftarrow r + \gamma\, \mathbb{E}_{a' \sim \pi(s')}[\hat Q(s', a')]$. The estimate is $\hat V_\text{FQE}(\pi) = \mathbb{E}_{s_0}\big[\sum_a \pi(a|s_0)\, \hat Q(s_0, a)\big]$. FQE does not require behavior propensity and is robust to deterministic $\mu_1$ data.
* **Doubly-Robust (DR).** Combines PDIS with FQE as a control variate: $\hat V_\text{DR} = \mathbb{E}_n\big[\hat V_\text{FQE}(s_0^n) + \sum_t \gamma^t \tilde w_{n,t}\big(r_{n,t} + \gamma \hat V_\text{FQE}(s'_{n,t}) - \hat Q_\text{FQE}(s_{n,t}, a_{n,t})\big)\big]$.

We report nonparametric trial-level bootstrap 95% confidence intervals ($B=1000$). Critically, we **validate OPE estimates against on-policy Monte-Carlo ground truth** computed by simulating $\pi$ on the deterministic per-trial MDP: across a panel of $\ge 30$ candidate policies (mixtures of trained CQL agents at various conservatism, temperature, and random-mixing settings), we plot OPE estimate versus $V_\text{GT}$ and report Pearson correlation, RMSE, and CI coverage.

## F. Datasets and Preprocessing

All datasets are accessed via the MOABB \cite{MOABB2018} library:
* BCI Competition IV-2a (BNCI2014\_001): 9 subjects, 4-class MI, 22 EEG channels, $\sim$576 trials/subject \cite{Tangermann2012BCI4}.
* BCI Competition IV-2b (BNCI2014\_004): 9 subjects, 2-class MI, 3 channels, $\sim$720 trials/subject.
* Lee2019 MI: 54 subjects, 2-class MI, 62 channels.

**Preprocessing.** Resample to 250~Hz, band-pass 4--40~Hz (zero-phase Butterworth, order 4), common-average reference, conversion to microvolts, per-channel z-scoring across the recording. Optional ICLabel-based ocular component removal is included in the pipeline but disabled by default during pilot experiments and re-enabled for the full sweep.

**Splitting.** Trial-level 70/15/15 train/val/test for within-subject experiments; leave-one-subject-out for cross-subject experiments. The encoder is supervised-pretrained on training trials only — the validation and test splits are never seen during encoder training. A per-subject calibration buffer (10\% of trials per class, drawn from the train split) is reserved for the RECAL action.

## G. Baselines

We compare against (i) FBCSP+LDA at fixed window \cite{Ang2008FBCSP}; (ii) EEGNet at fixed window \cite{Lawhern2018}; (iii) EEG-Conformer at fixed window \cite{Song2023EEGConformer}; (iv) EEGNet with classifier-confidence threshold dynamic stopping; (v) Bianchi-Liti Bayesian early stopping \cite{Bianchi2023}; (vi) BTSPRT \cite{Liu2017BTSPRT}; (vii) MarkovType-style POMDP REINFORCE \cite{Rezaei2024MarkovType} ported from RSVP to MI; (viii) EEG\_RL-Net Dueling DQN \cite{Atefi2024EEG_RL_Net}; (ix) Random and (x) Oracle bounds. All baselines share the same preprocessing, splits, and feature backbone where applicable.

## H. Evaluation Metrics

* **Information Transfer Rate (ITR)** in bits/min via the Wolpaw formula:
$$
B = \log_2 K + a \log_2 a + (1-a) \log_2 \frac{1-a}{K-1},\quad
\text{ITR} = B \cdot \frac{60}{\bar t_\text{decision}}
$$
where $a$ is commit accuracy and $\bar t_\text{decision}$ is the mean wall time per decision (including any RECAL cost).
* **Commit accuracy** (over commit episodes only).
* **Mean and median decision latency.**
* **REQUEST\_RECAL and ABSTAIN rates.**
* **CVaR$_\alpha$(error cost):** the lower-tail mean of the per-episode error-cost distribution.

The headline NeuroPolicy artifact is the *learned Pareto frontier* over (ITR, accuracy, latency, recal-rate, abstain-rate) traced by sweeping $\alpha_\text{CVaR}$, $\epsilon_\text{CMDP}$, and $\lambda$.
