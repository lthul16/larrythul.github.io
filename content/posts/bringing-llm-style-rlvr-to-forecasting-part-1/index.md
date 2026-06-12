---
title: "Bringing LLM-Style RLVR to Time-Series Forecasting (Part 1)"
date: 2026-06-08
math: true
description: "Building a market forecaster from first principles as a discrete token policy, so the reinforcement learning that post-trains language models can post-train it to seek alpha."
tags: ["forecasting", "reinforcement-learning", "transformers", "post-training"]
ShowToc: true
---

I've spent a lot of time in my career building time series forecasting algorithms. I've used
hierarchical forecasting with trees,
XGBoost, Chronos, PatchTST, and the rest of the usual toolkit. They work, up to a point. But
every time I looked at where machine learning was actually compounding, the answer was the
same. It was language models. The entire research stack, the tooling, the scaling laws, the
reinforcement-learning machinery that turns a pretrained model into something that reasons,
all of it was being poured into one architecture.

So I kept returning to one question. What would it take to forecast markets with
that architecture, not as an analogy, but literally? Not "transformers for time series,"
which already exists, but a model that is a decoder-only language model in every structural
sense, where the only thing that changes is the alphabet. If I could pull that off, then
every tool built for LLMs would apply to my problem for free, including the part I care about
most: post-training a policy with reinforcement learning against a verifiable reward.

This post is me reasoning through that design from the ground up. I will start from what a
forecaster has to do, show where the usual recipe stops short, and let each fix introduce
exactly one piece of the architecture. By the end we will have something that looks like an
LLM, ingests hundreds of correlated instruments at once, and produces probabilistic
forecasts by sampling. Then I will show you where we actually are with it, honestly, on data
the model never trained on, and where this is all heading: using RL to turn the thing into an
alpha-seeking policy.

![A language model consuming word-piece tokens beside the same decoder-only transformer consuming signal token groups; identical backbone, different alphabet](fig_llm_parallel.png)

The whole bet in one picture. A decoder-only language model and an autoregressive forecaster
are the same machine: a causal transformer predicting the next symbol from the past. The
backbone, the mask, and the softmax head are shared. Only the alphabet changes, from
word-pieces to tokenized market signal.

---

## What a forecaster actually has to do

Strip the problem down. I observe a price series, or a few hundred of them, and I want to say
something useful about the future. Three requirements survive that stripping, and each one
will force a design decision later, so I want them on the table first.

I need a distribution, not a number. A point estimate of tomorrow's return is close to
useless on its own, because the entire game is the shape of the cone of outcomes and where
its mass sits. Whatever the model emits, I have to be able to compute quantiles, intervals,
and calibration from it.

I need to use many series at once. Assets move together. A forecast for one name that
ignores the hundred correlated names trading beside it throws away most of the information in
the tape. The model has to read a cross-section.

I need no look-ahead, ever. The moment a forecast sizes a position, any leakage of future
information becomes unrecoverable and quietly inflates every backtest I run. Causality has to
be enforced by the architecture, not by my good intentions.

Hold those three. The rest of the post earns them one at a time.

---

## Where the standard recipe stops short

The obvious starting point is the recipe patch-based time-series transformers already use,
the PatchTST and Chronos lineage. Chop each series into fixed-length patches, project each
patch through a linear map into the model dimension, run a transformer, attach a
regression or Gaussian head. It works. It is also where my "literally an LLM" constraint
starts to bite, in two places.

The first problem is the input. A linear patch embedding is a fixed, content-blind basis.
Every patch, whether it is a sleepy drift or a once-a-year volatility shock, goes through the
same projection and gets the same share of representational budget. A language model does the
opposite. Its tokenizer spends short codes on common pieces and keeps expressive structure
for rare ones, and crucially the embedding table is learned, so representation follows the
data. I want that for signal shapes. I want a quiet five-day drift and a gap-up to be
genuinely different tokens, not two points in the same linear subspace.

The second problem is the output. A regression head gives a point. A Gaussian head gives
a parametric interval. Neither gives me a distribution I can sample arbitrary trajectories
from, and more importantly neither is a policy. There is no finite set of actions, each
with an exact probability attached. I am going to want exactly that later, when I post-train
with RL, so I refuse the continuous head now rather than rip it out after the fact.

Both problems point the same way. Replace the linear patch map with a learned, discrete
vocabulary of signal shapes, and keep everything downstream discrete. That one decision is
the heart of the model, so it gets the longest section.

---

## The core move: a learned alphabet for market signal

I want to turn a patch of a series into a token, the way a tokenizer turns a chunk of text
into a word-piece. Doing that on raw returns fails for a reason worth stating, because the
fix is the first real piece of the architecture.

### Normalize first, so one vocabulary can span every instrument

A five-day window of ten-year-yield returns and a five-day window of crypto returns differ in
volatility by one to two orders of magnitude. A single codebook fit on both would waste most
of its capacity encoding scale rather than shape. So I factor scale out before quantizing.

Let $P_t$ be the price and $r_t = \log(P_t / P_{t-1})$ the log return. I divide by a
trailing, lagged volatility estimate $\hat{\sigma}_t$, an exponentially weighted moving
average over returns strictly before $t$, to get a vol-normalized return

$$
x_t \;=\; \frac{r_t}{\hat{\sigma}_t + \epsilon}.
$$

That puts a bond and a coin on the same footing. Then I z-score the patch itself. With patch
mean $\mu$ and patch standard deviation $\sigma$ taken over the normalized values, the
shape is

$$
\hat{x} \;=\; \frac{x - \mu}{\sigma + \epsilon},
$$

a roughly unit-scale curve that means the same thing wherever it came from. The two scalars I
peeled off, $\mu$ and $\sigma$, are not thrown away. I will re-emit them as tokens in a
moment, because anything I normalize away has to come back as a prediction target if I ever
want to forecast it. De-normalization at decode time just inverts the chain,

$$
r \;=\; \big(\hat{x}\,\sigma + \mu\big)\,\hat{\sigma}_t .
$$

One discipline runs through all of this and it is non-negotiable: every statistic uses only
data at or before its own patch. The vol estimate is lagged, the patch stats are local,
nothing is fit on data that postdates what it touches. That is the no-look-ahead requirement
showing up at the very first step, and getting the inverse of this chain exactly right turned
out to matter more than I expected, which I will come back to in the results.

![One raw patch flowing through vol-normalize, z-score, and a four-codebook residual VQ into the token group sigma-mu-c1-c2-c3-c4](fig_pipeline.png)

Tokenizing one patch. Scale is divided out first so a single codebook serves every
instrument, the level $\mu,\sigma$ is peeled off, and the residual VQ turns the remaining
shape into code indices.

### Quantize the shape with a learned codebook, not a projection

Now I turn the shape $\hat{x}$ into discrete codes. The naive version snaps it to the nearest
entry in a single codebook, but one nearest-neighbor lookup is a blunt instrument. Real
patches are blends, a trend with noise riding on it, and the closest single template throws
the blend away.

So I use Residual Vector Quantization. Keep a stack of $K$ codebooks, each of size $V$,
with embedding vectors $e_k[\cdot]$. The first codebook captures the coarse shape, I subtract
its reconstruction, and pass the residual to the next codebook, which refines it, and so on
through all $K$ levels:

$$
\hat{x} \;\approx\; \sum_{k=1}^{K} e_k[c_k], \qquad c_k \in \{1, \dots, V\}.
$$

The shape becomes an ordered tuple $(c_1, \dots, c_K)$, coarse to fine. This buys an effective
vocabulary of $V^K$ distinct shapes while storing only $K \cdot V$ embedding rows, and it
represents blends naturally, because each level corrects what the previous one missed. I train
it end to end with a straight-through estimator and a commitment loss, plus the usual VQ
hygiene to stop the codebook collapsing: EMA code updates, dead-code revival, and a
utilization target.

I can watch the refinement happen on a real patch. This one takes a volatile five-day window
and rebuilds it one codebook at a time. The first code lands a coarse approximation, and each
level chips away the residual, so the reconstruction error falls monotonically.

![Residual VQ rebuilding a real patch coarse-to-fine, with reconstruction error falling at each of four levels](fig_residual_refinement.png)

Four levels of $V=256$ codes take one shock patch from NMSE 0.135 to 0.006. Coarse first,
then residual refinements.

Once trained, the codebook is a dictionary of recurring shapes the data taught it, not a
basis I imposed. I can look at the dictionary directly by decoding each code back into the
five-day curve it stands for. These are the most frequently used first-level atoms across
tens of thousands of real patches, and they read like an alphabet of elementary moves: ramps,
reversals, single-day pops, quiet drifts.

![A grid of learned level-1 codebook atoms, each a five-day shape, annotated with how often real patches select it](fig_atoms_level1.png)

The 24 most-used first-level atoms, ranked by selection frequency over 67,233 real patches.
Each panel is one learned shape.

The test that matters is whether real patches survive the round trip, especially the ones a
forecaster cannot afford to blur. They do. Here are three patches a model has to get right, a
clean trend, a choppy reversal, and a volatility shock, each decoded back from its four codes.

![Three real five-day patches — trend, chop, shock — each overlaid with its four-code RVQ reconstruction](fig_reconstructions.png)

Real patches (solid) and their tokenized reconstructions (dashed). The bracketed integers
are the four code indices; titles report reconstruction NMSE, all around 0.006.

### The levels become tokens too

I still have the two scalars I peeled off, the patch drift $\mu$ and the patch volatility
$\sigma$. I could feed them back as raw numbers, but that reintroduces a continuous quantity
into a model I am keeping discrete, and a scalar magnitude multiplying a vector tends to
collapse the representation onto a single ray. So I bin them and emit them as their own
tokens, each with its own embedding row. The volatility token is binned as a vol
innovation, realized patch vol relative to the trailing estimate, which sits near one and is
far more stationary across instruments than absolute vol. The drift token bins $\mu$ on a
symmetric grid around zero.

A patch of one instrument at one time now serializes to a fixed-length token group:

$$
\big[\,\sigma\,\big]\ \big[\,\mu\,\big]\ \big[\,c_1\,\big]\ \big[\,c_2\,\big]\ \cdots\ \big[\,c_K\,\big].
$$

The analogy is exact. A patch is a word, and its token group is the spelling of that word in
an alphabet of scales and shapes. Everything the model reads or writes is now a token from a
finite vocabulary, which is the property I set out to preserve.

---

## Hundreds of series in one sequence

I can tokenize one patch. A forecaster that reads one series at a time still fails the second
requirement, so now I lay every instrument into a single sequence.

Every instrument is just a stream id, and I flatten all of them into one sequence of token
groups sorted by $(\text{timestamp}, \text{stream})$. This is an any-variate layout. A
series that trades rarely, or has a gap, simply contributes no tokens where it has no
observation. No imputation grid, no per-channel padding. A weekly series drops in one group a
week, slotted into the timeline beside the daily names. Because every stream lives in the same
sequence, attention is what correlates them: a token at time $t$ can attend to the tokens of
any stream at earlier timestamps. Cross-asset structure gets learned by the same mechanism
that learns time. I deliberately do not split the streams into independent channels, because
the relationships between assets are the most valuable thing in the data.

That leaves the causal rule, the one genuinely subtle part, and it is where the third
requirement gets enforced. The mask is block-causal by timestamp. Writing $\mathrm{ts}$,
$\mathrm{stream}$, and $\mathrm{slot}$ for a token's timestamp, stream, and position within
its group, query $a$ may attend key $b$ iff

$$
\mathrm{ts}_b < \mathrm{ts}_a
\;\;\lor\;\;
\big(\mathrm{ts}_b = \mathrm{ts}_a \,\land\, \mathrm{stream}_b = \mathrm{stream}_a \,\land\, \mathrm{slot}_b < \mathrm{slot}_a\big)
\;\;\lor\;\;
a = b .
$$

A token attends everything strictly earlier in time on any stream, and earlier slots inside
its own group so that $c_2$ sees $c_1$. Tokens that share a timestamp but belong to different
streams do not attend to each other while they are being predicted. They are contemporaneous,
not causal predecessors. Once realized they become ordinary context for everything at $t+1$
and later. I give up same-bar cross-asset information at prediction time, and I take that
trade on purpose, because a lost contemporaneous signal is a known limitation while look-ahead
bias is an unrecoverable error.

![The block-causal attention matrix for a two-stream, three-timestamp sequence; allowed cells shaded, the same-bar cross-stream cell marked denied](fig_mask.png)

The actual mask, computed on a tiny two-stream by three-timestamp example. Green is allowed.
The highlighted cell is the one rule that costs us something: at the same timestamp, stream A
may not look at stream B.

Positions and identity follow LLM practice with one adjustment for the two-dimensional layout.
Rotary embeddings run over the time axis, and a learned instrument embedding acts as a
segment marker so the model knows which stream a token belongs to. Calendar features, the
signal family, and the volatility regime enter as additive conditioning embeddings, exactly
how you would condition an LLM on side information. One generalization comes almost for free:
non-price signals, event probabilities or physical-flow volumes, ride in as extra streams with
their own per-family tokenizers, masked out of the loss so they inform forecasts without ever
becoming forecast targets.

---

## Predicting the next group, and reading a forecast off samples

With the inputs settled the output side is almost entirely standard. The model predicts the
next token group autoregressively with a single softmax head over the unified vocabulary, tied
to the input embeddings, just like an LLM. The only addition is a grammar: the slot a token
occupies inside its group decides which tokens are legal, and the sampler masks the rest. The
group factorizes into a short autoregressive chain of its own, nested inside the chain over
time:

$$
p(\text{group}_t \mid \text{context}) \;=\; p(\sigma)\;p(\mu \mid \sigma)\;\prod_{k=1}^{K} p\big(c_k \mid \sigma, \mu, c_{\lt k}\big).
$$

Training is the causal language-modeling loss you would expect, cross-entropy on the predicted
token slots. One honest note from pretraining, because it looks like a bug and is not: the
drift token's head stays close to uninformative. Daily drift is tiny next to daily volatility,
which is just market efficiency asserting itself, so the model learns that predicting drift
sharply is mostly a way to be wrong. I leave that head honest for now. Sharpening it is a job
for later, and it is exactly the job I want RL to do.

I do not emit an interval. I sample. From a shared context I draw $G$ rollouts at
temperature $\tau$, decode each group back to returns through the inverse of the normalization
chain, and read quantiles, coverage, and any scoring rule off the empirical ensemble of
trajectories. This is the Chronos approach, and it is the only output scheme that is at once
fully probabilistic, directly calibratable, and a legitimate token policy with exact per-token
log-probabilities. It is also cheap, for the same reason LLM serving is cheap: one prefill of
the shared context, then fork the KV cache $G$ ways and decode in parallel.

---

## Where we actually are

That is the architecture. Now the honest part, because a clean design is worth nothing until
it survives contact with held-out data. Everything below is on the project's fixed walk-forward
eval folds, windows the model never trained on, with a thirty-day embargo between the training
cutoff and the evaluation window.

Start with tokenization, on the hardest data I have. This is palladium futures through
February to April 2020, deep in the held-out COVID window and well after the model's
January-2020 cutoff. The market never seen, a violent crash included.

![45 held-out days of palladium futures through the COVID crash, the tokenized reconstruction tracking the raw path including a -23% day, and the token strip of nine patches](fig_sequence_tokenized.png)

A held-out window strung together as tokens. Panel (a) is the raw series, (b) overlays the
reconstruction decoded from the tokens, and (c) is the token sequence itself. The production
group carries one extra slot, a tail-event "surprise" token $s$, which fires on the −23% crash
day where the move clears four sigma, recording the extreme that the clipped shape codes
cannot. The vol tokens track the elevated volatility throughout.

The reconstruction tracks the crash, and the tail-event token fires exactly where it should.
The tokenizer generalizes. The harder question is forecasting, so let me show you the case it
gets wrong first, on purpose. Same held-out COVID window, now forecasting the last two patches.

![Sampled forecast quantile bands against the realized COVID path, where a +19% spike blows through the cone, beside the forecasted and realized token groups](fig_sequence_forecast.png)

Forecasting the held-out window. The model gets the regime right, its sampled vol tokens sit
in the high-volatility bins, but it cannot call the specific +19% one-day spike, which blows
clean through the cone before the path settles back inside it. The median drift stays near
zero and the band widens rather than chasing the move. That is the honest limit: at the
extreme, the size of a single move is not forecastable.

I picked the crash because it is the hard case, but the everyday case is what decides whether
the forecaster is useful, and there I have many more examples. Here are held-out forecasts for
nine different instruments across the calm 2017 fold, the quietest stretch in the data.

![A three-by-three gallery of held-out calm-fold forecasts on nine instruments, where the quantile bands cover the realized paths, with per-panel coverage shown](fig_forecast_gallery.png)

Nine held-out forecasts, one instrument per panel, from the calm eval fold. Black is the
context tail, blue the quantile bands and median, orange what actually happened. Across forty
held-out windows, the model's 80% interval covers the realized return about 88% of the time.

So where does that leave us. The tokenizer round-trips cleanly on held-out data, including
crashes. The forecaster is calibrated in normal conditions, with its 80% band covering the
realized path about 88% of the time, erring mildly conservative rather than overconfident. And
it is honest about its limit: it flags a high-volatility regime through the vol token, but it
will not pretend to predict the direction and size of a one-day shock, because nothing should.

I want to be precise about what this is and is not. This is a calibration result, not yet a
skill-beats-baseline result. The forecast median sits near zero, like a last-value baseline,
because daily drift is barely forecastable from cross-entropy alone. The model's edge so far is
in volatility and in the shape of the distribution, not in calling direction. Getting even this
far required some unglamorous correctness work, including catching a decode bug where the
inverse normalization applied the volatility scaling twice and quietly crushed every sampled
amplitude. The lesson I keep relearning is that in this domain the boring inversion details are
where the alpha leaks out. Calling direction with an edge is the next problem, and it is not a
pretraining problem.

---

## Where this is going: RL for alpha

Here is why I built the whole thing discrete, every step of the way. Because every emitted
token comes from a frozen finite vocabulary with an exact softmax probability, the model is a
discrete autoregressive policy

$$
\pi_\theta(\text{token} \mid \text{context}),
$$

with exact per-token log-probabilities. That is the precise interface modern
reinforcement-learning-from-verifiable-rewards methods expect, the GRPO and GSPO family that
post-trains large language models. They apply here without modification, and the reason I am
confident they fit, rather than being borrowed on faith, comes down to two facts about
markets.

The first is that the verifier is free. I do not need a learned reward model or a simulator.
The realized future is the ground truth, revealed after the fact. An episode is a date $t$
and a universe of instruments, a rollout is a sampled set of forecast trajectories for the next
$H$ steps, and the reward is a deterministic function of those trajectories and what actually
happened. The market scores the policy.

The second is the elegant part, and it is what makes group-relative methods the right tool. I
sample a group of $G$ rollouts from the same context, and every rollout in that group faces the
same realized future. GRPO forms a group-relative advantage,

$$
\hat{A}_i \;=\; \frac{R_i - \bar{R}}{\mathrm{std}(R)}, \qquad \bar{R} = \frac{1}{G}\sum_{j=1}^{G} R_j .
$$

Now suppose the reward decomposes into a common market move $m$ shared by every rollout plus a
rollout-specific skill term, $R_i = m + s_i$. The market term is identical across the group, so
it cancels in the centering:

$$
\hat{A}_i \;\propto\; R_i - \bar{R} \;=\; (m + s_i) - \big(m + \bar{s}\big) \;=\; s_i - \bar{s}.
$$

The common market shock drops out of the advantage. What survives is the difference in
positioning between rollouts on the same realized world, which is relative forecast skill. The
group baseline subtracts beta and leaves alpha. That is not a trick I added; it falls straight
out of the cross-sectional, same-future structure of the episode.

To make sure the policy is actually rewarded for selection skill and not for riding a factor, I
never let forecasts become positions freely. A fixed, deliberately simple, non-learnable sizing
rule maps each rollout's cross-sectional forecasts to a dollar-neutral long-short book: rank the
predicted drifts, go long the top names and short the bottom, vol-scaled, with $\sum_i w_i = 0$.
A dollar-neutral book has roughly zero beta, so riding equity or commodity beta earns nothing,
and the only way to score is cross-sectional ranking skill. The headline reward is the rank
information coefficient, the Spearman correlation between the rollout's predicted drift ranking
and the realized residualized return ranking, with returns residualized against a small factor
model first so "long momentum" cannot masquerade as selection. A pinball-loss anchor keeps the
forecasts honest and calibrated, and a small, winsorized, cost-aware portfolio term gets added
last.

Why bother with RL at all when cross-entropy already optimizes shape, differentiably and far
more sample-efficiently? Because the objective I actually care about cannot be backpropagated
token by token. Alpha is a property of the ranking across the cross-section against a
residualized future, with transaction costs and factor-exposure penalties, evaluated on
realized data the model has to forecast first. That is a verifiable reward, not a differentiable
loss, and it is exactly the regime where RL pays for its sample cost. There is even a clean
statistical reason to expect it to converge here where per-name PnL never would, Grinold's
fundamental law,

$$
\mathrm{IR} \;\approx\; \mathrm{IC}\cdot\sqrt{\text{breadth}},
$$

which says the information ratio scales with skill times the square root of the number of
independent bets. One cross-sectional episode over a few hundred names is a few hundred noisy
bets averaged together. The market noise that would swamp a single-name reward gets averaged
down, and a small, honest cross-sectional IC turns into a usable signal.

The drift head that looked uninformative after pretraining is the thing RL is meant to sharpen.
Its entropy is the metric I will watch most closely, because the failure mode here is a policy
that earns the calibration anchor by predicting volatility and zero drift everywhere, and
quietly stops taking directional bets. That is exactly why the rank-IC term exists: it pays only
for drift discrimination. The whole point of keeping the model a plain token policy is that the
RL trainers built for LLMs, the verl and TRL and OpenRLHF stacks, plug in as-is, with a custom
environment that serves held-out context windows and scores rollouts against the reserved future
already sitting in the data.

---

## The dictionary

Read the model back through the LLM it was built to imitate.

| Language model | This forecaster |
|---|---|
| word-piece token | learned signal-shape code (RVQ) plus scale and drift tokens |
| BPE tokenizer | frozen, versioned RVQ tokenizer |
| embedding table | shared token embeddings plus family / instrument / calendar / vol conditioning |
| one stream of tokens | $(\text{timestamp}, \text{stream})$-sorted token groups, any-variate |
| causal mask | block-causal by timestamp, strict no-look-ahead |
| next-token softmax | next-token-group softmax with a per-slot grammar |
| temperature-sampled completions | sampled forecast trajectories scored as an ensemble |
| KV-cache decoding | KV-cache-forked group sampling, $G$ rollouts from one prefill |
| RLHF / RLVR | RLVR with the realized market as the verifier |

I did not invent a bespoke architecture and then build tooling for it. I borrowed the most
heavily engineered architecture in machine learning and changed only the alphabet. The discipline
that bought, a frozen action space and exact log-probabilities, is the thing that makes the next
chapter possible: a forecaster I can post-train against the market itself, where a group baseline
cancels beta and what is left to optimize is alpha. The tokenizer generalizes, the forecasts are
calibrated on held-out data, and the hard, interesting work, turning calibration into ranked
selection skill, is where we are pointed next. I will write that part up when the reward curves
have something honest to say.

---

Part 2 is coming soon, with experimental results from post-training this policy with RLVR to
seek alpha.
