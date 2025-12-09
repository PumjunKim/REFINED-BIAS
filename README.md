# REFINED-BIAS: a **RE**liable **F**ramework for **IN**tegrated **E**valuation and **D**isentangled **B**enchmark of **I**nterpretable **A**lignment of **S**hape/texture in neural networks

Pum Jun Kim, Seung-Ah Lee, Seongho Park, [Dongyoon Han](https://scholar.google.com/citations?user=jcP7m1QAAAAJ&hl=ko&oi=ao), [Jaejun Yoo](https://scholar.google.co.kr/citations?hl=en&user=7NBlQw4AAAAJ)

[Paper]

## Abstract
> Understanding how neural networks utilize visual cues provides a human-interpretable perspective
on their internal decision processes. Building on this motivation, the cue-conflict benchmark has initiated
important progress in bridging human and model perception. However, despite its value, it falls short of meeting
the necessary conditions for a precise bias analysis: (1) it relies on stylized images that blend shape and texture cues,
blurring their distinction and offering no control over the relative contribution of each cue; (2) limiting evaluation
to preselected classes distorts model predictions based on cues; and (3) the cue-conflict metric fails to distinguish
models that genuinely utilize the cues. Collectively, these limitations hinder an accurate interpretation of model bias.
To address this, we introduce REFINED-BIAS, a diagnostic benchmark provides refined and more accurate measurements. 
REFINED-BIAS generates artifact-free samples while preserving human defined shape and texture as faithfully as possible,
and quantifies cue sensitivity across the full label space using Mean Reciprocal Rank, enabling a fairer cross-model comparisons.
Extensive evaluations across diverse training regimes and architectures demonstrate that REFINED-BIAS not only provides a more
accurate assessment of shape and texture biases than prior benchmark, but also reveals new insights into how models utilize cues,
clarifying previously inconsistent findings.


