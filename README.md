# On the Reliabilty of Cue Conflict and Beyond

[Pum Jun Kim](https://scholar.google.com/citations?user=WGJgXskAAAAJ&hl=ko&oi=ao)$^{\dagger}$, [Seung-Ah Lee](https://scholar.google.com/citations?hl=ko&user=lbdS3joAAAAJ) $^{\dagger}$, Seongho Park, [Dongyoon Han](https://scholar.google.com/citations?user=jcP7m1QAAAAJ&hl=ko&oi=ao), [Jaejun Yoo](https://scholar.google.co.kr/citations?hl=en&user=7NBlQw4AAAAJ)

[Paper] | [Project Page](https://pumjunkim.github.io/REFINED-BIAS/) | **Quick Start**: [REFINED-BIAS](https://github.com/PumjunKim/REFINED-BIAS#-evaluating-refined-bias) | [Colab]

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

## Overview of REFINED-BIAS Dataset
![REFINED-BIAS_Dataset](https://github.com/PumjunKim/REFINED-BIAS/blob/main/figures/overview.png?raw=true)

We (1) define disentangled stimuli based on human perception rather than model-derived heuristics, 
ensuring that each cue carries pure and interpretable information, and
(2) select classes suited for bias evaluation and generate data to maximize the predictive strength 
of both shape and texture cues, thereby balancing cue informativeness.

# How shape/texture bias metric of REFINED-BIAS is defined?
Our metric computes the reciprocal ranks of the correct shape and texture labels within the model's full prediction ranking.
We refer to these two components as $$\text{RB}_S$$ for shape and $$\text{RB}_T$$ for texture. Note that unlike conventional MRR,
our ranking is computed over the logits:

$$\text{RB}_S=\frac{1}{N}\sum^N_{i=1}\frac{1}{r_{\text{shape},i}},\quad \text{RB}_T=\frac{1}{N}\sum^N_{i=1}\frac{1}{r_{\text{texture},i}}$$

Here, $$N$$ is the total number of samples, $$r_{\text{shape},i}$$ and $$r_{\text{texture},i}$$ are the ranks of the correct shape and texture
labels for the $$i$$-th sample in the model's ranked predictions, respectively. The relative bias for shape and texture is written as:

$$\text{RB}^{rel}_S=\frac{\text{RB}_S}{(\text{RB}_S+\text{RB}_T)},\quad \text{RB}^{rel}_T=\frac{\text{RB}_T}{(\text{RB}_S+\text{RB}_T)}$$

## 🚀 Evaluating via REFINED-BIAS

To evaluate **REFINED-BIAS** across different **model architectures** or **learning strategies**, run the commands below.  
All required checkpoints will be **downloaded automatically**.

## Installation

```python
conda create --name refined python=3.8.20 -y
conda activate refined
pip install -r ./requirements.txt
```

---

## Dataset
<details>
<summary><b> Dataset Structure</b></summary>

<br>

<pre>

datasets/
├── refined_bias_shape/
│   ├── balloon/
│   │   ├── balloon_0.png
│   │   ├── balloon_1.png
│   │   └── ...
│   ├── book/
│   └── ...
└── refined_bias_texture/
    ├── brain_coral/
    │   ├── 4x4_brain_coral_0.png
    │   ├── 4x4_brain_coral_1.png
    │   └── ...
    ├── texture/
    └── ...

</pre>

**Image size**: (3, 224, 224)

</details>

---

### 🔍 Evaluate Model Architectures
```bash
# REFINED-BIAS Shape Cue
python eval_refined_bias.py --dataset refined_bias_shape --across arch

# REFINED-BIAS Texture Cue
python eval_refined_bias.py --dataset refined_bias_texture --across arch
```

---

### ⚙️ Evaluate Learning Strategies
```bash
# REFINED-BIAS Shape Cue
python eval_refined_bias.py --dataset refined_bias_shape --across strategy

# REFINED-BIAS Texture Cue
python eval_refined_bias.py --dataset refined_bias_texture --across strategy
```

---

### 📄 Example Output
```text
REFINED-BIAS Shape Bias (across: arch)
  • bagnet9   : 0.0518
  • bagnet17  : 0.0988
  • bagnet33  : 0.2438
  • ...
```

Detailed per-class scores for each model and learning strategy can be found in the `.json` files located under:
- `./results/across_model_architecture`
- `./results/across_learning_strategy`
