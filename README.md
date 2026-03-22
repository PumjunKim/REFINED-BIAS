# REFINED-BIAS: A Disentangled Benchmark and Framework for Integrated Evaluation of Shape/Texture Bias in Neural Networks


## Abstract
Understanding how neural networks rely on visual cues offers a humaninterpretable view of their internal decision processes. The cue-conflict benchmark
has been influential in probing shape-texture preference and in motivating the
insight that stronger, human-like shape bias is often associated with improved
in-domain performance. However, we find that the current stylization-based instantiation can yield unstable and ambiguous bias estimates. Specifically, stylization
may not reliably instantiate perceptually valid and separable cues nor control their
relative informativeness, ratio-based bias can obscure absolute cue sensitivity,
and restricting evaluation to preselected classes can distort model predictions by
ignoring the full decision space. Together, these factors can confound preference with cue validity, cue balance, and recognizability artifacts. We introduce
REFINED-BIAS, an integrated dataset and evaluation framework for reliable and
interpretable shape–texture bias diagnosis. REFINED-BIAS constructs balanced,
human- and model- recognizable cue pairs using explicit definitions of shape and
texture, and measures cue-specific sensitivity over the full label space via a rankingbased metric, enabling fairer cross-model comparisons. Across diverse training
regimes and architectures, REFINED-BIAS enables fairer cross-model comparison,
more faithful diagnosis of shape and texture biases, and clearer empirical conclusions, resolving inconsistencies that prior cue-conflict evaluations could not
reliably disambiguate. Our code is publicly available at REFINED-BIAS.

## How to reproduce our results
### Environmental Setup 
```
conda create --name refined-bias python=3.8.20 -y
conda activate refined-bias
pip install -r ./requirements.txt
```

### Dataset
You can evaluate our REFINED-BIAS shape cue & Texture cue in `./datasets/refined_bias_shape` and `./datasets/refined_bias_texture`. The dataset structure is shown as:
```
datasets
├── refined_bias_shape
│   ├── balloon
│       ├── balloon_0.png
│       ├── balloon_1.png
│           ...
│   ├── book
│   ...
└── refined_bias_texture
│   ├── brain_coral
│       ├── 4x4_brain_coral_0.png
│       ├── 4x4_brain_coral_1.png
│           ...
│   ├── texutre
    ...
* shape : (3, 224, 224) 
```
### Evaluation
To evaluate REFINED-BIAS across different model architectures and learning strategies, simply run the following command. The necessary checkpoints will be fetched and downloaded automatically.
```
# evaluate across model architecture on REFINED-BIAS shape cue
python eval_refined_bias.py --dataset refined_bias_shape --across arch

# evaluate across model architecture on REFINED-BIAS texture cue
python eval_refined_bias.py --dataset refined_bias_texture --across arch

# evaluate across learning strategy on REFINED-BIAS shape cue
python eval_refined_bias.py --dataset refined_bias_shape --across strategy

# evaluate across learning strategy on REFINED-BIAS texture cue
python eval_refined_bias.py --dataset refined_bias_texture --across strategy

```
The above command will print out as shown in the example below:
```
REFINED-BIAS Shape Bias on arch
> bagnet9               : 0.0518
> bagnet17              : 0.0988
> bagnet33              : 0.2438
> ...
```
Detailed per-class scores for each model and learning strategy can be found in the .json files located under `./results/across_model_architecture` and `./results/across_learning_strategy`.
