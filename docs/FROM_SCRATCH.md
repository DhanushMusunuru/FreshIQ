# FreshIQ from first principles

Everything under the hood, with the actual arithmetic. If you read this properly
you could rebuild the whole system without the code.

Worked examples use **your real numbers** from the training run wherever possible.

**Contents**

1. [An image is a grid of numbers](#1-an-image-is-a-grid-of-numbers)
2. [The neuron](#2-the-neuron)
3. [Why not just connect everything](#3-why-not-just-connect-everything)
4. [Convolution, worked by hand](#4-convolution-worked-by-hand)
5. [Activation functions](#5-activation-functions)
6. [Stride, and the shrinking grid](#6-stride-and-the-shrinking-grid)
7. [Batch normalisation](#7-batch-normalisation)
8. [Depthwise separable convolution](#8-depthwise-separable-convolution)
9. [The inverted residual block](#9-the-inverted-residual-block)
10. [The width multiplier](#10-the-width-multiplier)
11. [Global average pooling](#11-global-average-pooling)
12. [Softmax](#12-softmax)
13. [Cross-entropy](#13-cross-entropy)
14. [Gradient descent](#14-gradient-descent)
15. [Backpropagation, worked by hand](#15-backpropagation-worked-by-hand)
16. [Adam](#16-adam)
17. [Transfer learning](#17-transfer-learning)
18. [Two heads, one loss](#18-two-heads-one-loss)
19. [Stopping it memorising](#19-stopping-it-memorising)
20. [Splits, and the leak that would have fooled you](#20-splits-and-the-leak-that-would-have-fooled-you)
21. [Class imbalance](#21-class-imbalance)
22. [Measuring it honestly](#22-measuring-it-honestly)
23. [Quantisation](#23-quantisation)
24. [Rebuilding it from nothing](#24-rebuilding-it-from-nothing)

---

## 1. An image is a grid of numbers

A 128×128 colour photo is **49,152 numbers**: 128 × 128 × 3, one per pixel per
colour channel, each 0–255.

```
pixel (0,0) = [R=210, G=40, B=40]     a red pixel
pixel (0,1) = [R=205, G=38, B=42]
...
```

We divide everything by 255 so values land in **0.0 – 1.0**. This is
**normalising**, and it matters because of what happens in section 14: gradients
are proportional to inputs, so feeding in numbers around 200 produces updates
200× larger than feeding in numbers around 1. Training either explodes or has to
crawl.

MobileNet specifically wants **−1.0 to +1.0**, so the model's first layer does:

```
x_scaled = 2·x − 1        0.0 → −1.0,  0.5 → 0.0,  1.0 → +1.0
```

That is the `Rescaling(2.0, offset=-1.0)` layer. It lives **inside** the model so
it can never be forgotten at inference time.

---

## 2. The neuron

One artificial neuron:

```
y = f( w₁x₁ + w₂x₂ + ... + wₙxₙ + b )
```

- `x` — inputs
- `w` — **weights**, the adjustable numbers
- `b` — **bias**, lets the neuron fire even when all inputs are zero
- `f` — **activation**, a non-linear squash

**Worked example.** Three inputs, three weights:

```
x = [0.5, 0.2, 0.9]
w = [1.0, -2.0, 0.5]
b = 0.1

sum = (0.5)(1.0) + (0.2)(−2.0) + (0.9)(0.5) + 0.1
    = 0.5 − 0.4 + 0.45 + 0.1
    = 0.65

y = ReLU(0.65) = 0.65
```

That is the entire computation a neural network does, repeated a few hundred
million times.

**Why the activation is mandatory.** Without `f`, stacking layers is pointless:

```
layer1: y = W₁x
layer2: z = W₂y = W₂(W₁x) = (W₂W₁)x = W₃x
```

Two linear layers collapse into one. A hundred linear layers collapse into one.
**Non-linearity is the only reason depth buys you anything.**

---

## 3. Why not just connect everything

Suppose you connect every pixel to a modest 1,000-neuron layer.

```
weights = 128 × 128 × 3 × 1000 = 49,152,000
```

**49 million weights, for one layer.** Your entire model is 446,076.

There is a worse problem than size. A fully-connected layer learns each pixel
position independently. Teach it that mould at position (10,10) means rot, and it
has learned *nothing* about mould at (60,60) — a completely different weight.

Convolution fixes both at once.

---

## 4. Convolution, worked by hand

A **kernel** is a small grid of weights, usually 3×3. Lay it over a patch, multiply
element by element, sum. That one number is one output pixel. Slide, repeat.

**Worked example.** Patch and kernel:

```
patch          kernel (a vertical-edge detector)
18  92  40     −1   0   +1
31  88  55     −2   0   +2
22  79  37     −1   0   +1
```

Multiply and sum:

```
(18)(−1) + (92)(0) + (40)(+1)  =  −18 +  0 + 40  =  22
(31)(−2) + (88)(0) + (55)(+2)  =  −62 +  0 +110  =  48
(22)(−1) + (79)(0) + (37)(+1)  =  −22 +  0 + 37  =  15
                                              sum  =  85
```

Output pixel = **85**. A large positive value means "bright on the right, dark on
the left" — a vertical edge.

**Nobody wrote those nine numbers.** Training discovered them. That is the whole
point: you specify the *shape* of the computation, and gradient descent fills in
the values.

### The saving

Those 9 weights are reused at **every position** in the image. So:

| | Weights |
|---|---|
| Fully connected, 128×128×3 → 1000 | 49,152,000 |
| One 3×3 conv kernel | **9** |
| 32 such kernels over 3 channels | 3 × 3 × 3 × 32 = **864** |

And because the same kernel slides everywhere, a pattern learned in one place is
detected everywhere. That property is called **translation equivariance**.

### The cost formula

For a `K×K` kernel, `M` input channels, `N` output channels, over a `D×D` grid:

```
multiply-accumulates = K · K · M · N · D · D
```

Remember this. Section 8 attacks it.

---

## 5. Activation functions

MobileNetV2 uses **ReLU6**:

```
ReLU6(x) = min( max(x, 0), 6 )
```

```
x = −3.0  →  0.0        negatives clipped away
x =  0.65 →  0.65       positives pass through
x =  9.2  →  6.0        capped at 6
```

**Why zero out negatives?** It makes the network sparse — many neurons output
exactly 0 for any given input, so each one specialises. It is also trivially
cheap, and its derivative is either 0 or 1, which keeps gradients well behaved.

**Why cap at 6?** Look ahead to section 23. In 8-bit arithmetic you have 256
possible values. If activations can reach 1000, one step is ~4 units and you lose
all fine detail. Capping at 6 means one step is 6/255 ≈ 0.024. **ReLU6 exists
specifically so the model quantises well.** A design decision made years before
your Pi existed.

---

## 6. Stride, and the shrinking grid

**Stride** is how far the kernel jumps. Stride 1 examines every position; stride 2
skips every other one and **halves the output grid**.

MobileNetV2 has 5 stride-2 stages, so the grid shrinks by **2⁵ = 32**:

```
128 → 64 → 32 → 16 → 8 → 4
```

Which is exactly why 128 input gives a 4×4 final grid. Your log confirmed it:
`tensor<?x4x4x1280>`.

Two things happen as the grid shrinks:

1. **Cost falls.** Cost is proportional to `D·D`; halving `D` quarters the work.
2. **Receptive field grows.** A 3×3 kernel at layer 1 sees 3×3 original pixels.
   By layer 10, after several halvings, each value summarises a large region.
   That is how the network progresses from *edges* to *textures* to *"this patch
   is fuzzy and desaturated"*.

The channel count rises as the grid falls — trading spatial detail for semantic
richness.

---

## 7. Batch normalisation

After each convolution, MobileNetV2 normalises the outputs across the batch:

```
        x − μ
x̂ = ───────────── ,      y = γ·x̂ + β
      √(σ² + ε)
```

- `μ`, `σ²` — mean and variance of that channel across the batch
- `γ`, `β` — learned scale and shift, so the network can undo it if it wants
- `ε` — a tiny constant (~1e-3) so you never divide by zero

**Why.** As training changes early layers, the distribution of inputs to later
layers shifts under them, and they spend their time chasing a moving target.
Normalising pins the distribution, which lets you use a much higher learning rate
and converge far faster.

At inference there is no batch, so it uses running averages collected during
training. This is why `model.predict()` and `model(x, training=True)` can give
different answers — and why `base(x, training=False)` appears in the code.

---

## 8. Depthwise separable convolution

**MobileNet's central idea.** A standard convolution does two jobs at once: it
mixes across *space* (the 3×3 neighbourhood) and across *channels*. Split them.

**Step 1 — depthwise.** One 3×3 kernel per input channel, applied independently.
Space only, no channel mixing.

```
cost = K · K · M · D · D
```

**Step 2 — pointwise.** A 1×1 convolution. Channel mixing only, no spatial extent.

```
cost = M · N · D · D
```

### The derivation

```
standard    = K·K·M·N·D·D

separable   = K·K·M·D·D  +  M·N·D·D

separable     K·K·M·D·D + M·N·D·D
─────────  =  ─────────────────────
standard         K·K·M·N·D·D

              K·K·M·D·D            M·N·D·D
           =  ───────────    +   ───────────
              K·K·M·N·D·D        K·K·M·N·D·D

                1         1
           =   ───   +   ───
                N        K·K
```

With `K = 3` and `N` reasonably large, `1/N` is negligible:

```
ratio ≈ 1/9  →  about NINE TIMES less arithmetic
```

**Concrete.** `D=8`, `M=32`, `N=64`, `K=3`:

```
standard  = 3·3·32·64·8·8 = 1,179,648 MACs
depthwise = 3·3·32·8·8    =    18,432
pointwise = 32·64·8·8     =   131,072
separable total           =   149,504

1,179,648 / 149,504 = 7.9× cheaper
```

That one substitution is the difference between a model that runs on a battery
and one that does not.

---

## 9. The inverted residual block

MobileNetV2's second idea. The block that appears 17 times in your model:

```
input  (D×D×M)
   │
   ├─ 1×1 conv  → expand M to 6M   + BN + ReLU6
   │
   ├─ 3×3 depthwise (maybe stride 2)  + BN + ReLU6
   │
   ├─ 1×1 conv  → project 6M to N     + BN, NO ACTIVATION
   │
   └─ + input   (only if stride 1 and M == N)
```

### Why "inverted"

A classic ResNet bottleneck goes **wide → narrow → wide**. This goes
**narrow → wide → narrow**, expanding 6× internally.

The reason is memory. The tensors flowing *between* blocks are the narrow ones,
so the largest thing you must hold at once stays small. The wide part exists only
inside one block and can be computed in chunks. On a device with limited RAM this
is what makes the model fit.

### Why the last layer has no activation — the "linear bottleneck"

The projection back down to `N` channels deliberately has **no ReLU**.

ReLU destroys information: everything negative becomes exactly 0, unrecoverably.
In a wide layer that is survivable — the same information is probably encoded
redundantly across many channels. In a **narrow** layer there is no redundancy,
and zeroing negatives destroys the representation outright.

The MobileNetV2 authors measured this. Removing that ReLU improved accuracy. It
looks like a bug in the code; it is the whole point.

---

## 10. The width multiplier

**α scales every channel count**, then rounds to a multiple of 8 (aligned sizes
are faster). Your model uses α = 0.35.

| Stage | α=1.0 | ×0.35 | rounded | your tensor at 128px |
|---|---|---|---|---|
| first conv | 32 | 11.2 | 16 | `64×64×16` ✓ |
| block 1 | 16 | 5.6 | 8 | `64×64×8` |
| block 2 | 24 | 8.4 | 8 | `32×32×8` |
| block 3 | 32 | 11.2 | 16 | `16×16×16` |
| block 4 | 64 | 22.4 | 24 | `8×8×24` |
| block 5 | 96 | 33.6 | 32 | `8×8×32` |
| block 6 | 160 | 56.0 | 56 | `4×4×56` |
| block 7 | 320 | 112.0 | 112 | `4×4×112` |
| final 1×1 | 1280 | — | 1280 | `4×4×1280` ✓ |

The ticked rows are shapes that appeared verbatim in your training log.

**Cost scales roughly as α².** A pointwise conv costs `M·N·D·D`, and both `M` and
`N` shrink by α, so cost drops by α² ≈ 0.1225 — about **8× cheaper** than α=1.0.

**Note the last row.** Keras does not scale the final 1×1 when α ≤ 1, so it
expands 112 → 1280:

```
112 × 1280 = 143,360 weights
```

Out of ~410,000 in the base — **over a third of the network in one layer**. If you
ever need to go smaller, cut that before cutting α again.

---

## 11. Global average pooling

The body outputs `4×4×1280`. The heads need a flat vector. Two options:

**Flatten:** 4·4·1280 = 20,480 values → a Dense layer to 2 classes costs
20,480 × 2 = 40,962 weights, and it would learn position-specific rules.

**Global average pool:** average each 4×4 map down to one number.

```
feature map 7 (4×4):        mean = (0.2+0.8+0.1+0.0
  0.2  0.8  0.1  0.0               +0.5+0.9+0.3+0.1
  0.5  0.9  0.3  0.1               +0.0+0.4+0.7+0.2
  0.0  0.4  0.7  0.2               +0.1+0.1+0.6+0.3) / 16
  0.1  0.1  0.6  0.3        = 5.3 / 16 = 0.331
```

1280 maps → **1280 numbers**. Head cost: 1280 × 2 = 2,562 weights. **16× cheaper**,
zero parameters in the pooling itself.

It also throws away *where* and keeps only *how much*, which for this task is
exactly right: mould is mould wherever it sits on the tomato.

---

## 12. Softmax

The final Dense layer emits arbitrary reals called **logits**. Softmax turns them
into probabilities.

```
              e^(zᵢ)
ŷᵢ  =  ─────────────────
          Σⱼ e^(zⱼ)
```

**Worked example.** Logits `z = [2.0, 1.0]`:

```
e^2.0 = 7.389
e^1.0 = 2.718
sum   = 10.107

ŷ_fresh   = 7.389 / 10.107 = 0.731
ŷ_spoiled = 2.718 / 10.107 = 0.269
                             ─────
                             1.000  ✓
```

**Why exponentiate?** Two reasons: it forces everything positive (probabilities
cannot be negative), and it **exaggerates differences**. A logit gap of 1.0 became
a probability ratio of 2.7:1. Bigger gaps become much more decisive.

**In practice** the max is subtracted first — `e^(zᵢ − max z)` — because `e^1000`
overflows to infinity. The result is mathematically identical since the constant
cancels. You will find exactly that in `ai/multihead.py`:

```python
ex = np.exp(raw - raw.max())
raw = ex / ex.sum()
```

---

## 13. Cross-entropy

The loss. How wrong are we?

```
L = − Σᵢ yᵢ · log(ŷᵢ)
```

With a one-hot target only one `yᵢ` is 1, so the whole sum collapses:

```
L = − log( ŷ_correct )
```

**Worked table:**

| ŷ for the correct class | L = −ln(ŷ) | reading |
|---|---|---|
| 0.99 | 0.010 | confident and right — almost free |
| 0.73 | 0.315 | right, not certain |
| 0.50 | 0.693 | coin flip |
| 0.10 | 2.303 | confident and wrong — punished |
| 0.01 | 4.605 | very confident, very wrong |
| 0.001 | 6.908 | → grows without bound |

**Why the logarithm.** The penalty is unbounded as confidence in a wrong answer
rises. This is exactly the pressure you want: the model is pushed hardest where it
is most badly, most confidently wrong. A squared-error loss would saturate and
stop caring.

**Its derivative is beautifully simple.** For softmax + cross-entropy together:

```
∂L/∂zᵢ  =  ŷᵢ − yᵢ
```

Predicted minus actual. That is *it*. This clean form is precisely why softmax and
cross-entropy are always paired.

**Check:** predicted `[0.731, 0.269]`, true `[1, 0]`:

```
∂L/∂z_fresh   = 0.731 − 1 = −0.269    push this logit UP
∂L/∂z_spoiled = 0.269 − 0 = +0.269    push this logit DOWN
```

---

## 14. Gradient descent

We have a loss. We need to reduce it by changing ~446,076 weights.

For each weight, `∂L/∂w` says how much the loss changes per unit change in that
weight. Move **against** it:

```
w ← w − η · ∂L/∂w
```

`η` is the **learning rate** — your `LR_FROZEN = 1e-3`.

**Worked example.**

```
w = 0.400,  ∂L/∂w = +2.0,  η = 0.001

w_new = 0.400 − (0.001)(2.0) = 0.398
```

The gradient was positive (increasing `w` increases loss), so `w` decreased.

**Why the learning rate is delicate:**

| η | What happens |
|---|---|
| too large | overshoots the minimum, bounces, diverges to NaN |
| too small | takes forever, or stalls in a bad spot |
| 1e-3 | sensible for fresh random head weights |
| 1e-4 | for fine-tuning — pretrained weights are already good, nudge gently |

This is exactly why stage 2 uses a 10× smaller rate. Fine-tuning at 1e-3 would
destroy ImageNet features in a handful of batches.

---

## 15. Backpropagation, worked by hand

We need `∂L/∂w` for every weight. Computing each independently would need
446,076 separate passes. Backpropagation gets them all in **one** backward sweep.

It is the **chain rule**:

```
if  L depends on y, and y depends on w:

    ∂L/∂w = (∂L/∂y) · (∂y/∂w)
```

### A complete worked example

A two-layer network, one weight each, no biases, ReLU between.

```
x ──[w₁]──► a ──ReLU──► h ──[w₂]──► ŷ ──► L = ½(ŷ − y)²
```

**Values:**

```
x  = 2.0
w₁ = 0.5
w₂ = 1.5
y  = 3.0   (target)
```

**Forward pass:**

```
a = w₁ · x    = 0.5 × 2.0 = 1.0
h = ReLU(a)   = 1.0
ŷ = w₂ · h    = 1.5 × 1.0 = 1.5
L = ½(1.5 − 3.0)² = ½(−1.5)² = ½(2.25) = 1.125
```

**Backward pass** — start at the loss, work left:

```
∂L/∂ŷ = ŷ − y = 1.5 − 3.0 = −1.5

∂ŷ/∂w₂ = h = 1.0
∂L/∂w₂ = (∂L/∂ŷ)(∂ŷ/∂w₂) = (−1.5)(1.0) = −1.5     ← gradient for w₂

∂ŷ/∂h  = w₂ = 1.5
∂L/∂h  = (−1.5)(1.5) = −2.25

∂h/∂a  = 1  (because a > 0; it would be 0 if a < 0)
∂L/∂a  = (−2.25)(1) = −2.25

∂a/∂w₁ = x = 2.0
∂L/∂w₁ = (−2.25)(2.0) = −4.5                       ← gradient for w₁
```

**Update** with η = 0.1:

```
w₂ ← 1.5 − (0.1)(−1.5) = 1.65
w₁ ← 0.5 − (0.1)(−4.5) = 0.95
```

**Verify it improved:**

```
a = 0.95 × 2.0 = 1.9
h = 1.9
ŷ = 1.65 × 1.9 = 3.135
L = ½(3.135 − 3.0)² = ½(0.0182) = 0.0091
```

Loss fell from **1.125 to 0.0091**. One step.

### The three things to take from that

1. **Each gradient reuses the one to its right.** `∂L/∂a` was built from `∂L/∂h`.
   Nothing is recomputed. That reuse is what makes it O(one pass) instead of
   O(number of weights).
2. **ReLU's derivative gates the flow.** If `a` had been negative, `∂h/∂a = 0` and
   every gradient to its left becomes zero — that path learns nothing this step.
3. **Backprop is not a learning algorithm.** It is an efficient derivative
   calculator. Gradient descent (section 14) does the learning.

### What "frozen" means, mechanically

`base.trainable = False` does **not** stop data flowing forward, and does not stop
gradients flowing backward *through* those layers. It only excludes their weights
from the update step. Gradients still pass through to reach anything below.

---

## 16. Adam

Plain gradient descent uses one learning rate for every weight. Adam gives each
weight its own, adapted from its recent gradient history.

```
mₜ = β₁·mₜ₋₁ + (1−β₁)·g          running mean of the gradient      β₁ = 0.9
vₜ = β₂·vₜ₋₁ + (1−β₂)·g²         running mean of the SQUARED grad  β₂ = 0.999

m̂ₜ = mₜ / (1 − β₁ᵗ)              bias correction (m starts at 0)
v̂ₜ = vₜ / (1 − β₂ᵗ)

w ← w − η · m̂ₜ / ( √v̂ₜ + ε )
```

**The intuition is in that last line.** Dividing by `√v̂` — the typical *magnitude*
of this weight's gradients — means:

- A weight with **consistently large** gradients gets a **smaller** effective step
  (it is already moving fast).
- A weight with **small or noisy** gradients gets a **larger** one (it needs help).

`m` adds momentum: it averages over recent steps, so a single strange batch does
not throw the weight off course.

**Worked, first step:**

```
g = 0.1,  η = 0.001,  ε = 1e-8

m₁ = 0.9(0)   + 0.1(0.1)     = 0.01
v₁ = 0.999(0) + 0.001(0.01)  = 0.00001

m̂₁ = 0.01     / (1 − 0.9)     = 0.1
v̂₁ = 0.00001  / (1 − 0.999)   = 0.01

w ← w − 0.001 · 0.1 / (√0.01 + ε)
  = w − 0.001 · 0.1 / 0.1
  = w − 0.001
```

Note the update is ≈ η regardless of gradient size. **Adam normalises the step
size** — which is why it works well without careful per-layer tuning, and why it
is the default.

---

## 17. Transfer learning

Training from random weights means learning edges, colour and texture from
scratch. ImageNet has 1.4 million photos; you have ~12,775 unique originals.

MobileNetV2 already learned that visual vocabulary. Its early kernels detect edges
and colour gradients — universal facts about photographs, not facts about dogs.
Its late layers detect textures and materials, which is where *"fuzzy, desaturated
patch"* lives.

So: **keep the body, replace the head.**

| | From random | Transfer (yours) |
|---|---|---|
| Images needed | hundreds of thousands+ | 12,775 unique worked |
| Weights trained, stage 1 | 446,076 | **19,215** (4.3%) |
| Time | days | 47 + 25 min |

**Two stages, and the order is not optional:**

1. **Frozen.** New heads start random, so their gradients are large and
   meaningless. Applying those to pretrained convolutional filters would wreck
   them before they were ever used.
2. **Fine-tune** the top ~30 layers at 1e-4. Early layers stay frozen — an edge
   detector needs no adjustment for fruit.

---

## 18. Two heads, one loss

One body, two Dense layers:

```
                    ┌── Dense(2)  → softmax → condition
GAP(1280) → Dropout ┤
                    └── Dense(13) → softmax → food
```

Two losses, combined into one number to minimise:

```
L_total = 1.0 · L_condition  +  0.5 · L_food
```

**Why 0.5.** Condition is the product; food identity is a bonus, and the operator
can override it with `food_hint` anyway. Weighting condition 2× tells the shared
body whose features to prioritise when the two tasks compete for capacity.

**The gradient just splits.** For any shared weight `w` in the body:

```
∂L_total/∂w = 1.0 · ∂L_cond/∂w  +  0.5 · ∂L_food/∂w
```

Both tasks pull on the same features; the weights set who pulls harder.

**Multi-task learning often helps both.** Learning "this is a banana" forces the
body to encode shape and texture properly — which makes the rot question easier.
Your run bears this out: condition 94.5% and food 95.5% together.

---

## 19. Stopping it memorising

With 446,076 weights and 12,775 unique images, the model *can* simply memorise.
Four defences:

**Dropout (0.3).** During training, randomly zero 30% of values each step. The
model cannot rely on any single feature — it might vanish — so it spreads its
reasoning. At inference dropout is off and outputs are scaled to compensate.

**Augmentation.** Random flips, ±8° rotation, ±10% zoom, ±12% brightness and
contrast. It teaches invariance: *a banana is a banana at any angle*.

> **Ours is deliberately timid, and that is a real decision.** Heavy colour jitter
> would *manufacture* a brown patch on a fresh banana while still labelling it
> fresh — actively teaching the model that brown means fresh. When your labels
> depend on colour, you cannot augment colour aggressively.

**Early stopping (patience 5).** Watch validation loss. When it stops improving
for 5 epochs, halt and **restore the best epoch's weights**. Your run used it —
`Restoring model weights from the end of the best epoch: 7`.

**Weight decay / BN.** Batch norm has a mild regularising effect too, since batch
statistics inject a little noise.

**How to see overfitting:** training loss keeps falling while validation loss
bottoms out and turns up. The gap widening *is* the signature.

---

## 20. Splits, and the leak that would have fooled you

Three piles: **train** (70%) learns, **validation** (15%) decides when to stop,
**test** (15%) is opened once.

Why three? Because anything you make decisions from, you have partly tuned
towards. Validation picks the stopping epoch and the threshold, so it is no longer
neutral. Test influences nothing.

### The duplicate problem, in numbers

Your run:

```
97,022 photos  →  12,775 genuinely different originals
Average 7.59 copies of each original
Biggest cluster: 5,264 copies of ONE photo
```

If copies of one photo land in both train and test, the model memorises it and
aces the exam. Your log computed the damage:

```
>>> A random split would have leaked roughly 87% of test photos into training <<<
```

**87%.** You would have reported something near 99% that meant nothing.

### dHash — the maths

Byte comparison fails: a brightened copy shares no bytes. We compare *appearance*.

1. Resize to **9×8** greyscale — fine detail gone, structure kept
2. For each row, compare each pixel to its right neighbour: `1` if brighter
3. 8 rows × 8 comparisons = **64 bits**

**Why brightness cannot break it:**

```
original:   62  110  188  110
            62<110? yes → 1
            110<188? yes → 1
            188<110? no  → 0

+40 to all:  102  150  228  150
            102<150? yes → 1
            150<228? yes → 1
            228<150? no  → 0     identical bits
```

Adding a constant preserves every ordering, so the hash is unchanged.

**Hamming distance** — count differing bits:

```
a = 1011 0110
b = 1011 0010
        ^
XOR = 0000 0100  →  1 bit differs  →  distance 1
```

In code: `bin(a ^ b).count("1")`. Distance ≤ 4 out of 64 → same original.

**Blocking.** Comparing 97,022 photos pairwise is ~4.7 billion comparisons. So we
bucket by 16-bit chunks of the hash and only compare within buckets — near
identical hashes almost always share a chunk. That is why grouping took **24
seconds** instead of hours.

**Union-Find** then merges the pairs into groups. If A matches B and B matches C,
all three end up together even though A and C were never compared. Path
compression makes lookups effectively O(1).

Groups are assigned to splits **whole**. Never split a group.

---

## 21. Class imbalance

Your data after filtering:

```
apple    33,652        mango       434
banana   28,962        cucumber  1,575
orange   24,798        okra      1,939
```

A model that always answers "apple" scores well and learns nothing. Fix: make
mistakes on rare classes cost more.

```
              N_total
weight = ────────────────────
          n_cells × n_in_cell
```

**Worked:**

```
N_total = 66,466 (train),  n_cells = 16

apple/fresh (about 12,000):   66,466 / (16 × 12,000) = 0.35
mango/fresh (about 200):      66,466 / (16 ×    200) = 20.8
```

**Then we clip to [0.2, 5.0]** — and that is a trade-off worth being able to
defend. Uncapped, one mislabelled mango photo would shove the model as hard as
21 correct ones. The gradient becomes dominated by noise in the smallest cell and
training stops settling.

Clipping says: *correct most of the imbalance, but do not let 200 photos run the
model.* You accept residual bias toward common classes in exchange for stable
training.

*(Keras `class_weight` does not work for multi-output models — hence per-sample
weights.)*

---

## 22. Measuring it honestly

Accuracy alone is useless on imbalanced data — 95% "fresh" data gives 95%
accuracy for answering "fresh" every time.

### Your actual confusion matrix

Your run reported:

```
class            precision   recall      f1   photos
fresh                0.948    0.941   0.944    8,947
spoiled              0.942    0.950   0.946    9,076
OVERALL                              0.9451

spoiled, called fresh : 457 of 9,076 (5.04%)
fresh, called spoiled : 532 of 8,947
```

Which pins the matrix exactly:

```
                    predicted
                fresh    spoiled     total
  true fresh     8,415       532     8,947
  true spoiled     457     8,619     9,076
                                    ──────
                                    18,023
```

```
precision = TP / (TP + FP)     of what it CALLED spoiled, how much was?
recall    = TP / (TP + FN)     of what IS spoiled, how much did it catch?
F1        = 2·P·R / (P + R)    harmonic mean
```

**Spoiled — check every number against the table above:**

```
TP = 8,619    FP = 532    FN = 457

precision = 8619 / (8619 + 532) = 8619 / 9151 = 0.942  ✓
recall    = 8619 / (8619 + 457) = 8619 / 9076 = 0.950  ✓
F1        = 2(0.942)(0.950) / (0.942 + 0.950) = 0.946  ✓

overall   = (8415 + 8619) / 18,023 = 17,034 / 18,023 = 0.9451  ✓
```

Every figure reconciles. If yours ever do not, something upstream is wrong.

**Why the harmonic mean?** It punishes imbalance. Precision 1.0 with recall 0.1
gives an arithmetic mean of 0.55 (looks fine) but F1 of **0.18** (correctly awful).

### The asymmetry that matters

| | said fresh | said spoiled |
|---|---|---|
| **really fresh** | correct | wasteful — binned a good tomato |
| **really spoiled** | **DANGEROUS — rot passed as good** | correct |

For a food-inspection machine, **recall on spoiled is the headline number**. Yours
is **0.9566** on the shipped model.

### Threshold tuning

`argmax` treats both errors as equal. They are not. Lower the bar for calling
something spoiled and you catch more rot at the cost of more false alarms:

```
if P(spoiled) ≥ 0.35:  say spoiled
```

**Pick the threshold from validation, never test.** Choosing it from test turns
your final exam into a tuning set.

---

## 23. Quantisation

Weights are `float32` — 4 bytes, ~7 decimal digits. int8 is **1 byte**, 256
possible values.

### The mapping

```
real = scale × (quantised − zero_point)
```

Given a range of real values `[r_min, r_max]`:

```
scale      = (r_max − r_min) / 255
zero_point = round( −r_min / scale )
```

**Worked.** A weight tensor spanning −2.5 to +3.1:

```
scale      = (3.1 − (−2.5)) / 255 = 5.6 / 255 = 0.02196
zero_point = round(2.5 / 0.02196) = round(113.8) = 114

encode  0.5   →  round(0.5 / 0.02196) + 114 = round(22.8) + 114 = 137
decode  137   →  0.02196 × (137 − 114) = 0.505
```

Error: **0.005**. That is the price — about half a percent here.

### Why the representative dataset is not optional

Weights have known ranges — you can read them off the trained model. **Activations
do not.** Their range depends on what data flows through.

So the converter runs a few hundred real training images through and records the
actual min/max at every layer. Skip that and it guesses, the scales are wrong, and
you get a model that is fast, small and produces garbage.

### The payoff, measured on your run

```
size:  float32 → int8         4× smaller     (0.65 MB)
speed: integer ALUs are far faster on ARM than float
cost:  condition 0.9451 → 0.9377   (−0.7 points)
       spoiled recall 0.9496 → 0.9566   (+0.7, within noise)
```

ReLU6 (section 5) is why the damage is this small.

---

## 24. Rebuilding it from nothing

The full recipe, in order, with the reason for each step.

**1. Define the decision.** Not "classify food" — *"given photos from one camera
over one tray, is this item fresh or spoiled, and which is the costly error?"*
The costly error determines your metric before you write any code.

**2. Get labelled data.** Folders per class. Then **audit it before trusting it**:
count classes, look at samples with your own eyes, check for greyscale, check for
duplicates. Your dataset was 43% greyscale and 7.6× duplicated — neither was in
the description.

**3. Deduplicate perceptually.** dHash → Hamming distance → union-find → groups.
Without this your score is fiction.

**4. Split by group, stratified, with a fixed seed.** 70/15/15. Assert no group
spans two splits.

**5. Build the input pipeline.** decode → resize → **cache** → shuffle → augment →
normalise → batch → prefetch. Cache *before* augmentation or you freeze one set of
random transforms. Cache as uint8 not float32 (4× smaller).

**6. Pick an architecture that fits the target device.** Input size, α, and
quantisation are all deployment decisions made *before* training.

**7. Transfer learn.** Pretrained body, new head. Freeze, train the head, then
fine-tune the top at 10× lower LR.

**8. Handle imbalance** with sample weights, clipped. Be able to say what the
clipping cost you.

**9. Guard against overfitting.** Dropout, conservative augmentation, early
stopping on validation loss with best-weight restore.

**10. Evaluate once, honestly.** Confusion matrix. Per-class precision/recall/F1.
Name the costly error and report its rate. Never tune on test.

**11. Quantise, then re-measure.** Representative dataset from real training
images. Verify the shrunk model, do not assume.

**12. Write `labels.json`.** Class order, input size, preprocessing contract,
output-tensor mapping. The model is meaningless without it.

**13. Make training and inference preprocessing byte-identical.** This is the
single most common silent killer. Same colour order, same resize, same scaling.

**14. Deploy behind a quality gate.** Reject unusable frames rather than guessing.
A structured `{"status": "invalid_image", "reason": "blurred"}` is worth more than
a confident wrong answer.

**15. Test on real data from the real device.** Everything above measures the
model against its training distribution. Only the real rig tells you the truth.

---

## Where each idea lives in the code

| Concept | File |
|---|---|
| Normalisation contract (§1) | `ai/preprocessing.py::prepare_for_model` |
| Softmax, logit detection (§12) | `ai/multihead.py::_to_probs` |
| Architecture, α, two heads (§8-10, 18) | `training/train.py::build_model` |
| Augmentation (§19) | `training/dataset.py` |
| dHash, union-find, split (§20) | notebook §5-6 |
| Sample weights (§21) | notebook §8 |
| Confusion matrix (§22) | `training/evaluate.py`, notebook §12 |
| Quantisation (§23) | `training/export.py` |
| Quality gate (§24.14) | `ai/quality_check.py` |
| Multi-view aggregation | `ai/aggregation.py` |

---

## The honest footnote

Everything above is machinery for one thing: estimating how food **looks**. It
cannot see inside, cannot detect bacteria, and cannot tell you whether something
is safe to eat. `spoiled: 82` means the model is 82% confident the *appearance*
matches its spoiled class.

Knowing precisely what a system does **not** do is part of understanding it.
See [`LIMITATIONS.md`](LIMITATIONS.md).
