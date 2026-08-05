# Sprite LoRA stack — experiment log (2026-08-04)

> **Unmerged experiment branch (`test/sprite-stack`).** The code change on this
> branch is a **regression** against `main` and should not be shipped as-is.
> Kept for the findings and the evidence renders in
> [`sprite-stack-experiment/`](sprite-stack-experiment/), so none of this has
> to be rediscovered. `main` is unchanged.

## What prompted it

The sprite LoRA we use (`pkspbf_nb_v1`, the back&front variant of
[Pokemon Sprite XL PixelArt LoRA](https://civitai.com/models/378602)) is
published by its author alongside a **stack of three step-distillation LoRAs**,
with weights given in the sample prompt on its Civitai page:

```
<lora:pkspbf_nb_v1:1> <lora:Hyper-SDXL-8steps-lora:.4>
<lora:sdxl_lightning_2step_lora:.3> <lora:sd_xl_turbo_lora_v1:.3>
```

We fused `pkspbf` alone. Reference renders on the full stack (Civitai's hosted
generator) report **CFG 2.5 / 10 steps / Euler / CLIP skip 2** — so the
distillation is load-bearing, not a garnish, and the sampling settings are part
of the same package.

The trigger for investigating was a shipped sprite (`output/Thundro`) that
looked soft and mushy next to those reference renders.

## Config delta that was tried

| | reference | `main` | this branch |
|---|---|---|---|
| LoRAs | pkspbf 1.0 + hyper 0.4 + lightning 0.3 + turbo 0.3 | pkspbf 1.0 | full stack |
| base | NoobAI-XL **V-Pred-1.0** | NoobAI-XL 1.1 (**epsilon**) | epsilon (unchanged) |
| sampler | Euler | EulerAncestral | Euler |
| steps | 10 | 28 | 10 |
| CFG | 2.5 | 5.5 | 2.5 |
| CLIP skip | 2 (A1111) | 1 (diffusers default) | `clip_skip=1` (= A1111 2) |
| negative | empty | booru quality tags | `None` |
| canvas | 1216×832 (3:2) | 1536×768 (2:1) | 1536×768 (unchanged) |

Implementation: `_apply_lora` names the pkspbf adapter, loads the three
UNet-only accel LoRAs via `load_lora_weights(adapter_name=...)`, then
`set_adapters([1.0, 0.4, 0.3, 0.3])` and a single unscaled `fuse_lora()`.
Weights must be set *before* fusing — `fuse_lora(lora_scale=)` is a multiplier
over all adapters alike and cannot express a stack.

## Findings

### 1. `timestep_spacing="trailing"` is mandatory — and fails silently

**The most reusable finding here.** ByteDance's own diffusers instructions for
both the Lightning and Hyper-SD LoRAs specify:

```python
EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
```

NoobAI's `scheduler_config.json` ships `timestep_spacing: "leading"` with
`steps_offset: 1`. At 10 steps that is not a rounding difference:

```
leading    t = [901, 801, 701, 601, 501, 401, 301, 201, 101, 1]   first_sigma  8.39
trailing   t = [999, 899, 799, 699, 599, 499, 399, 299, 199, 99]  first_sigma 14.62
```

The distillation LoRAs are trained on a trajectory starting at t=999. Leading
never visits it, putting every step ~100 timesteps off — a full step of
misalignment at 10 steps. The model under-denoises: pale, ghostly, outline-only
renders with no fill. **Nothing errors.**

It is invisible at the pre-distillation 28 steps (leading starts at t=946 in
much finer increments), so it only bites in the few-step regime the accel LoRAs
exist to enable. The first sweep run on this branch was conducted *before* this
was found and its results are worthless — recorded here only so the mistake
isn't repeated.

### 2. Pair composition and tone trade off, with no overlap

All at seed 1234, 10 steps, trailing spacing, on the 1536×768 pair canvas:

| CFG | front/back pair? | tone | anatomy |
|---|---|---|---|
| 1.0 | **yes** (two creatures) | washed out, ghostly | quadruped |
| 1.5 | **yes** (front + rear view) | washed out, near-white | quadruped, crisper |
| 2.5 | no | pale | quadruped |
| 3.5 | no | **good** — gray body, saturated yellow, real outlines | **serpentine** |

The pair layout only survives at CFG ≤1.5; tone and prompt adherence only
arrive at 3.5. The trade is monotonic across the range — there is no setting
that gets both. CFG 3.5 is the only stacked render that produced the *serpent*
the prompt asked for rather than inventing legs, but it is one wide coil-knot
filling the canvas.

### 3. The baseline wins outright

`baseline_cfg5.5_s28_canvas.png` — `main`'s config, same seed 1234 — delivers a
clean front/back pair, true blacks, the correct storm-gray + yellow palette,
and a coiled serpent. Nothing in the stacked set matches it on any single axis,
let alone all three.

### 4. The premise was wrong

The investigation started from "our sprites are systematically mushy versus the
reference." They are not. The shipped `output/Thundro` sprite was an unlucky
seed; `main`'s config on seed 1234 stands up fine next to the reference
renders. A second seed would have cost six minutes and saved the entire detour.

Caveat in the other direction: "baseline is good" also rests on thin
evidence — one good seed here plus one mediocre shipped sprite.

## Still unresolved

**V-Pred-1.0 at CFG 1.0–1.5.** ~~This is the one untested lever~~ **Executed
2026-08-05 — see the addendum below. Verdict: closed for production.**

Lower-priority unknowns:

- **Alpha handling / fused-fp16 vs runtime patching.** We fuse four adapters
  into fp16 weights sequentially; ComfyUI (Civitai's backend) patches at
  runtime. Hyper and Lightning carry almost no metadata (`format: pt` only), so
  effective scale comes purely from in-file `.alpha` tensors — a real place for
  our 0.4/0.3/0.3 to differ from theirs.
- **Canvas aspect.** The accel LoRAs were distilled at 1024²; our 1536×768 is
  far off that. Note the reference renders were 3:2 and **never demonstrated a
  working pair either** — both are single creatures. There is no evidence the
  accel stack and pkspbf's pair layout coexist on *any* base.
- **Text-encoder LoRA application.** The UNet fused delta was verified
  (~12% of baseline weight magnitude, adapters registered correctly). The text
  encoders were **not** — and the loader prints a key-mismatch report with
  params "newly initialized". Unverified.

## Reproducing

The scripts used were throwaway. Each loads `load_txt2img_pipeline()`, renders
`build_prompt()` of Thundro's `sprite_prompt` from `output/Thundro/run.json` at
1536×768 with `_make_generator(1234)`, and saves the raw canvas — raw, because
the split heuristics otherwise sit between you and what the model did.

Two practical notes: renders take ~3 min each on the RTX 4000, so launch
detached (`Start-Process`) rather than under a tool timeout; and verify a LoRA
stack actually landed via `pipe.get_list_adapters()` plus a before/after weight
delta, rather than assuming.

## Addendum (2026-08-05): the V-Pred test — executed and closed

Ran the pre-registered test and two follow-up rounds on
`Laxhar/noobai-XL-Vpred-1.0` (fp16, `prediction_type="v_prediction"`,
`rescale_betas_zero_snr=True`, trailing spacing, full stack, 10 steps, Euler,
seed 1234, no negative, `clip_skip=1`). Renders: 12–13 s each on this GPU,
peak 6.5 GiB — ~14x faster than main's 28-step config. Evidence in
`sprite-stack-experiment/vpred_*.png`.

**Round 1 — the pre-registered ladder (CFG 1.0 / 1.5 / 2.0, plus 1.5 with
`guidance_rescale=0.7`).** V-Pred delivered exactly what motivated it: tone
holds at low CFG (storm-gray with real values, saturated blue wings and yellow
accents — nothing washed out, at every rung). But the **pair is gone at every
CFG, including 1.0** where epsilon held it: one wide centred creature filling
the 2:1 canvas each time. Same prompt (preserved in
`resources/Screenshot 2026-08-04 *.png` — Thundro's `run.json` was lost in an
archive move; the tag string in those screenshots is the same prompt).

**Round 2 — summoning the pair.** Two levers: the author's 3:2 aspect
(1216×832), and *asking* for the layout — `multiple views, front and back,`
inserted after `gen3,` (danbooru vocabulary the base knows; on epsilon the
pair came from the LoRA prior alone, the prompt never asked). The aspect alone
does nothing. **The tag works**: clean two-view canvases on both aspects at
CFG 1.5, well-separated with a splittable background gap.

**Round 3 — does the tagged pair survive real guidance?** CFG 2.0 / 2.5 / 3.0
with the tag on 2:1. The layout now survives to ~2.5 (tone improving with
CFG), and the halves begin merging across the midline at 3.0. So the epsilon
trade ("pair XOR tone") genuinely resolves on V-Pred + tag at CFG ~2.5.

**Why it is still closed:** across every variant, all seeds-1234 renders
produce **mirrored twins, never a genuine back view** — the right-hand
creature shows its face. The `multiple views` tag summons the *layout* but
not the trained front/back semantics, which apparently do not survive on this
base at any tested setting. Design richness also stays visibly below the
epsilon baseline (`baseline_cfg5.5_s28_canvas.png` — compare its true
back-view right half and detailed mane). A back sprite that is a mirrored
front is worse than no back sprite: it lies about the creature.

**What survives the closure:**

- 13-second renders make this stack the right harness for *future rendering
  experiments* (prompt A/Bs, seed sweeps) even though production stays on
  main — 14x cheaper iteration.
- The `multiple views, front and back` tag finding is prompt-side and
  base-independent in principle; if the pair layout ever weakens on main's
  epsilon config, it is the first thing to try there.
- One seed only, as before. The mirrored-twin failure was consistent across
  all 7 pair renders, though, which is more evidence than the single-seed
  caveats elsewhere in this log.
