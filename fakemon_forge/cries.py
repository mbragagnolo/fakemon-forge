"""Procedural GBA-style creature-cry synthesis.

Synthesizes a Gen 3-style Fakemon "cry" as a mono / 8-bit unsigned / 10512 Hz
``cry.wav`` — procedurally, using only the standard library (no ML, no GPU, no
network, no time/OS entropy).

The single public entry point is :func:`generate_cry`. The cry is a deterministic
function of ``(line_name, stage, types)``: identical arguments produce a
byte-identical WAV, and a whole evolution line (sharing ``line_name`` — stage 1's
name) shares one seeded "voice" and motif, with per-stage transforms making later
stages longer and lower-pitched.

The synthesis model (voice source, motif, per-type register bands, stage
transform) is audition-validated against a corpus of official Gen 3 cries. The
numeric tables below are tunable starting points, not contracts.
"""

import hashlib
import math
import random
import wave

_SR = 10512  # sample rate (Hz) — matches Gen 3 cry playback rate
_PEAK = 120  # target peak absolute deviation from 128 (reference cries peak ~120)

# Type profile rows: register band (Hz), syllable-count weights (counts 1-4),
# noise base, AM base, duration multiplier. Audition-validated defaults.
_PROFILES = {
    "Grass": ((130, 420), (3, 3, 1, 0), 0.10, 0.45, 1.0),
    "Dragon": ((100, 300), (4, 2, 0, 0), 0.16, 0.50, 1.0),
    "Fighting": ((140, 380), (2, 3, 2, 0), 0.12, 0.45, 1.0),
    "Rock": ((110, 320), (3, 2, 1, 0), 0.18, 0.50, 1.0),
    "Ground": ((110, 340), (3, 2, 1, 0), 0.14, 0.45, 1.0),
    "Normal": ((450, 1800), (1, 3, 3, 1), 0.03, 0.30, 0.85),
    "Fairy": ((900, 2400), (1, 2, 3, 2), 0.02, 0.30, 0.75),
    "Flying": ((700, 2200), (1, 3, 2, 1), 0.04, 0.35, 0.80),
    "Electric": ((220, 700), (2, 3, 1, 0), 0.10, 0.60, 0.90),
    "Bug": ((260, 800), (2, 2, 2, 1), 0.08, 0.65, 0.90),
    "Psychic": ((350, 1100), (3, 2, 1, 0), 0.03, 0.50, 1.25),
    "Ghost": ((300, 900), (3, 2, 0, 0), 0.05, 0.55, 1.10),
    "Poison": ((500, 1500), (3, 2, 1, 0), 0.15, 0.50, 1.05),
    "Water": ((250, 850), (2, 3, 1, 0), 0.03, 0.35, 1.35),
    "Ice": ((400, 1300), (2, 3, 1, 0), 0.04, 0.35, 1.20),
    "Fire": ((100, 320), (3, 2, 1, 0), 0.25, 0.40, 1.0),
    "Steel": ((180, 550), (2, 2, 2, 0), 0.14, 0.55, 1.0),
    "Dark": ((120, 360), (3, 2, 1, 0), 0.12, 0.50, 1.0),
}
# Used when types is empty or the primary type is unrecognized.
_DEFAULT_PROFILE = ((300, 900), (2, 3, 1, 0), 0.08, 0.40, 1.0)

# Melodic interval set the motif draws from (ratios above the register root).
_SCALE_SET = (1.0, 1.19, 1.34, 1.5, 1.78, 2.0, 0.84, 0.67)
_FM_RATIOS = (1.5, 2.0, 2.77, 3.51)
_CONTOURS = ("fall", "rise", "bend", "flat", "trill")


def _seeded_rng(line_name: str) -> random.Random:
    """A local RNG seeded solely from ``line_name`` (never the global module)."""
    digest = hashlib.sha256(line_name.encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)


def _build_voice(rng: random.Random, band, noise_base, am_base) -> dict:
    """Draw the per-line timbre from ``rng`` within the type's profile."""
    lo, hi = band
    register = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    source = rng.choice(("pwm", "saw", "triangle", "fm", "ring"))
    return {
        "register": register,
        "source": source,
        "duty": rng.uniform(0.12, 0.5),
        "fm_ratio": rng.choice(_FM_RATIOS),
        "fm_index": rng.uniform(0.8, 3.0),
        "detune": rng.uniform(1.002, 1.03),
        # Noise is kept subordinate and low-passed so it adds grit without
        # swamping the tonal core's zero-crossing rate.
        "noise_amt": min(noise_base * rng.uniform(0.5, 1.6), 0.30),
        "am_rate": rng.uniform(18.0, 90.0),
        "am_depth": am_base * rng.uniform(0.6, 1.0),
        "vib_rate": rng.uniform(4.5, 11.0),
        "vib_depth": rng.uniform(0.008, 0.03),
        "trill_rate": rng.uniform(11.0, 22.0),
        # A per-line noise seed so the grit bed varies by voice yet stays
        # identical across a line's stages (same line_name -> same seed).
        "noise_seed": rng.getrandbits(32),
    }


def _build_motif(rng: random.Random, weights) -> list:
    """Draw the per-line melodic motif (1-4 syllables) from ``rng``."""
    count = rng.choices((1, 2, 3, 4), weights=weights)[0]
    syllables = []
    for i in range(count):
        length = rng.uniform(0.8, 1.2)
        if i == count - 1:
            length *= 1.6  # the final syllable lands longer
        syllables.append(
            {
                "interval": rng.choice(_SCALE_SET),
                "contour": rng.choice(_CONTOURS),
                "length": length,
                "depth": rng.uniform(0.25, 0.6),
                "gap": rng.uniform(0.015, 0.070),
            }
        )
    return syllables


def _source_sample(phase: float, voice: dict, harden: float) -> float:
    """Evaluate the voice's oscillator at a given phase (in cycles)."""
    source = voice["source"]
    frac = phase - math.floor(phase)
    if source == "pwm":
        base = 1.0 if frac < voice["duty"] else -1.0
    elif source == "saw":
        base = 2.0 * frac - 1.0
    elif source == "triangle":
        base = 4.0 * abs(frac - 0.5) - 1.0
    elif source == "fm":
        index = voice["fm_index"] + harden  # hardening brightens the attack
        base = math.sin(
            2.0 * math.pi * phase + index * math.sin(2.0 * math.pi * voice["fm_ratio"] * phase)
        )
    else:  # ring
        base = math.sin(2.0 * math.pi * phase) * math.sin(2.0 * math.pi * phase * voice["detune"])
    if harden > 0.0:
        # Mix in an octave-up rough partial so later stages sound harsher.
        octave = 2.0 * (phase - math.floor(phase)) - 1.0
        base = base + 0.25 * harden * octave
    return base


def _render(voice: dict, motif: list, sound_time: float, stage_pitch: float, harden: float) -> list:
    """Render the motif to a float buffer (roughly in [-1, 1] before normalizing)."""
    total_weight = sum(s["length"] for s in motif)
    buf: list[float] = []
    phase = 0.0
    lp = 0.0  # one-pole low-pass state for the noise bed
    nrng = random.Random(voice["noise_seed"])  # per-line noise, identical across stages
    register = voice["register"]

    for syl in motif:
        dur = sound_time * (syl["length"] / total_weight)
        n = max(1, int(dur * _SR))
        freq = register * syl["interval"] * stage_pitch
        contour, depth = syl["contour"], syl["depth"]
        for i in range(n):
            t = i / n
            # Melodic contour (glides the fundamental within the syllable).
            if contour == "fall":
                cm = 1.0 - 0.4 * depth * t
            elif contour == "rise":
                cm = 1.0 + 0.4 * depth * t
            elif contour == "bend":
                cm = 1.0 + 0.4 * depth * math.sin(math.pi * t)
            elif contour == "trill":
                cm = 1.0 + 0.12 * depth * math.sin(2.0 * math.pi * voice["trill_rate"] * (i / _SR))
            else:  # flat + vibrato
                cm = 1.0
            vib = 1.0 + voice["vib_depth"] * math.sin(2.0 * math.pi * voice["vib_rate"] * (i / _SR))
            phase += freq * cm * vib / _SR

            src = _source_sample(phase, voice, harden)

            # Low-passed, subordinate noise: grit without inflating the ZCR.
            white = nrng.uniform(-1.0, 1.0)
            lp += 0.06 * (white - lp)
            src += voice["noise_amt"] * lp * 2.0

            # Tremolo (stays positive, so it adds no zero crossings).
            am = 1.0 - voice["am_depth"] * (0.5 + 0.5 * math.sin(2.0 * math.pi * voice["am_rate"] * (i / _SR)))

            # Per-syllable envelope: ~6% attack, gentle decay.
            if t < 0.06:
                env = t / 0.06
            else:
                env = 1.0 - 0.3 * ((t - 0.06) / 0.94)

            buf.append(src * am * env)

        # Inter-syllable gap (silence).
        buf.extend([0.0] * int(syl["gap"] * _SR))

    return buf


def _finalize(buf: list) -> bytes:
    """Fade, peak-normalize to ~120, and quantize to unsigned 8-bit bytes."""
    n = len(buf)
    # Linear fade over the final 10% of samples.
    fade = max(1, n // 10)
    for i in range(n - fade, n):
        buf[i] *= (n - i) / fade

    peak = max((abs(v) for v in buf), default=0.0)
    scale = (_PEAK / peak) if peak > 0.0 else 0.0

    out = bytearray(n)
    for i, v in enumerate(buf):
        sample = int(round(128 + v * scale))
        out[i] = 0 if sample < 0 else 255 if sample > 255 else sample
    return bytes(out)


def generate_cry(line_name: str, stage: int, types: list, output_path: str) -> None:
    """Synthesize a Gen 3-style cry and write it as a WAV to ``output_path``.

    Args:
        line_name: Stage 1's name, shared across the evolution line. Sole seed
            source — the whole line shares one voice + motif.
        stage: Evolution stage (``>= 1``). Stages ``> 1`` are longer, lower, and
            harsher; duration is always capped at 1.5 s.
        types: The stage's types. ``types[0]`` selects the profile; an empty list
            or an unknown primary type falls back to the default profile.
        output_path: Full destination path for the ``cry.wav`` (written verbatim).
    """
    rng = _seeded_rng(line_name)

    primary = types[0] if types else None
    band, weights, noise_base, am_base, dur_mult = _PROFILES.get(primary, _DEFAULT_PROFILE)

    voice = _build_voice(rng, band, noise_base, am_base)
    motif = _build_motif(rng, weights)

    base_dur = rng.uniform(0.45, 1.0)
    stage_stretch = 1.0 + 0.20 * (stage - 1)
    total = min(base_dur * dur_mult * stage_stretch, 1.5)

    gaps = sum(s["gap"] for s in motif)
    sound_time = max(0.2, total - gaps)

    stage_pitch = 0.90 ** (stage - 1)
    harden = 0.25 * (stage - 1)

    buf = _render(voice, motif, sound_time, stage_pitch, harden)
    data = _finalize(buf)

    with wave.open(output_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(_SR)
        w.writeframes(data)
