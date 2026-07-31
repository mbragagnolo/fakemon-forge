"""Tests for fakemon_forge.cries — procedural GBA-style cry.wav synthesis.

Pure stdlib module (no torch/diffusers), so these are regular tests and run
everywhere, including the keep sandbox container. Do NOT mark them `ml`.
"""

import statistics
import wave

from fakemon_forge.cries import generate_cry

_SR = 10512


def _read_wav(path):
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    return params, frames


def _samples(frames):
    # 8-bit unsigned PCM: one byte per frame.
    return list(frames)


def _zero_crossing_rate(samples):
    """ZCR over the sustained middle portion, relative to the 128 midpoint."""
    lo = len(samples) // 4
    hi = len(samples) * 3 // 4
    seg = [s - 128 for s in samples[lo:hi]]
    if len(seg) < 2:
        return 0.0
    crossings = 0
    for a, b in zip(seg, seg[1:]):
        if (a >= 0) != (b >= 0):
            crossings += 1
    return crossings / len(seg)


def test_format(tmp_path):
    out = tmp_path / "cry.wav"
    generate_cry("Florabud", 1, ["Grass"], str(out))
    params, frames = _read_wav(out)
    assert params.nchannels == 1
    assert params.sampwidth == 1
    assert params.framerate == _SR
    duration = params.nframes / _SR
    assert 0.35 <= duration <= 1.55, duration
    samples = _samples(frames)
    assert all(0 <= s <= 255 for s in samples)
    peak = max(abs(s - 128) for s in samples)
    assert abs(peak - 120) <= 3, peak


def test_determinism(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    generate_cry("Zapfin", 2, ["Electric", "Flying"], str(a))
    generate_cry("Zapfin", 2, ["Electric", "Flying"], str(b))
    assert a.read_bytes() == b.read_bytes()


def test_stage_growth_durations_strictly_increasing(tmp_path):
    durations = []
    for stage in (1, 2, 3):
        out = tmp_path / f"s{stage}.wav"
        generate_cry("Emberling", stage, ["Fire"], str(out))
        params, _ = _read_wav(out)
        durations.append(params.nframes)
    assert durations[0] < durations[1] < durations[2], durations


def test_type_register_zcr(tmp_path):
    fire = tmp_path / "fire.wav"
    fairy = tmp_path / "fairy.wav"
    generate_cry("Cindermon", 1, ["Fire"], str(fire))
    generate_cry("Glimmermon", 1, ["Fairy"], str(fairy))
    _, ff = _read_wav(fire)
    _, yf = _read_wav(fairy)
    fire_zcr = _zero_crossing_rate(_samples(ff))
    fairy_zcr = _zero_crossing_rate(_samples(yf))
    assert fairy_zcr > fire_zcr, (fire_zcr, fairy_zcr)


def test_stage_pitch_drops(tmp_path):
    def zcr_median(stage):
        out = tmp_path / f"pitch{stage}.wav"
        generate_cry("Dracling", stage, ["Dragon"], str(out))
        _, frames = _read_wav(out)
        samples = _samples(frames)
        # windowed ZCR medians over the sustained region
        seg = samples[len(samples) // 4: len(samples) * 3 // 4]
        win = _SR // 20
        rates = []
        for i in range(0, len(seg) - win, win):
            rates.append(_zero_crossing_rate(seg[i:i + win]))
        return statistics.median(rates) if rates else 0.0

    assert zcr_median(3) < zcr_median(1)


def test_empty_types(tmp_path):
    out = tmp_path / "empty.wav"
    generate_cry("Nulltype", 1, [], str(out))
    params, frames = _read_wav(out)
    assert params.nchannels == 1
    assert params.framerate == _SR
    assert 0.35 <= params.nframes / _SR <= 1.55


def test_unknown_type(tmp_path):
    out = tmp_path / "unknown.wav"
    generate_cry("Mysterion", 1, ["Cosmic"], str(out))
    params, _ = _read_wav(out)
    assert params.nchannels == 1
    assert 0.35 <= params.nframes / _SR <= 1.55


def test_empty_line_name(tmp_path):
    out = tmp_path / "noname.wav"
    generate_cry("", 1, ["Water"], str(out))
    params, _ = _read_wav(out)
    assert params.sampwidth == 1
    assert params.framerate == _SR
