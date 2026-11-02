"""Digital Communications Lab, Experiment 3: analog modulation (PM and AM).

Reconstruction in NumPy of the three GNU Radio Companion flowgraphs used in
the lab (screenshots in flowgraphs/). Each GRC block is mirrored by a short
function, so the scripts run without GNU Radio:

  fg1  phase modulation of a 4 kHz tone (beta = 2 rad) onto a 100 kHz carrier
  fg2  AM (mu = 0.5) + homodyne receiver with a DC blocker
  fg3  audio over AM through a noisy channel, received back as a WAV file

    python analog_modulation.py          -> results/*.png, results/*.wav
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from scipy.signal import firwin, lfilter, resample_poly  # noqa: E402
from scipy.special import jv  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rng = np.random.default_rng(1405)


# ------------------------------------------------------------------ GRC blocks
def signal_source(fs, f, n, amp=1.0, complex_out=False):
    t = np.arange(n) / fs
    return amp * (np.exp(2j * np.pi * f * t) if complex_out else np.cos(2 * np.pi * f * t))


def phase_mod(x, sensitivity):
    return np.exp(1j * sensitivity * x)


def channel_model(x, noise_voltage):
    """AWGN with total complex noise power noise_voltage**2."""
    n = noise_voltage / np.sqrt(2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + n


def dc_blocker(x, length=32):
    """GNU Radio 'long form' DC blocker: subtract a two-stage moving average."""
    ma = lambda v: lfilter(np.ones(length) / length, 1, v)  # noqa: E731
    delay = length - 1
    avg = ma(ma(x))
    xd = np.concatenate([np.zeros(delay), x[:-delay]])
    return xd - avg


def spectrum(x, fs, nfft=4096):
    win = np.hanning(nfft)
    segs = [x[i:i + nfft] * win for i in range(0, len(x) - nfft, nfft // 2)]
    p = np.mean([np.abs(np.fft.fftshift(np.fft.fft(s))) ** 2 for s in segs], axis=0)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    return f, 10 * np.log10(p / p.max())


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------- flowgraph 1: PM
def fg1_phase_modulation():
    fs, fm, fc, beta = 2.048e6, 4e3, 100e3, 2.0          # tone x Constant(2) -> Phase Mod(1)
    n = 2 ** 18
    msg = signal_source(fs, fm, n) * 2.0
    s = phase_mod(msg, 1.0) * signal_source(fs, fc, n, complex_out=True)

    f, p = spectrum(s, fs, nfft=2 ** 16)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    t = np.arange(n) / fs
    axs[0].plot(t[:1024] * 1e3, s.real[:1024], lw=0.8, label="Re{s(t)}")
    axs[0].plot(t[:1024] * 1e3, np.cos(2 * np.pi * fm * t[:1024]), "k--", lw=0.8, label="message")
    axs[0].set_xlabel("time (ms)")
    axs[0].set_title("PM signal: constant envelope, phase follows the message")
    axs[0].legend(fontsize=8)
    axs[1].plot((f - fc) / 1e3, p, lw=0.8)
    k = np.arange(-6, 7)
    lines = 20 * np.log10(np.abs(jv(k, beta)) / np.abs(jv(k, beta)).max())
    axs[1].plot(k * fm / 1e3, lines, "rv", label="|J_n(β)|, β = 2 (theory)")
    carson = 2 * (beta + 1) * fm
    axs[1].axvspan(-carson / 2e3, carson / 2e3, color="g", alpha=0.08, label=f"Carson BW = {carson/1e3:.0f} kHz")
    axs[1].set_xlim(-40, 40)
    axs[1].set_ylim(-60, 3)
    axs[1].set_xlabel("frequency offset from the 100 kHz carrier (kHz)")
    axs[1].set_ylabel("dB")
    axs[1].set_title("Spectrum: Bessel sidebands every 4 kHz")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
    save(fig, "fg1_phase_modulation.png")
    pwr = np.sum(jv(np.arange(-3, 4), beta) ** 2)
    print(f"FG1  PM beta=2: {100*pwr:.1f} % of the power within +-3 sidebands (Carson {carson/1e3:.0f} kHz)")


# ------------------------------------------------------------- flowgraph 2: AM
def fg2_am_homodyne(fc_tx=10e3, fc_lo=10e3):
    fs, fm, mu = 32e3, 1e3, 0.5
    n = 2 ** 15
    msg = signal_source(fs, fm, n)
    am = (1 + mu * msg) * signal_source(fs, fc_tx, n)        # Constant(1) + 0.5*msg, x carrier
    tx = am + 1j * am                                         # Float To Complex (re = im = AM)
    rx = channel_model(tx, 0.0)
    base = rx * signal_source(fs, -fc_lo, n, complex_out=True)  # LO at -10 kHz (complex)
    out_lab = dc_blocker(base)                                # as in the lab: no low-pass filter
    lpf = firwin(129, 3e3, fs=fs)
    out_fixed = dc_blocker(lfilter(lpf, 1, base))             # added: LPF before the DC blocker
    return fs, msg, am, out_lab, out_fixed


def fg2_plots():
    fs, msg, am, out_lab, out_fixed = fg2_am_homodyne()
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    t = np.arange(len(msg)) / fs * 1e3
    sl = slice(2000, 2000 + 128)
    axs[0, 0].plot(t[sl], am[sl], lw=0.9, label="AM, μ = 0.5")
    axs[0, 0].plot(t[sl], 1 + 0.5 * msg[sl], "k--", lw=0.9, label="envelope 1 + 0.5 m(t)")
    axs[0, 0].set_title("Transmitted AM (10 kHz carrier, 1 kHz tone)")
    axs[0, 0].set_xlabel("time (ms)")
    axs[0, 0].legend(fontsize=8)
    f, p = spectrum(am, fs)
    axs[0, 1].plot(f / 1e3, p)
    axs[0, 1].set_title("AM spectrum: carrier + sidebands at ±1 kHz")
    axs[0, 1].set_xlabel("frequency (kHz)")
    for out, lab in ((out_lab, "lab receiver (mixer + DC blocker)"), (out_fixed, "with a 3 kHz low-pass filter")):
        f, p = spectrum(out, fs)
        axs[1, 1].plot(f / 1e3, p, lw=0.9, label=lab)
        axs[1, 0].plot(t[sl], out.real[sl], lw=0.9, label=lab)
    axs[1, 0].set_title("Demodulated output (real part)")
    axs[1, 0].set_xlabel("time (ms)")
    axs[1, 0].legend(fontsize=8)
    axs[1, 1].set_title("Output spectrum: the 2f_c term aliases to ±12 kHz")
    axs[1, 1].set_xlabel("frequency (kHz)")
    axs[1, 1].legend(fontsize=8)
    for ax in axs[:, 1]:
        ax.set_ylim(-80, 3)
        ax.set_ylabel("dB")
    for ax in axs.ravel():
        ax.grid(alpha=0.3)
    save(fig, "fg2_am_homodyne.png")

    # LO mismatch: the GUI slider moves the TX carrier but the LO stays at 10 kHz
    fs, msg, _, _, out = fg2_am_homodyne(fc_tx=10.2e3, fc_lo=10e3)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    t = np.arange(len(msg)) / fs * 1e3
    ax.plot(t[:3200], out.real[:3200], lw=0.8, label="Re{output}")
    ax.plot(t[:3200], np.abs(out[:3200]), "k", lw=1, label="|output|")
    ax.set_xlabel("time (ms)")
    ax.set_title("Carrier at 10.2 kHz, LO at 10 kHz: the message beats at the 200 Hz offset")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "fg2_lo_mismatch.png")


# ------------------------------------------------- flowgraph 3: audio over AM
def test_audio(fs=48_000, seconds=3.0):
    """Synthetic test signal (no third-party audio): a short melody plus a chirp."""
    t = np.arange(int(fs * seconds)) / fs
    notes = [440, 554, 659, 880, 659, 554]
    melody = np.concatenate([0.5 * np.sin(2 * np.pi * f * t[: len(t) // 12]) for f in notes])
    chirp = 0.3 * np.sin(2 * np.pi * (200 * t[: len(t) // 2] + 3000 * t[: len(t) // 2] ** 2))
    x = np.concatenate([melody, chirp])[: len(t)]
    return x / np.max(np.abs(x))


def fg3_audio_over_am(noise_voltage=0.05, write=True):
    fa, fs, fc, mu = 48_000, 1_000_000, 100e3, 0.5
    audio = test_audio(fa)
    x = resample_poly(audio, 125, 6)                         # 48 kS/s -> 1 MS/s
    x = lfilter(firwin(255, 20e3, fs=fs), 1, np.concatenate([x, np.zeros(127)]))[127:]  # LPF 20 kHz, delay removed
    n = len(x)
    am = (1 + mu * x) * signal_source(fs, fc, n)
    rx = channel_model(am.astype(complex), noise_voltage)
    base = 2 * rx * signal_source(fs, fc, n)                  # coherent product detector
    # Decimate first, then remove DC. A 32-tap DC blocker at 1 MS/s would act as a
    # high-pass filter with its corner near 15 kHz and remove most of the audio.
    y = dc_blocker(resample_poly(base.real, 6, 125), 2048) / mu   # back to 48 kS/s
    # align and measure the output SNR against the input
    lag = np.argmax(np.correlate(y[:20000], audio[:20000], "full")) - 19999
    y_al = y[lag:lag + len(audio)] if lag >= 0 else np.concatenate([np.zeros(-lag), y])[:len(audio)]
    m = min(len(y_al), len(audio))
    s = slice(int(0.2 * fa), m)                                # skip the DC blocker start-up transient
    g = np.dot(y_al[s], audio[s]) / np.dot(audio[s], audio[s])
    snr = 10 * np.log10(np.sum(audio[s] ** 2) / np.sum((y_al[s] / g - audio[s]) ** 2))
    if write:
        os.makedirs(OUT, exist_ok=True)
        wavfile.write(os.path.join(OUT, "fg3_tx_audio.wav"), fa, (0.8 * audio * 32767).astype(np.int16))
        yy = y_al / np.max(np.abs(y_al))
        wavfile.write(os.path.join(OUT, "fg3_rx_audio.wav"), fa, (0.8 * yy * 32767).astype(np.int16))
    return audio, y_al, am, fs, snr


def fg3_plots():
    audio, y, am, fs, snr = fg3_audio_over_am()
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    f, p = spectrum(am, fs, nfft=8192)
    axs[0].plot(f / 1e3, p, lw=0.8)
    axs[0].set_xlim(0, 200)
    axs[0].set_ylim(-80, 3)
    axs[0].set_title("AM spectrum around the 100 kHz carrier")
    axs[0].set_xlabel("frequency (kHz)")
    t = np.arange(len(audio)) / 48e3
    sl = slice(int(0.1 * 48e3), int(0.1 * 48e3) + 480)
    axs[1].plot(t[sl] * 1e3, audio[sl], "k", lw=1, label="sent")
    gain = np.dot(y[sl], audio[sl]) / np.dot(y[sl], y[sl])     # least-squares scale for display
    axs[1].plot(t[sl] * 1e3, gain * y[sl], "--", lw=1, label="received")
    axs[1].set_title(f"Recovered audio (output SNR {snr:.1f} dB)")
    axs[1].set_xlabel("time (ms)")
    axs[1].legend(fontsize=8)
    nv = np.logspace(-3, 0.5, 12)
    snrs = [fg3_audio_over_am(v, write=False)[4] for v in nv]
    cnr = 10 * np.log10(0.5 / nv ** 2)                        # carrier power 1/2 vs. noise nv^2
    axs[2].plot(cnr, snrs, "o-")
    axs[2].set_xlabel("carrier-to-noise ratio in the 1 MHz band (dB)")
    axs[2].set_ylabel("output audio SNR (dB)")
    axs[2].set_title("Coherent AM: output SNR follows the channel CNR")
    for ax in axs:
        ax.grid(alpha=0.3)
    save(fig, "fg3_audio_over_am.png")
    print(f"FG3  audio over AM: output SNR {snr:.1f} dB at noise voltage 0.05")


if __name__ == "__main__":
    fg1_phase_modulation()
    fg2_plots()
    fg3_plots()
