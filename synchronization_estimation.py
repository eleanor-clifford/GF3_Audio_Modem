#  vim: set ts=4 sw=4 tw=0 et :
from config import (
    PLOTTING_ENABLED,
    CHIRP_START,
    CHIRP_END,
    N_KNOWN_OFDM_FRAME_START,
    N_KNOWN_OFDM_FRAME_END,
    MAX_NUMBER_OF_SYMBOLS_IN_FRAME,
    OFDM_BODY_LENGTH,
    OFDM_CYCLIC_PREFIX_LENGTH,
    OFDM_DATA_INDEX_RANGE,
)
import numpy as np
import scipy.signal as signal

if PLOTTING_ENABLED:
    import matplotlib.pyplot as plt


def synchronise_crop(transmitted_signal: np.ndarray):
    convolve_start = signal.convolve(transmitted_signal, CHIRP_START[::-1])
    convolve_end = signal.convolve(transmitted_signal, CHIRP_END[::-1])

    start_idx = np.argmax(np.abs(convolve_start))
    end_idx = np.argmax(np.abs(convolve_end))

    cropped_signal = transmitted_signal[start_idx + len(CHIRP_START): end_idx]
    frame_length = (
        (OFDM_BODY_LENGTH + OFDM_CYCLIC_PREFIX_LENGTH) *
        (MAX_NUMBER_OF_SYMBOLS_IN_FRAME + N_KNOWN_OFDM_FRAME_START + N_KNOWN_OFDM_FRAME_END)
    )
    N_FRAMES = round(len(cropped_signal) / frame_length)
    difference = len(cropped_signal) - N_FRAMES * frame_length
    print(f"initial sync correction: {difference} samples")

    return np.reshape(cropped_signal[:N_FRAMES * frame_length], (-1, OFDM_BODY_LENGTH + OFDM_CYCLIC_PREFIX_LENGTH))


def estimate_synchronization_drift(first_signal: np.ndarray, second_signal: np.ndarray, plot=False):
    assert first_signal.size == second_signal.size
    signal_cross_correlation = signal.correlate(first_signal, second_signal, mode='same')
    maximum_cross_correlation_index = np.argmax(signal_cross_correlation)

    # Take 5 point around the correlation peak
    t = np.linspace(maximum_cross_correlation_index - 2, maximum_cross_correlation_index + 2, 5)

    quadratic_fit_of_peak = np.polyfit(t, signal_cross_correlation[
        maximum_cross_correlation_index - 2:
        maximum_cross_correlation_index + 3
    ], 2)

    peak = -quadratic_fit_of_peak[1] / (2 * quadratic_fit_of_peak[0])
    sampling_drift = peak - first_signal.size / 2

    if PLOTTING_ENABLED and plot:
        t2 = np.linspace(peak - 2, peak + 2, 100)
        plt.figure()
        plt.plot(np.arange(len(signal_cross_correlation)) - first_signal.size / 2, signal_cross_correlation)
        plt.plot(t2 - first_signal.size / 2, np.poly1d(quadratic_fit_of_peak)(t2))
        plt.xlabel("Drift")
        plt.ylabel("Cross-correlation magnitude")
        _, _, ymin, ymax = plt.axis()
        plt.axis([-25, 25, ymin, ymax])
        plt.vlines([sampling_drift], ymin, np.poly1d(quadratic_fit_of_peak)(peak), color='C2')
        plt.savefig("plots/cross_correlation.pgf")
        plt.savefig("plots/cross_correlation.pdf")

    print('total drift: ', -sampling_drift)
    return -sampling_drift  # by convention


def estimate_channel_coefficients_and_variance(
    recorded_known_ofdm_blocks: np.ndarray,
    known_ofdm_blocks: np.ndarray,
    drift: float,
    drift_per_sample: float,
    plot = False
):
    sum_of_gains = np.zeros(OFDM_BODY_LENGTH, dtype=np.complex128)

    assert len(recorded_known_ofdm_blocks[0]) == OFDM_BODY_LENGTH
    assert len(known_ofdm_blocks[0]) == OFDM_BODY_LENGTH

    for block_idx, (recorded_block, known_block) in enumerate(zip(recorded_known_ofdm_blocks, known_ofdm_blocks)):
        block_drift = drift + drift_per_sample * OFDM_BODY_LENGTH * block_idx
        sum_of_gains += estimate_frequency_gains_from_block(
            recorded_block, known_block, block_drift, plot
        )

    channel_fft = sum_of_gains / len(recorded_known_ofdm_blocks)

    channel_fft[:OFDM_DATA_INDEX_RANGE["min"]] = 1e99
    channel_fft[OFDM_DATA_INDEX_RANGE["max"]:] = 1e99

    decoded = (np.fft.fft(recorded_known_ofdm_blocks[5]) / channel_fft)[
        OFDM_DATA_INDEX_RANGE["min"]:OFDM_DATA_INDEX_RANGE["max"]
    ]

    if PLOTTING_ENABLED and plot:
        plot = [
            np.angle(estimate_frequency_gains_from_block(r, k, 0, False)[
                OFDM_DATA_INDEX_RANGE["min"]:OFDM_DATA_INDEX_RANGE["min"] + 100
            ])
            for r, k in zip(recorded_known_ofdm_blocks, known_ofdm_blocks)
        ]
        plt.imshow(plot, cmap="hot", interpolation="nearest")
        plt.savefig("plots/phase.png")

        plt.figure()
        plt.plot(channel_fft)
        plt.savefig("plots/channel.png")
        plt.figure()
        plt.scatter(decoded.real, decoded.imag)
        plt.axis([-1000, 1000, -1000, 1000])
        plt.gca().set_aspect("equal")
        plt.savefig("plots/constellation.png")
        plt.figure()
        plt.hist(np.abs(np.fft.fft(recorded_known_ofdm_blocks[0]) / channel_fft))
        plt.savefig("plots/hist.png")

    frequency_bin_variance = np.var(np.sqrt(2) * recorded_known_ofdm_blocks / known_ofdm_blocks, axis=0)
    normalized_variance = frequency_bin_variance / (channel_fft * np.conjugate(channel_fft))

    return channel_fft, normalized_variance.real


def estimate_frequency_gains_from_block(
    recorded_block_t: np.ndarray,
    original_block_t: np.ndarray,
    drift: float,
    plot: bool = False
):
    recorded_block = np.fft.fft(recorded_block_t, OFDM_BODY_LENGTH)
    original_block = np.fft.fft(original_block_t, OFDM_BODY_LENGTH)

    original_block[0] = 1e99
    original_block[OFDM_BODY_LENGTH // 2] = 1e99

    r = np.concatenate([
        np.arange(0,  OFDM_BODY_LENGTH // 2, 1) / OFDM_BODY_LENGTH,
        [0],
        np.arange(-1, -OFDM_BODY_LENGTH // 2, -1)[::-1] / OFDM_BODY_LENGTH
    ])

    drift_corrected = recorded_block * np.exp(2j * np.pi * drift * r)

    assert drift_corrected[OFDM_BODY_LENGTH // 2].imag == 0
    assert drift_corrected[0].imag == 0

    assert np.max(np.abs(
        drift_corrected[1:OFDM_BODY_LENGTH // 2] -
        np.conjugate(drift_corrected[OFDM_BODY_LENGTH // 2 + 1:][::-1])
    )) < 1e-10

    if PLOTTING_ENABLED and plot:
        ip = np.fft.ifft(drift_corrected / original_block, OFDM_BODY_LENGTH).real
        plt.figure(69)
        plt.plot(range(64, 128), ip[64:128], linewidth = 0.3)

    return drift_corrected / original_block
