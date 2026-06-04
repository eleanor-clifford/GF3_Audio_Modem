# vim: set ts=4 sw=4 tw=0 et :
from config import (
    PLOTTING_ENABLED,
    CONSTELLATION_BITS,
    CONSTELLATION_SYMBOLS,
    N_KNOWN_OFDM_FRAME_START,
    N_KNOWN_OFDM_FRAME_END,
    MAX_NUMBER_OF_SYMBOLS_IN_FRAME,
    PAD_LAST_FRAME,
    OFDM_BODY_LENGTH,
    OFDM_CYCLIC_PREFIX_LENGTH,
    OFDM_DATA_INDEX_RANGE,
    OFDM_SYMBOL_LENGTH,
    PEAK_SUPPRESSION_STATS_ENABLED,
    PEAK_SUPPRESSION_ENABLED,
    PEAK_SUPPRESSION_IMPULSE_APPROXIMATOR,
    PEAK_SUPPRESSION_SEQUENCE,
    PEAK_SUPPRESSION_THRESH,
    SONG_ENABLED,
    SONG_BLOCK,
    NORMALISE_PER_SYMBOL,
)
from common import split_list_to_chunks_of_length

import numpy as np
import scipy

import random


def map_to_constellation_symbols(data: bytes):
    constellation_symbols = []

    current_group = 0
    current_group_size = 0
    for byte in data:
        for shift in range(8, 0, -1):
            current_group <<= 1
            current_group_size += 1

            current_group |= 1 & (byte >> (shift - 1))

            if current_group_size == CONSTELLATION_BITS:
                constellation_symbols.append(CONSTELLATION_SYMBOLS[current_group])

                current_group_size = 0
                current_group = 0

    # TODO: we can pad with zeros here
    assert current_group == 0
    assert current_group_size == 0

    return constellation_symbols


def pad_symbols(symbols, alignment):
    unmatched_symbols = len(symbols) % alignment

    padding_count = (alignment - unmatched_symbols) % alignment

    for _ in range(padding_count):
        symbols.append(random.choice(CONSTELLATION_SYMBOLS))

    return symbols


def suppress_peaks(data: np.ndarray, plot=False):
    if not PEAK_SUPPRESSION_ENABLED:
        return data

    assert data.size == OFDM_BODY_LENGTH

    # this is overwritten on each pass
    improved_time_domain_block = np.copy(data)

    for pass_idx, (thresh_std_devs, SAMPLE_VIEW_RANGE, IMPULSE_SHIFT_RANGE) in enumerate(PEAK_SUPPRESSION_SEQUENCE):

        peak_thresh = thresh_std_devs * np.sqrt(improved_time_domain_block.var())

        window_chunks = []  # chunks of the original signal with the peaks in them
        for sample_idx, sample in enumerate(improved_time_domain_block):
            if abs(sample) > peak_thresh:
                # expand a window if this sample is in one, else make a new window
                for i, (peak, c_min, c_max) in enumerate(window_chunks):
                    if c_min <= sample_idx + SAMPLE_VIEW_RANGE \
                            and sample_idx - SAMPLE_VIEW_RANGE <= c_max:
                        window_chunks[i] = (
                            max(abs(sample), peak),
                            min(c_min, sample_idx - SAMPLE_VIEW_RANGE),
                            max(c_max, sample_idx + SAMPLE_VIEW_RANGE)
                        )
                        break
                else:
                    window_chunks.append((
                        abs(sample),
                        sample_idx - SAMPLE_VIEW_RANGE,
                        sample_idx + SAMPLE_VIEW_RANGE
                    ))

        if len(window_chunks) == 0:
            # can't believe this edge case actually happened
            # (no peaks above the threshold)
            continue

        # combine edge windows (note the time domain is cyclic)
        first_window_peak, first_window_left, first_window_right = window_chunks[0]
        last_window_peak, last_window_left, last_window_right = window_chunks[-1]

        if last_window_right - OFDM_BODY_LENGTH >= first_window_left:
            # if they overlap, combine them on the left
            window_chunks[0] = (
                max(first_window_peak, last_window_peak),
                last_window_left - OFDM_BODY_LENGTH,
                first_window_right
            )
            window_chunks.pop(-1)

        # suppress the biggest peaks first
        window_chunks.sort(reverse=True)

        for _, c_min, c_max in window_chunks:
            shifted_impulses = []
            for s in range(
                c_min + (SAMPLE_VIEW_RANGE - IMPULSE_SHIFT_RANGE),
                c_max - (SAMPLE_VIEW_RANGE - IMPULSE_SHIFT_RANGE)
            ):
                shifted_impulses.append(np.roll(PEAK_SUPPRESSION_IMPULSE_APPROXIMATOR, s - OFDM_BODY_LENGTH // 2))

            # deal with the edge peak case
            if c_min < 0:
                cut_time_domain_target = np.concatenate([
                        -improved_time_domain_block[c_min + OFDM_BODY_LENGTH:],
                        -improved_time_domain_block[:c_max],
                    ])
                cut_shifted_impulses = [
                        np.concatenate([
                            imp[c_min + OFDM_BODY_LENGTH:],
                            imp[:c_max],
                        ])
                    for imp in shifted_impulses
                ]
            else:
                cut_time_domain_target = -improved_time_domain_block[c_min:c_max]
                cut_shifted_impulses = [imp[c_min:c_max] for imp in shifted_impulses]

            # get rid of the imaginary parts (they should be very close to zero)
            shifted_impulses = np.array(shifted_impulses).real
            cut_shifted_impulses = np.array(cut_shifted_impulses).real
            cut_time_domain_target = cut_time_domain_target.real

            # take least squares as first-order approximation of the suppressor
            initial_coeffs, _, _, _ = scipy.linalg.lstsq(cut_shifted_impulses.T, cut_time_domain_target)

            # get a better approximation with maximum difference, which
            # is the correct cost function for maximum peak suppression

            def max_diff(x):
                return np.max(np.abs(cut_shifted_impulses.T @ x - cut_time_domain_target))
            res = scipy.optimize.minimize(max_diff, initial_coeffs)
            coeffs = res.x

            maybe_improved_block = improved_time_domain_block + shifted_impulses.T @ coeffs
            local_improvement = -cut_time_domain_target + cut_shifted_impulses.T @ coeffs

            least_squares_improvement = 100 - 100 * np.max(np.abs(-cut_time_domain_target + cut_shifted_impulses.T @ initial_coeffs)) \
                / np.max(np.abs(improved_time_domain_block))
            opt_improvement = 100 - 100 * np.max(np.abs(-cut_time_domain_target + cut_shifted_impulses.T @ coeffs)) \
                / np.max(np.abs(improved_time_domain_block))


            if PLOTTING_ENABLED and plot and pass_idx == 0:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(-cut_time_domain_target, label="Original")
                plt.plot(-cut_time_domain_target + cut_shifted_impulses.T @ initial_coeffs, label=f"Least squares, {least_squares_improvement:.2f}% reduction")
                plt.plot(-cut_time_domain_target + cut_shifted_impulses.T @ coeffs, label=f"Maximum difference optimisation, {opt_improvement:.2f}% reduction")
                plt.legend()
                plt.xlabel("Time domain sample in window")
                plt.ylabel("Amplitude")
                plt.savefig('plots/optimisation.pdf')
                plt.savefig('plots/optimisation.pgf')

                plt.figure()
                for idx, (imp, c) in enumerate(zip(cut_shifted_impulses, coeffs)):
                    if idx % 4 == 3:
                        plt.plot(imp * c)
                plt.xlabel("Time domain sample in window")
                plt.ylabel("Amplitude")
                plt.savefig('plots/pseudo-impulse.pdf')
                plt.savefig('plots/pseudo-impulse.pgf')

            if np.max(np.abs(maybe_improved_block)) <= np.max(np.abs(improved_time_domain_block)) \
                    and np.max(np.abs(local_improvement)) < np.max(np.abs(cut_time_domain_target)):
                # only update the block if we've actually made an improvement
                # (this method can be unstable)
                improved_time_domain_block = maybe_improved_block

    return improved_time_domain_block


def modulate_bytes(data: bytes):
    length_of_data_per_ofdm_block = (
        OFDM_DATA_INDEX_RANGE["max"] - OFDM_DATA_INDEX_RANGE["min"]
    )

    constellation_symbols = map_to_constellation_symbols(data)
    constellation_symbols = pad_symbols(
        constellation_symbols, length_of_data_per_ofdm_block
    )

    data_blocks = np.split(
        np.array(constellation_symbols),
        len(constellation_symbols) // length_of_data_per_ofdm_block,
    )

    if PAD_LAST_FRAME:
        n_pad_blocks = MAX_NUMBER_OF_SYMBOLS_IN_FRAME - len(data_blocks) % MAX_NUMBER_OF_SYMBOLS_IN_FRAME
        for _ in range(n_pad_blocks):
            data_blocks.append(np.array([
                random.choice(CONSTELLATION_SYMBOLS)
                for _ in range(length_of_data_per_ofdm_block)
            ]))

    avg_suppression = 0
    original_papr = []
    suppressed_papr = []

    ofdm_symbols = []
    plotted = False
    for block_idx, block in enumerate(data_blocks):
        # Information is only mapped to frequency bins 1 to 511.
        # Frequency bins 513 to 1023 contain the reverse ordered conjugate
        # complex values of frequency bins 1 to 511. Frequency bins 0 and 512
        # contain 0 (no information, value 0.) This all ensures that the
        # output of the OFDM modulator is a real (baseband) vector.

        if SONG_ENABLED:
            fun_block = np.zeros(OFDM_DATA_INDEX_RANGE["min"] - 1)

            # account for known ofdm blocks
            song_block_idx = (
                (MAX_NUMBER_OF_SYMBOLS_IN_FRAME + N_KNOWN_OFDM_FRAME_START + N_KNOWN_OFDM_FRAME_END)
                * (block_idx // MAX_NUMBER_OF_SYMBOLS_IN_FRAME)
                + block_idx % MAX_NUMBER_OF_SYMBOLS_IN_FRAME
                + N_KNOWN_OFDM_FRAME_START
            )

            fun_block = SONG_BLOCK(song_block_idx)
            if NORMALISE_PER_SYMBOL:
                fun_block *= np.max(np.abs(block))
        else:
            fun_block = np.random.choice(list(CONSTELLATION_SYMBOLS.values()), OFDM_DATA_INDEX_RANGE["min"] - 1)


        np.random.seed(0)
        block_with_all_freqs = np.concatenate([
            fun_block,
            block,
            np.random.choice(list(CONSTELLATION_SYMBOLS.values()), OFDM_BODY_LENGTH // 2 - OFDM_DATA_INDEX_RANGE["max"])
        ])

        full_block_with_all_freqs = np.concatenate(
            [[0], block_with_all_freqs, [0], np.conjugate(block_with_all_freqs[::-1])]
        )
        assert full_block_with_all_freqs.size == OFDM_BODY_LENGTH

        time_domain_block_with_peaks = np.fft.ifft(full_block_with_all_freqs, OFDM_BODY_LENGTH)
        improved_time_domain_block = suppress_peaks(time_domain_block_with_peaks, plot=(block_idx == 15))

        if PEAK_SUPPRESSION_STATS_ENABLED:
            original_peak = np.max(np.abs(time_domain_block_with_peaks))
            improved_peak = np.max(np.abs(improved_time_domain_block))
            original_papr.append(original_peak**2 / time_domain_block_with_peaks.var())
            suppressed_papr.append(improved_peak**2 / improved_time_domain_block.var())
            suppression_prc = 100 - 100 * improved_peak / original_peak
            avg_suppression += suppression_prc
            print(f"[{block_idx + 1}/{len(data_blocks)}] "
                  f"Peak suppression: {suppression_prc:.2f}%, "
                  f"PAPR: {original_papr[-1]:.2f} -> {suppressed_papr[-1]:.2f}")

            if PLOTTING_ENABLED and not plotted and np.max(np.abs(fun_block)) < 1e-10:
                import matplotlib.pyplot as plt
                plotted = True
                plt.figure()
                plt.plot(time_domain_block_with_peaks, label="Unsuppressed")
                plt.plot(improved_time_domain_block, label="Suppressed")
                plt.xlabel("Time domain sample")
                plt.ylabel("Magnitude")
                plt.legend()
                plt.savefig("plots/suppression_time.pdf")
                plt.savefig("plots/suppression_time.pgf")

                plt.figure()

                plt.plot(np.fft.fft(improved_time_domain_block, OFDM_BODY_LENGTH), label="Suppressed", color='C1')
                plt.plot(np.fft.fft(time_domain_block_with_peaks, OFDM_BODY_LENGTH), label="Unsuppressed", color='C0')
                plt.xlabel("Frequency bin")
                plt.ylabel("Magnitude")
                plt.legend()
                plt.savefig("plots/suppression_freq.pdf")
                plt.savefig("plots/suppression_freq.pgf")

        block_with_cyclic_prefix = np.concatenate(
            [improved_time_domain_block[-OFDM_CYCLIC_PREFIX_LENGTH:], improved_time_domain_block]
        )

        if NORMALISE_PER_SYMBOL:
            block_with_cyclic_prefix /= np.max(np.abs(block_with_cyclic_prefix.real))

        # Ensure imaginary component is zero
        assert np.max(block_with_cyclic_prefix.imag) < 1e-10
        ofdm_symbols.append(block_with_cyclic_prefix.real)

    if PEAK_SUPPRESSION_STATS_ENABLED:
        avg_suppression /= len(data_blocks)
        print(f"Average peak suppression: {avg_suppression:.2f}%", flush=True)
        if PLOTTING_ENABLED:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(original_papr, label="Unsuppressed")
            plt.plot(suppressed_papr, label="Suppressed")
            plt.title(r"Peak-to-Average Power Ratio, defined as $max(X^2)/Var(X)$")
            plt.axhline(PEAK_SUPPRESSION_THRESH**2, color='C2', linestyle='--')
            plt.text(10, PEAK_SUPPRESSION_THRESH**2 + 0.5, "Peak detection threshold", color='C2')
            plt.xlabel("OFDM symbol index")
            plt.ylabel("PAPR")
            plt.legend()
            plt.savefig("plots/papr_reduction.pdf")
            plt.savefig("plots/papr_reduction.pgf")

    return ofdm_symbols


def demodulate_signal(channel_coefficients_fft: np.ndarray, signal: np.ndarray,  normalized_variance_start: np.ndarray, drift_per_sample: float):
    ofdm_blocks = list(split_list_to_chunks_of_length(signal, OFDM_SYMBOL_LENGTH))

    output_llr = []
    frame_offset_drift = drift_per_sample * (N_KNOWN_OFDM_FRAME_START * OFDM_BODY_LENGTH + OFDM_CYCLIC_PREFIX_LENGTH)
    total_drift_in_data_symbols = drift_per_sample * len(ofdm_blocks) * OFDM_SYMBOL_LENGTH
    drifts = np.linspace(frame_offset_drift, frame_offset_drift + total_drift_in_data_symbols, len(ofdm_blocks))
    for drift, block in zip(drifts, ofdm_blocks):
        block_without_cyclic_prefix = block[OFDM_CYCLIC_PREFIX_LENGTH:]
        dft_of_block = np.fft.fft(block_without_cyclic_prefix, OFDM_BODY_LENGTH)

        equalized_dft = dft_of_block / channel_coefficients_fft
        equalized_dft *= np.exp(2j * np.pi * drift * np.linspace(0, 1, OFDM_BODY_LENGTH))

        for index, (noisy_symbol, var) in enumerate(zip(equalized_dft, normalized_variance_start)):
            if index < OFDM_DATA_INDEX_RANGE["min"]:
                continue  # These frequencies contain no data

            if index >= OFDM_DATA_INDEX_RANGE["max"]:
                break

            if var == 0:
                # when testing with no channel
                output_llr.append(np.sign(noisy_symbol.imag) * 1e10)
                output_llr.append(np.sign(noisy_symbol.real) * 1e10)
            else:
                output_llr.append(noisy_symbol.imag / var)
                output_llr.append(noisy_symbol.real / var)

    return output_llr
