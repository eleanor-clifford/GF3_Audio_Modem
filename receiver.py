#  vim: set ts=4 sw=4 tw=0 et :
from OFDM import demodulate_signal
from config import (
    PLOTTING_ENABLED,
    OFDM_BODY_LENGTH,
    OFDM_SYMBOL_LENGTH,
    OFDM_CYCLIC_PREFIX_LENGTH,
    PLOTTING_ENABLED,
    SAMPLE_RATE,
    MAX_RECORDING_DURATION,
    RECORDING_OUTPUT_DIR,
    N_KNOWN_OFDM_FRAME_START,
    N_KNOWN_OFDM_FRAME_END,
    NEXT_KNOWN_OFDM_BLOCK,
    MAX_NUMBER_OF_SYMBOLS_IN_FRAME,
)
from common import (
    save_data_to_file,
    set_audio_device_or_warn,
    finalize_argparse_for_sounddevice,
)
from error_stats import bit_error, plot_cumulative_error
from metadata import decode_received_file
from synchronization_estimation import (
    crop_frame_into_parts,
    # crop_signal_into_overlapping_frames,
    synchronise_crop,
    estimate_channel_coefficients_and_variance,
)
from ldpc_tools import decode_from_llr


import numpy as np
from scipy.io import wavfile
import sounddevice as sd
from dvg_ringbuffer import RingBuffer

from argparse import ArgumentParser
from pathlib import Path
import sys

if PLOTTING_ENABLED:
    import matplotlib.pyplot as plt


def receive_signal(signal):

    # signal_frames = crop_signal_into_overlapping_frames(signal)
    symbols = synchronise_crop(signal)

    assert N_KNOWN_OFDM_FRAME_START == 1
    assert N_KNOWN_OFDM_FRAME_END == 0

    received_known_symbols = symbols[range(0, len(symbols), MAX_NUMBER_OF_SYMBOLS_IN_FRAME + 1)]
    data_symbols = symbols[np.arange(0, len(symbols)) % (MAX_NUMBER_OF_SYMBOLS_IN_FRAME + 1) != 0]
    assert len(data_symbols) + len(received_known_symbols) == len(symbols)

    known_blocks = np.array([NEXT_KNOWN_OFDM_BLOCK(-1) for _ in range(len(received_known_symbols))])

    received_known_blocks = received_known_symbols[:, OFDM_CYCLIC_PREFIX_LENGTH:]
    channel_coefficients, normalised_variance = estimate_channel_coefficients_and_variance(received_known_blocks, known_blocks, 0, 0, False)

    llr_for_each_bit = demodulate_signal(channel_coefficients, np.concatenate(data_symbols), normalised_variance, 0)
    decoded_bytes = decode_from_llr(np.array(llr_for_each_bit))
    return decode_received_file(decoded_bytes)


def record_until_enter_key():
    buffer = RingBuffer(SAMPLE_RATE * MAX_RECORDING_DURATION)

    def record_callback(in_data, frames, time, status):
        buffer.extend(in_data.flatten())
        if status:
            print(status, file=sys.stderr)

    with sd.InputStream(callback=record_callback, channels=1, samplerate=SAMPLE_RATE):
        input("Press enter to stop recording")

    save_data_to_file(RECORDING_OUTPUT_DIR, buffer)
    return np.array(buffer)


if __name__ == "__main__":
    parser = ArgumentParser(description="OFDM receiver")
    parser.add_argument("file", nargs="?", help="Numpy waveform file")
    parser.add_argument("--expected_output", help="Expected transmission")
    parser.add_argument("--wav", action="store_true")
    args = finalize_argparse_for_sounddevice(parser)

    if args.file is not None:
        if args.wav:
            _, recorded_signal = wavfile.read(args.file)
        else:
            recorded_signal = np.load(args.file)
    else:
        set_audio_device_or_warn(args)
        recorded_signal = record_until_enter_key()

    filename, demodulated_file = receive_signal(recorded_signal)

    if args.expected_output is not None:
        with open(args.expected_output, "rb") as expected_file:
            expected_bytes = expected_file.read()

        demodulated_file = demodulated_file[: len(expected_bytes)]
        print(
            f"[INFO] Bit error of received file: {bit_error(demodulated_file, expected_bytes)}"
        )
        plot_cumulative_error(demodulated_file, expected_bytes)


    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    open(filename, "wb").write(demodulated_file)
