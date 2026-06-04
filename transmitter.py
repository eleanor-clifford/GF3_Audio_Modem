# vim: set ts=4 sw=4 tw=0 et :
from config import (
    AUDIO_SCALE_FACTOR,
    CHIRP_START,
    CHIRP_END,
    CHIRP_FRAME_START,
    CHIRP_FRAME_END,
    NEXT_KNOWN_OFDM_BLOCK,
    N_KNOWN_OFDM_FRAME_START,
    N_KNOWN_OFDM_FRAME_END,
    MAX_NUMBER_OF_SYMBOLS_IN_FRAME,
    OFDM_CYCLIC_PREFIX_LENGTH,
    SAMPLE_RATE,
    TRANSMISSION_OUTPUT_DIR,
)
from common import (
    finalize_argparse_for_sounddevice,
    save_data_to_file,
    set_audio_device_or_warn,
    split_list_to_chunks_of_length,
)
from metadata import generate_bytes_for_transmission
from signal_builder import SignalBuilder
import OFDM
import ldpc_tools

import numpy as np
import sounddevice as sd

from argparse import ArgumentParser


def modulate_into_frames(ofdm_symbols):
    """
    See standard for definitions:
        https://www.overleaf.com/project/628befa0a5f784cb3d188f72
    """
    signal_builder = SignalBuilder()

    for chunk_idx, chunk in enumerate(split_list_to_chunks_of_length(
        ofdm_symbols, MAX_NUMBER_OF_SYMBOLS_IN_FRAME
    )):
        if CHIRP_FRAME_START is not None:
            signal_builder.append_signal_part(CHIRP_FRAME_START)

        if N_KNOWN_OFDM_FRAME_START > 0:
            known_block = NEXT_KNOWN_OFDM_BLOCK(chunk_idx * (MAX_NUMBER_OF_SYMBOLS_IN_FRAME + N_KNOWN_OFDM_FRAME_START + N_KNOWN_OFDM_FRAME_END))

            signal_builder.append_signal_part(known_block[-OFDM_CYCLIC_PREFIX_LENGTH:])
            for _ in range(N_KNOWN_OFDM_FRAME_START):
                signal_builder.append_signal_part(known_block)

        assert len(chunk) == MAX_NUMBER_OF_SYMBOLS_IN_FRAME
        for ofdm_symbol in chunk:
            signal_builder.append_signal_part(ofdm_symbol)

        if N_KNOWN_OFDM_FRAME_END > 0:
            known_block = NEXT_KNOWN_OFDM_BLOCK()
            signal_builder.append_signal_part(known_block[-OFDM_CYCLIC_PREFIX_LENGTH:])
            for _ in range(N_KNOWN_OFDM_FRAME_END):
                signal_builder.append_signal_part(known_block)

        if CHIRP_FRAME_END is not None:
            signal_builder.append_signal_part(CHIRP_FRAME_END)

    signal = signal_builder.get_signal()
    return signal / np.max(np.abs(signal))


def modulate_file(filename: str, file_data: bytes):
    data_for_transmission = generate_bytes_for_transmission(filename, file_data)
    transmission = ldpc_tools.encode_bytes(data_for_transmission)

    signal_builder = SignalBuilder()

    signal_builder.append_signal_part(CHIRP_START)

    all_ofdm_symbols = OFDM.modulate_bytes(transmission)
    signal_builder.append_signal_part(modulate_into_frames(all_ofdm_symbols))

    signal_builder.append_signal_part(CHIRP_END)

    return signal_builder.get_signal()


def transmit_signal(signal):
    mono_channel = np.array([signal, np.zeros(signal.size)]).transpose()
    sd.play(AUDIO_SCALE_FACTOR * mono_channel, samplerate=SAMPLE_RATE, blocking=True)


if __name__ == "__main__":
    # fmt: off
    parser = ArgumentParser(description="OFDM transmitter")
    parser.add_argument("file", help="File to transmit")
    parser.add_argument("--wave_file", "-w", action="store_true", help="Transmit a numpy waveform without additional processing")
    parser.add_argument("--silent", "-s", action="store_true", help="Modulate but don't play signal")
    parser.add_argument("--save-to", default=None, help=".npy filename (default is random)")
    args = finalize_argparse_for_sounddevice(parser)
    # fmt: on

    if not args.silent:
        set_audio_device_or_warn(args)

        if args.wave_file:
            signal_from_file = np.load(args.file)
            transmit_signal(signal_from_file)
            exit(0)

    print("Building OFDM symbols, please wait... ", flush=True)

    with open(args.file, "rb") as input_file:
        modulated_signal = modulate_file(args.file, input_file.read())

    save_data_to_file(TRANSMISSION_OUTPUT_DIR, modulated_signal, filename=args.save_to)

    if not args.silent:
        input("Press enter to play.")
        transmit_signal(modulated_signal)
