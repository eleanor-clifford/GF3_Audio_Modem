import LDPC.ldpc as ldpc

import numpy as np
from scipy import signal

from pathlib import Path

PLOTTING_ENABLED = False
PLOT_THEME = 'dracula'

if PLOTTING_ENABLED:
	from os import mkdir
	try:
		mkdir('plots')
	except FileExistsError:
		pass
	import matplotlib
	import matplotlib.pyplot as plt

	matplotlib.use("pgf")
	matplotlib.rcParams.update({
		"pgf.texsystem": "pdflatex",
		'font.family': 'serif',
		'text.usetex': True,
		'pgf.rcfonts': False,
	})
	plt.style.use(PLOT_THEME)


def get_index_of_frequency(f):
    return int(round(f * OFDM_BODY_LENGTH / SAMPLE_RATE))


AUDIO_SCALE_FACTOR = 0.3
SAMPLE_RATE = 48_000
MAX_RECORDING_DURATION = 5 * 60  # seconds
RECORDING_OUTPUT_DIR = Path("recordings")
TRANSMISSION_OUTPUT_DIR = Path("transmissions")

CHIRP_DURATION = 0.5  # Seconds
CHIRP_MIN_FREQUENCY = 0  # Hz
CHIRP_MAX_FREQUENCY = 3_000  # Hz

t = np.linspace(0, CHIRP_DURATION, int(CHIRP_DURATION * SAMPLE_RATE))
CHIRP_END = np.sin(np.pi * (CHIRP_MIN_FREQUENCY + (CHIRP_MAX_FREQUENCY - CHIRP_MIN_FREQUENCY) * t / CHIRP_DURATION) * t)
CHIRP_START = CHIRP_END[::-1]

CHIRP_FRAME_START = None
CHIRP_FRAME_END = None

OFDM_BODY_LENGTH = 1 << 13
OFDM_CYCLIC_PREFIX_LENGTH = 1 << 11
OFDM_SYMBOL_LENGTH = OFDM_BODY_LENGTH + OFDM_CYCLIC_PREFIX_LENGTH
OFDM_DATA_INDEX_RANGE = {  # following python standard range convention
    "min": 200,
    "max": 2143 + 1,
}

NORMALISE_PER_SYMBOL = False

# TODO: fill last frame
MAX_NUMBER_OF_SYMBOLS_IN_FRAME = 4
PAD_LAST_FRAME = True
N_KNOWN_OFDM_FRAME_START = 1
N_KNOWN_OFDM_FRAME_END = 0
_known_ofdm_rng = np.random.default_rng(42)


def NEXT_KNOWN_OFDM_BLOCK(block_idx):
	bits = _known_ofdm_rng.integers(0, 2, 2 * (OFDM_BODY_LENGTH // 2 - 1))
	data = []
	for i in range(0, len(bits), 2):
		data.append(CONSTELLATION_SYMBOLS[bits[i] << 1 | bits[i + 1]])


	symbol_fft = np.concatenate([[0], data, [0], np.conjugate(data[::-1])])

	if SONG_ENABLED and block_idx != -1:
		song_block = SONG_BLOCK(block_idx)
		if NORMALISE_PER_SYMBOL:
			song_block *= np.max(np.abs(symbol_fft))

		symbol_fft[1:OFDM_DATA_INDEX_RANGE["min"]] = song_block

	symbol = np.fft.ifft(symbol_fft, OFDM_BODY_LENGTH).real

	if NORMALISE_PER_SYMBOL:
		symbol /= np.max(np.abs(symbol))

	return symbol


CONSTELLATION_BITS = 2
CONSTELLATION_SYMBOLS = {
    0b00: +1 + 1j,
    0b01: +1 - 1j,
    0b10: -1 + 1j,
    0b11: -1 - 1j,
}

LDPC_CODER = ldpc.code(standard='802.11n', z=81, rate='2/3')

PEAK_SUPPRESSION_STATS_ENABLED = True
PEAK_SUPPRESSION_ENABLED = True
PEAK_SUPPRESSION_THRESH = 3.5  # stddevs

# Peak suppression stuff {{{
_perfect_time_impulse = np.zeros(OFDM_BODY_LENGTH)
_perfect_time_impulse[OFDM_BODY_LENGTH // 2] = 1
_fft_impulse_approximator = np.fft.fft(_perfect_time_impulse)
_fft_impulse_approximator[OFDM_DATA_INDEX_RANGE["min"]:OFDM_DATA_INDEX_RANGE["max"]] = 0
_fft_impulse_approximator[-OFDM_DATA_INDEX_RANGE["max"]:-OFDM_DATA_INDEX_RANGE["min"]] = 0

PEAK_SUPPRESSION_IMPULSE_APPROXIMATOR = np.fft.ifft(_fft_impulse_approximator, OFDM_BODY_LENGTH)
PEAK_SUPPRESSION_SEQUENCE = [
    # sequence of passes through peak suppression algorithm
    # peak detection threshold (in stddevs), sample view range, impulse shift range
    # this should be tweaked more!!
    # (PEAK_SUPPRESSION_THRESH, 500, 40),
    (PEAK_SUPPRESSION_THRESH, 300, 30),
    (PEAK_SUPPRESSION_THRESH, 100, 20),
    (PEAK_SUPPRESSION_THRESH, 50, 13),
    (PEAK_SUPPRESSION_THRESH, 50, 13),
    (PEAK_SUPPRESSION_THRESH, 15, 8),
    (PEAK_SUPPRESSION_THRESH, 10, 5),
    (PEAK_SUPPRESSION_THRESH, 5, 3),
    (PEAK_SUPPRESSION_THRESH, 4, 2),
    (PEAK_SUPPRESSION_THRESH, 5, 3),
]

# }}}

SONG_ENABLED = True
SONG_VOLUME = 80


def SONG_BLOCK(block_idx):
	fun_block = np.zeros(OFDM_DATA_INDEX_RANGE["min"] - 1)

	song_idx = 0
	for note_len, notes in SONG:
		if song_idx <= block_idx % SONG_LEN < song_idx + note_len:
			if len(notes) > 0 and notes[0] == "sequential":
				fun_block = SPLIT_SONG_BLOCK(notes[1:])
			else:
				for note in notes:
					freq = SONG_NOTES[note]
					fun_block[get_index_of_frequency(freq)] += 1
		song_idx += note_len

	fun_block *= SONG_VOLUME
	return fun_block


def SPLIT_SONG_BLOCK(notes):
	# ok ummm
	assert len(notes) == 2
	perfect_time_domain = []
	for n in notes:
		full = np.zeros(OFDM_BODY_LENGTH)
		if n:
			full[get_index_of_frequency(SONG_NOTES[n])] = 1
		t = np.fft.ifft(full, OFDM_BODY_LENGTH).real
		perfect_time_domain.append(t[:len(t) // len(notes)])
	perfect_time_domain = np.concatenate(perfect_time_domain)

	assert len(perfect_time_domain) == OFDM_BODY_LENGTH
	perfect_freq_domain = np.fft.fft(perfect_time_domain, OFDM_BODY_LENGTH)
	return perfect_freq_domain[1:OFDM_DATA_INDEX_RANGE["min"]]


# Song stuff {{{
SONG_NOTES = {
    "G4": 392,
    "A4": 440,
    "B4": 493.88,
    "C#5": 554.37,
    "D5": 587.33,
    "E5": 659.29,
    "F#5": 739.99,
    "G5": 783.99,
    "A5": 880,
}
SONG_OLD = [
    (6, ("G4", "B4", "D5")),
    (6, ("A4", "C#5", "E5")),
    (4, ("A4",)),
    # --
    (6, ("A4", "C#5", "E5")),
    (6, ("B4", "D5", "F#5")),
    (1, ("A5",)),
    (1, ("G5",)),
    (1, ("F#5",)),
    (1, ("D5",)),
    # --
    (6, ("G4", "B4", "D5")),
    (6, ("A4", "C#5", "E5")),
    (4, ("A4",)),
    # --
    (6, ("A4",)),
    (4, tuple()),
    (1, ("A4",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, tuple()),
    (1, ("B4",)),
    # --
    (6, ("G4", "B4", "D5")),
    (6, ("A4", "C#5", "E5")),
    (4, ("A4",)),
    # --
    (6, ("A4", "C#5", "E5")),
    (6, ("B4", "D5", "F#5")),
    (1, ("A5",)),
    (1, ("G5",)),
    (1, ("F#5",)),
    (1, ("D5",)),
    # --
    (6, ("G4", "B4", "D5")),
    (6, ("A4", "C#5", "E5")),
    (4, ("A4",)),
    # --
    (6, ("A4",)),
    (4, tuple()),
    (1, ("A4",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, tuple()),
    (1, ("B4",)),
    # --
    (4, ("G4", "B4", "D5")),
    (2, ("G4", "B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("D5",)),
    (2, ("E5",)),
    (2, ("C#5",)),
    # --
    (1, ("C#5",)),
    (1, ("B4",)),
    (14, ("A4",)),
    # --
    (2, tuple()),
    (2, ("B4",)),
    (2, ("B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("B4",)),
    (2, tuple()),
    (2, ("A4",)),
    # --
    (4, ("A5",)),
    (2, ("A5",)),
    (10, ("E5",)),
    # --
    (2, tuple()),
    (2, ("B4",)),
    (2, ("B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("B4",)),
    (2, ("D5",)),
    (2, ("E5",)),
    # --
    (2, tuple()),
    (2, ("C#5",)),
    (2, ("B4",)),
    (1, ("C#5",)),
    (1, ("B4",)),
    (8, ("A4",)),
    # --
    (2, tuple()),
    (2, ("B4",)),
    (2, ("B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("B4",)),
    (4, ("A4",)),
    # --
    (2, ("E5",)),
    (2, ("E5",)),
    (2, ("E5",)),
    (2, ("F#5",)),
    (8, ("E5",)),
    # --
    (10, ("D5",)),
    (2, ("E5",)),
    (2, ("F#5",)),
    (2, ("D5",)),
    # --
    (2, ("E5",)),
    (2, ("E5",)),
    (2, ("E5",)),
    (2, ("F#5",)),
    (2, ("E5",)),
    (2, ("A4",)),
    (4, ("A4",)),
    # --
    (8, tuple()),
    (2, ("B4",)),
    (2, ("C#5",)),
    (2, ("D5",)),
    (2, ("B4",)),
    # --
    (2, tuple()),
    (2, ("E5",)),
    (2, ("F#5",)),
    (6, ("E5",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (2, ("F#5",)),
    (1, tuple()),
    (2, ("F#5",)),
    (1, tuple()),
    (6, ("E5",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (2, ("E5",)),
    (1, tuple()),
    (2, ("E5",)),
    (1, tuple()),
    (3, ("D5",)),
    (1, ("C#5",)),
    (2, ("B4",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (4, ("D5",)),
    (2, ("E5",)),
    (3, ("C#5",)),
    (1, ("B4",)),
    (3, ("A4",)),
    (1, tuple()),
    (2, ("A4",)),
    # --
    (4, ("E5",)),
    (8, ("D5",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (2, ("F#5",)),
    (1, tuple()),
    (2, ("F#5",)),
    (1, tuple()),
    (6, ("E5",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (4, ("A5",)),
    (2, ("C#5",)),
    (3, ("D5",)),
    (1, ("C#5",)),
    (2, ("B4",)),
    (1, ("A4",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (4, ("D5",)),
    (2, ("E5",)),
    (3, ("C#5",)),
    (1, ("B4",)),
    (3, ("A4",)),
    (1, tuple()),
    (2, ("A4",)),
    # --
    (4, ("E5",)),
    (8, ("D5",)),
    (4, tuple()),
]
SONG = [
    (3, ("G4", "B4", "D5")),
    (3, ("A4", "C#5", "E5")),
    (2, ("A4",)),
    # --
    (3, ("A4", "C#5", "E5")),
    (3, ("B4", "D5", "F#5")),
    (1, ("sequential", "A5", "G5",)),
    (1, ("sequential", "F#5", "D5")),
    # --
    (3, ("G4", "B4", "D5")),
    (3, ("A4", "C#5", "E5")),
    (2, ("A4",)),
    # --
    (3, ("A4",)),
    (2, tuple()),
    (1, ("sequential", "A4", "A4")),
    (1, ("sequential", "B4", "D5")),
    (1, ("sequential", "", "B4",)),
    # --
    (3, ("G4", "B4", "D5")),
    (3, ("A4", "C#5", "E5")),
    (2, ("A4",)),
    # --
    (3, ("A4", "C#5", "E5")),
    (3, ("B4", "D5", "F#5")),
    (1, ("sequential", "A5", "G5")),
    (1, ("sequential", "F#5", "D5")),
    # --
    (3, ("G4", "B4", "D5")),
    (3, ("A4", "C#5", "E5")),
    (2, ("A4",)),
    # --
    (3, ("A4",)),
    (2, tuple()),
    (1, ("sequential", "A4", "A4")),
    (1, ("sequential", "B4", "D5")),
    (1, ("sequential", "", "B4",)),
    # --
    (2, ("G4", "B4", "D5")),
    (1, ("G4", "B4",)),
    (1, ("C#5",)),
    (1, ("D5",)),
    (1, ("D5",)),
    (1, ("E5",)),
    (1, ("C#5",)),
    # --
    (1, ("sequential", "C#5", "B4")),
    (7, ("A4",)),
    # --
    (1, tuple()),
    (1, ("B4",)),
    (1, ("B4",)),
    (1, ("C#5",)),
    (1, ("D5",)),
    (1, ("B4",)),
    (1, tuple()),
    (1, ("A4",)),
    # --
    (2, ("A5",)),
    (1, ("A5",)),
    (5, ("E5",)),
    # --
    (1, tuple()),
    (1, ("B4",)),
    (1, ("B4",)),
    (1, ("C#5",)),
    (1, ("D5",)),
    (1, ("B4",)),
    (1, ("D5",)),
    (1, ("E5",)),
    # --
    (1, tuple()),
    (1, ("C#5",)),
    (1, ("B4",)),
    (1, ("sequential", "C#5", "B4")),
    (4, ("A4",)),
    # --
    (1, tuple()),
    (1, ("B4",)),
    (1, ("B4",)),
    (1, ("C#5",)),
    (1, ("D5",)),
    (1, ("B4",)),
    (2, ("A4",)),
    # --
    (1, ("E5",)),
    (1, ("E5",)),
    (1, ("E5",)),
    (1, ("F#5",)),
    (4, ("E5",)),
    # --
    (5, ("D5",)),
    (1, ("E5",)),
    (1, ("F#5",)),
    (1, ("D5",)),
    # --
    (1, ("E5",)),
    (1, ("E5",)),
    (1, ("E5",)),
    (1, ("F#5",)),
    (1, ("E5",)),
    (1, ("A4",)),
    (2, ("A4",)),
    # --
    (4, tuple()),
    (1, ("B4",)),
    (1, ("C#5",)),
    (1, ("D5",)),
    (1, ("B4",)),
    # --
    (1, tuple()),
    (1, ("E5",)),
    (1, ("F#5",)),
    (3, ("E5",)),
    (1, ("sequential", "A4", "B4")),
    (1, ("sequential", "D5", "B4")),
    # --
    (1, ("F#5",)),
    (1, ("sequential", "", "F#5")),
    (1, ("sequential", "F#5", "")),
    (3, ("E5",)),
    (1, ("sequential", "A4", "B4")),
    (1, ("sequential", "D5", "B4")),
    # --
    (1, ("E5",)),
    (1, ("sequential", "", "E5")),
    (1, ("sequential", "E5", "")),
    (1, ("D5",)),
    (1, ("sequential", "D5", "C#5")),
    (1, ("B4",)),
    (1, ("sequential", "A4", "B4")),
    (1, ("sequential", "D5", "B4")),
    # --
    (2, ("D5",)),
    (1, ("E5",)),
    (1, ("C#5",)),
    (1, ("sequential", "C#5", "B4")),
    (1, ("A4",)),
    (1, ("sequential", "A4", "")),
    (1, ("A4",)),
    # --
    (2, ("E5",)),
    (4, ("D5",)),
    (1, ("sequential", "A4", "B4")),
    (1, ("sequential", "D5", "B4")),
    # --
    (1, ("F#5",)),
    (1, ("sequential", "", "F#5")),
    (1, ("sequential", "F#5", "")),
    (3, ("E5",)),
    (1, ("sequential", "A4", "B4")),
    (1, ("sequential", "D5", "B4")),
    # --
    (2, ("A5",)),
    (1, ("C#5",)),
    (1, ("D5",)),
    (1, ("sequential", "D5", "C#5")),
    (1, ("B4",)),
    (1, ("sequential", "A4", "B4")),
    (1, ("sequential", "D5", "B4")),
    # --
    (2, ("D5",)),
    (1, ("E5",)),
    (1, ("C#5",)),
    (1, ("sequential", "C#5", "B4")),
    (1, ("A4",)),
    (1, ("sequential", "A4", "")),
    (1, ("A4",)),
    # --
    (2, ("E5",)),
    (4, ("D5",)),
    (2, tuple()),
]

SONG_LEN = sum((x for x, _ in SONG))
# }}}
