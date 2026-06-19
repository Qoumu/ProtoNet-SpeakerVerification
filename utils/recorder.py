import argparse
import sounddevice as sd
import numpy as np
import noisereduce as nr
from scipy.io.wavfile import write
from pathlib import Path

sample_rate = 16000
channels = 1
gain_db = 20

file_index = 1

parser = argparse.ArgumentParser()
parser.add_argument("--speaker_id", required=True, help="Speaker ID")
args = parser.parse_args()

speaker_id = args.speaker_id

save_dir = Path("recordings") / speaker_id
save_dir.mkdir(parents=True, exist_ok=True)

file_index = 1

def process_and_save(recording, filename):
    audio = np.concatenate(recording, axis=0).squeeze()
    audio_float = audio.astype(np.float32) / 32768.0

    gain = 10 ** (gain_db / 20)
    audio_gain = np.clip(audio_float * gain, -1.0, 1.0)

    audio_denoised = nr.reduce_noise(
        y=audio_gain,
        sr=sample_rate,
        stationary=False
    )

    write(filename, sample_rate, audio_denoised.astype(np.float32))
    print(f"Saved: {filename}")

while True:
    cmd = input("\nPress ENTER to start recording, or type q to quit: ")

    if cmd.lower() == "q":
        print("Exit.")
        break

    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())

    print("Recording... Press ENTER to stop this audio.")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        callback=callback
    ):
        input()

    filename = save_dir / f"{speaker_id}_{file_index}.wav"
    process_and_save(recording, filename)

    file_index += 1