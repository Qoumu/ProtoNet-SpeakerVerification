import random
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


class DataAugmentation:
    def __init__(
        self,
        sample_rate=16000,
        rir_dir=None,
        p=0.8,
        snr_range=(5, 20),
        waveform_dropout_range=(0.05, 0.20),
        freq_dropout_range=(0.05, 0.20),
        shift_range=(-0.5, 0.5),
        speed_range=(0.9, 1.1),
        max_rir_seconds=1.0,
    ):
        """
        sample_rate: target sample rate
        rir_dir: folder containing room impulse response wav files
        p: probability of applying augmentation
        snr_range: SNR range for Gaussian noise
        waveform_dropout_range: percentage of waveform to drop
        freq_dropout_range: percentage of frequency bins to suppress
        shift_range: random shift range in seconds, e.g. (-0.5, 0.5)
        speed_range: speed/pitch change rate, e.g. 0.9 to 1.1
        max_rir_seconds: maximum RIR tail kept for reverberation
        """

        self.sample_rate = sample_rate
        self.rir_dir = rir_dir
        self.p = p
        self.snr_range = snr_range
        self.waveform_dropout_range = waveform_dropout_range
        self.freq_dropout_range = freq_dropout_range
        self.shift_range = shift_range
        self.speed_range = speed_range
        self.max_rir_samples = (
            int(max_rir_seconds * sample_rate) if max_rir_seconds is not None else None
        )

        self.rir_files = self._load_wav_files(rir_dir)

        self.augmentations = [
            "waveform_dropout",
            "frequency_dropout",
            "reverberation",
            "gaussian_noise",
            "noise_reverberation",
            "shifting",
            "speed_change",
        ]

    def _resample_waveform(self, waveform, orig_freq, new_freq):
        """Resample a [C, T] waveform tensor."""
        if orig_freq == new_freq:
            return waveform

        return AF.resample(waveform, orig_freq, new_freq)

    def _load_wav_files(self, folder):
        if folder is None or not os.path.exists(folder):
            return []

        wav_files = []

        for root, _, files in os.walk(folder):
            for file in files:
                if not file.lower().endswith(".wav"):
                    continue

                path = os.path.join(root, file)
                # Kaldi's RIRS_NOISES package also contains long noise recordings.
                # Those are not valid impulse responses for this augmentation path.
                if "pointsource_noises" in os.path.normpath(path).split(os.sep):
                    continue

                wav_files.append(path)

        return wav_files

    def _trim_rir(self, rir):
        """Drop leading silence and cap the tail length for faster convolution."""
        if rir.numel() == 0:
            return rir

        peak_idx = torch.argmax(rir.abs()).item()
        rir = rir[:, peak_idx:]

        if self.max_rir_samples is not None and rir.shape[-1] > self.max_rir_samples:
            rir = rir[:, :self.max_rir_samples]

        return rir

    def _fft_convolve(self, waveform, rir):
        """Efficient full 1D convolution for long RIR kernels."""
        total_length = waveform.shape[-1] + rir.shape[-1] - 1
        n_fft = 1 << (total_length - 1).bit_length()

        waveform_spec = torch.fft.rfft(waveform, n=n_fft, dim=-1)
        rir_spec = torch.fft.rfft(rir, n=n_fft, dim=-1)
        reverbed = torch.fft.irfft(waveform_spec * rir_spec, n=n_fft, dim=-1)

        return reverbed[:, :total_length]

    def __call__(self, waveform):
        """
        Input:
            waveform: numpy.ndarray float32, shape [T]

        Output:
            waveform: numpy.ndarray float32, shape [T]
        """

        input_is_numpy = isinstance(waveform, np.ndarray)

        if input_is_numpy:
            waveform = torch.from_numpy(waveform.astype(np.float32))

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        original_length = waveform.shape[-1]

        if random.random() > self.p:
            output = waveform.squeeze(0)
            return output.numpy().astype(np.float32) if input_is_numpy else output

        aug_type = random.choice(self.augmentations)

        if aug_type == "waveform_dropout":
            waveform = self.waveform_dropout(waveform)

        elif aug_type == "frequency_dropout":
            waveform = self.frequency_dropout(waveform)

        elif aug_type == "reverberation":
            if len(self.rir_files) > 0:
                waveform = self.reverberation(waveform)

        elif aug_type == "gaussian_noise":
            waveform = self.gaussian_noise(waveform)

        elif aug_type == "noise_reverberation":
            if len(self.rir_files) > 0:
                waveform = self.reverberation(waveform)
            waveform = self.gaussian_noise(waveform)

        elif aug_type == "shifting":
            waveform = self.shifting(waveform)

        elif aug_type == "speed_change":
            waveform = self.speed_change(waveform)

        waveform = self.fix_length(waveform, original_length)
        waveform = self.normalize(waveform)

        waveform = waveform.squeeze(0)  # [T]

        if input_is_numpy:
            waveform = waveform.detach().cpu().numpy().astype(np.float32)

        return waveform

    def fix_length(self, waveform, target_length):
        """
        Trim or pad waveform to target length.
        """

        if waveform.shape[-1] > target_length:
            waveform = waveform[:, :target_length]

        elif waveform.shape[-1] < target_length:
            pad_len = target_length - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_len))

        return waveform

    def waveform_dropout(self, waveform):
        """
        Randomly replaces waveform chunks with zeros.
        """

        length = waveform.shape[-1]
        drop_ratio = random.uniform(*self.waveform_dropout_range)
        total_drop_samples = int(length * drop_ratio)

        num_chunks = random.randint(1, 5)
        chunk_len = max(1, total_drop_samples // num_chunks)

        waveform = waveform.clone()

        for _ in range(num_chunks):
            start = random.randint(0, max(0, length - chunk_len))
            end = start + chunk_len
            waveform[:, start:end] = 0.0

        return waveform

    def frequency_dropout(self, waveform):
        """
        Randomly masks frequency bands in the spectrum.
        This simulates random band-stop filtering.
        """

        waveform = waveform.clone()
        length = waveform.shape[-1]

        spectrum = torch.fft.rfft(waveform, dim=-1)
        num_freq_bins = spectrum.shape[-1]

        drop_ratio = random.uniform(*self.freq_dropout_range)
        drop_bins = int(num_freq_bins * drop_ratio)

        num_bands = random.randint(1, 3)
        band_width = max(1, drop_bins // num_bands)

        for _ in range(num_bands):
            start_bin = random.randint(0, max(0, num_freq_bins - band_width))
            end_bin = start_bin + band_width
            spectrum[:, start_bin:end_bin] = 0

        augmented = torch.fft.irfft(spectrum, n=length, dim=-1)

        return augmented

    def reverberation(self, waveform):
        """
        Applies reverberation by convolving waveform with a random RIR.
        """

        rir_path = random.choice(self.rir_files)
        rir_np, sr = sf.read(rir_path, dtype="float32", always_2d=False)
        rir_np = np.asarray(rir_np, dtype=np.float32)
        if rir_np.ndim > 1:
            rir_np = rir_np.mean(axis=1)
        rir = torch.from_numpy(rir_np.astype(np.float32)).unsqueeze(0)

        if sr != self.sample_rate:
            rir = self._resample_waveform(rir, sr, self.sample_rate)

        rir = self._trim_rir(rir[:1, :])
        if rir.shape[-1] == 0:
            return waveform

        rir = rir.to(device=waveform.device, dtype=waveform.dtype)
        rir = rir / (torch.norm(rir, p=2) + 1e-8)

        return self._fft_convolve(waveform, rir)

    def gaussian_noise(self, waveform):
        """
        Adds Gaussian noise with random SNR.
        No external noise file is required.
        """

        snr_db = random.uniform(*self.snr_range)

        noise = torch.randn_like(waveform)

        speech_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()

        snr_linear = 10 ** (snr_db / 10)

        scale = torch.sqrt(
            speech_power / (snr_linear * noise_power + 1e-8)
        )

        noisy_waveform = waveform + scale * noise

        return noisy_waveform

    def shifting(self, waveform):
        """
        Shift audio left or right by random seconds.

        Positive shift: move audio to the right.
        Negative shift: move audio to the left.
        Empty region is padded with zeros.
        """

        length = waveform.shape[-1]

        shift_seconds = random.uniform(*self.shift_range)
        shift_samples = int(shift_seconds * self.sample_rate)

        shifted = torch.zeros_like(waveform)

        if shift_samples > 0:
            # Shift right
            shifted[:, shift_samples:] = waveform[:, :length - shift_samples]

        elif shift_samples < 0:
            # Shift left
            shift_samples = abs(shift_samples)
            shifted[:, :length - shift_samples] = waveform[:, shift_samples:]

        else:
            shifted = waveform

        return shifted

    def speed_change(self, waveform):
        """
        Change speed and pitch by resampling.

        rate > 1.0:
            faster, shorter, higher pitch

        rate < 1.0:
            slower, longer, lower pitch
        """

        rate = random.uniform(*self.speed_range)

        original_length = waveform.shape[-1]

        new_sample_rate = int(self.sample_rate * rate)

        changed = self._resample_waveform(
            waveform,
            orig_freq=self.sample_rate,
            new_freq=new_sample_rate,
        )

        changed = self.fix_length(changed, original_length)

        return changed

    def normalize(self, waveform):
        """
        Prevent clipping after augmentation.
        """

        max_val = waveform.abs().max()

        if max_val > 1.0:
            waveform = waveform / max_val

        return waveform
