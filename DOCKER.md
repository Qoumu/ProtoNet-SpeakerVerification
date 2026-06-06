# Docker Usage

This container is set up for host-side PyTorch enrollment and verification with:

- model: `output/ECAPATDNN_DataAug_protonet_model.pth`
- default enrolled store: `output/enrolled_speakers.pt`
- default command: `verify`
- target deploy platform: Raspberry Pi 5 (`linux/arm64`)

The runtime image includes:

- `libsndfile1` for audio file loading
- `libgomp1` because `torchaudio` requires `libgomp.so.1` on Debian-based images

## Build

```bash
docker build -t protonet-sr .
```

Or use the helper script for either native or Raspberry Pi 5 builds:

```bash
chmod +x scripts/docker-build.sh
./scripts/docker-build.sh local
./scripts/docker-build.sh rpi5
```

Behavior:

- `local`: uses `docker build` and tags `protonet-sr:local`
- `rpi5`: uses `docker buildx build --platform linux/arm64 --load` and tags `protonet-sr:rpi5`

## Build And Push To Docker Hub For Raspberry Pi 5

Raspberry Pi 5 should run a 64-bit OS, so publish an `linux/arm64` image.

If you build `linux/arm64` from an `amd64` machine, Buildx usually uses QEMU emulation.
That works, but PyTorch wheel downloads are large and cross-builds are more fragile.
If you keep seeing network-related `pip` failures, build natively on the Pi or on an ARM64 CI runner instead.

Replace `DOCKERHUB_USER` with your Docker Hub namespace:

```bash
docker login
docker buildx create --use --name protonet-builder
docker buildx build \
  --platform linux/arm64 \
  -t DOCKERHUB_USER/protonet-sr:latest \
  -t DOCKERHUB_USER/protonet-sr:rpi5 \
  --push .
```

Or use the helper script:

```bash
chmod +x scripts/docker-buildx-push.sh
DOCKERHUB_USER=yourname ./scripts/docker-buildx-push.sh
```

If you also want an amd64 image for development machines:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t DOCKERHUB_USER/protonet-sr:latest \
  --push .
```

Verify the published manifest:

```bash
docker buildx imagetools inspect DOCKERHUB_USER/protonet-sr:latest
```

Pull on the Pi:

```bash
docker pull DOCKERHUB_USER/protonet-sr:latest
```

## Verify Against The Prepared `.pt` Store

Mount your query audio directory read-only. The container defaults to `verify`, so you only need to pass the query file.

```bash
docker run --rm \
  -v "$(pwd)/audio_samples:/data/audio:ro" \
  protonet-sr \
  --input-audio /data/audio/query.wav
```

Use a claimed speaker ID when you want strict verification instead of best-match identification:

```bash
docker run --rm \
  -v "$(pwd)/audio_samples:/data/audio:ro" \
  protonet-sr \
  --input-audio /data/audio/query.wav \
  --speaker-id alice \
  --match-threshold 0.70
```

## Enroll New Speakers Into A Persistent Store

Mount `output/` so the `.pt` store persists on the host, then run the `enroll` subcommand.

```bash
docker run --rm \
  -v "$(pwd)/audio_samples:/data/audio:ro" \
  -v "$(pwd)/output:/app/output" \
  protonet-sr \
  enroll \
  --speaker-id alice \
  --audio-files /data/audio/alice_01.wav /data/audio/alice_02.wav
```

Re-running `enroll` for the same `--speaker-id` updates that speaker in place. Using a new `--speaker-id` appends a new enrollment to the same store.

## Verify Against The Updated Store

After enrolling, verify with the mounted store:

```bash
docker run --rm \
  -v "$(pwd)/audio_samples:/data/audio:ro" \
  -v "$(pwd)/output:/app/output" \
  protonet-sr \
  --input-audio /data/audio/query.wav \
  --speaker-id alice
```

## Override The Store Path

If you want to keep a separate store file:

```bash
docker run --rm \
  -v "$(pwd)/audio_samples:/data/audio:ro" \
  -v "$(pwd)/stores:/stores" \
  protonet-sr \
  enroll \
  --speaker-id bob \
  --audio-files /data/audio/bob.wav \
  --store-path /stores/custom_enrolled.pt
```

The same `--store-path` can be used later with `verify`.

## Requirements Files

- `requirements_runtime.txt`: minimal packages used by the Docker image
- `requirements_enrollment.txt`: broader host-side repo dependencies

The active local Python environment used during validation was not a project `.venv`; it was the system interpreter and contained many unrelated packages. Those extra packages are intentionally not copied into the repo requirements files.

## Exit Codes

- `0`: accepted match or successful enrollment
- `3`: verification ran but the similarity score was below threshold
- `1`: runtime or input error
