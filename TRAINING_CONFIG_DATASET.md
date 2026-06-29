# Tong hop cau hinh huan luyen va tap du lieu

Tai lieu nay chi ghi lai tap du lieu huan luyen va cac tham so cau hinh lien quan truc tiep den qua trinh huan luyen cac mo hinh x-vector va ECAPA-TDNN, co va khong co Prototypical Network.

## 1. Tap du lieu huan luyen

### 1.1. Dataset mac dinh khi chay automation

Khi chay `train_speaker_models.sh`, tat ca cac stage trong script dung chung dataset:

| Tham so | Gia tri |
| --- | --- |
| `NUM_SPEAKERS` | `250` |
| `TRAIN_RATIO` | `0.6` |
| `VAL_RATIO` | `0.2` |
| `TEST_RATIO` | `0.2` |
| `MIN_SAMPLES_PER_SPEAKER` | `20` |
| `MAX_SAMPLES_PER_SPEAKER` | khong gioi han |

Dataset duoc chia theo speaker, khong chia ngau nhien cac file cua cung mot speaker sang nhieu split. Vi vay train, validation va test la cac tap speaker roi nhau.

### 1.2. Dataset mac dinh cua ECAPA-TDNN + ProtoNet

`main.py` dung cho ECAPA-TDNN + ProtoNet. Dataset mac dinh la:

| Tham so | Gia tri |
| --- | --- |
| So speaker | `250` |
| Split | `0.6 / 0.2 / 0.2` |
| Min samples/speaker | `20` |
| Max samples/speaker | `None` |

### 1.3. Dataset cuc bo trong repo

Thu muc `audio_data/` luu tru data thuc te duoc dung de kiem tra kha nang cua mo hinh khi thuc nghiem voi data thuc te

| Thong tin | Gia tri hien tai |
| --- | --- |
| Thu muc | `audio_data/` |
| So speaker | `24` |
| Tong file audio | `429` |

Phan bo file theo speaker:

| Speaker | So file |
| --- | ---: |
| `spk01` | 20 |
| `spk02` | 19 |
| `spk03` | 15 |
| `spk04` | 15 |
| `spk05` | 15 |
| `spk06` | 15 |
| `spk07` | 15 |
| `spk08` | 20 |
| `spk09` | 18 |
| `spk10` | 20 |
| `spk11` | 19 |
| `spk12` | 25 |
| `spk13` | 21 |
| `spk14` | 20 |
| `spk15` | 22 |
| `spk16` | 21 |
| `spk17` | 20 |
| `spk18` | 21 |
| `spk19` | 20 |
| `spk20` | 22 |
| `spk21` | 20 |
| `spk22` | 23 |
| `spk23` | 22 |
| `spk24` | 23 |

## 2. Cau hinh tien xu ly audio chung

| Tham so | Gia tri |
| --- | --- |
| `SR` | `16000` |
| `N_MELS` | `80` |
| `DURATION` | `5.0` giay |
| `N_FFT` | `512` |
| `HOP_LENGTH` | `256` |
| Chuan hoa mel | chuan hoa tung mau bang mean/std |
| Pad/crop audio | dua ve do dai co dinh theo `DURATION` |

## 3. Cau hinh VAD va augmentation

| Tham so | Gia tri |
| --- | --- |
| `VAD_ENABLED` | `1` khi chay `train_speaker_models.sh` |
| `VAD_TOP_DB` | `10.0` |
| `VAD_FRAME_LENGTH` | `2048` |
| `VAD_HOP_LENGTH` | `258` |
| `TRAIN_AUGMENT` | `1` |
| `AUGMENTATION_PROBABILITY` | `0.2` trong automation |
| `AUGMENTATION_RIR_DIR` | `rirs_noises/RIRS_NOISES/real_rirs_isotropic_noises` |
| RIR/noise files hien co | `419` |

Augmentation trong training gom cac nhom bien doi waveform: them nhieu theo SNR, reverberation bang RIR, dropout theo thoi gian, dropout theo tan so, dich thoi gian va thay doi toc do.

## 4. Cau hinh x-vector khong ProtoNet

Entrypoint: `train_xvector_no_protonet.py`

| Nhom | Tham so | Gia tri |
| --- | --- | --- |
| Kien truc | `XVECTOR_TDNN_CHANNELS` | `512` |
| Kien truc | `XVECTOR_STATS_CHANNELS` | `1500` |
| Kien truc | `XVECTOR_EMBEDDING_DIM` | `192` |
| Kien truc | `XVECTOR_DROPOUT` | `0.1` |
| Khoi tao | `XVECTOR_INIT_CHECKPOINT` | `output/xvector_init.pth` |
| Training | `XVECTOR_AAM_EPOCHS` | `50` |
| Training | `XVECTOR_AAM_BATCH_SIZE` | `128` |
| Training | `XVECTOR_AAM_LR` | `1e-4` |
| Training | `XVECTOR_AAM_WEIGHT_DECAY` | `0.01` |
| Training | `XVECTOR_AAM_NUM_WORKERS` | `0` |
| Loss | `AAM_SCALE` | `30.0` |
| Loss | `AAM_MARGIN` | `0.15` |
| Evaluation | `XVECTOR_AAM_MAX_EVAL_PAIRS` | `20000` |

Kieu training: supervised batch training voi AAM-Softmax, khong lay episode ProtoNet.

## 5. Cau hinh x-vector + ProtoNet

Entrypoint: `train_xvector_protonet.py`

| Nhom | Tham so | Gia tri |
| --- | --- | --- |
| Kien truc | `XVECTOR_TDNN_CHANNELS` | `512` |
| Kien truc | `XVECTOR_STATS_CHANNELS` | `1500` |
| Kien truc | `XVECTOR_EMBEDDING_DIM` | `192` |
| Kien truc | `XVECTOR_DROPOUT` | `0.1` |
| Khoi tao | `XVECTOR_INIT_CHECKPOINT` | `output/xvector_init.pth` |
| Episode | `PROTO_N_WAY` | `5` |
| Episode | `PROTO_K_SHOT` | `5` |
| Episode | `PROTO_N_QUERY` | `15` |
| Episode | `PROTO_EPISODES` | `500` |
| Episode | `PROTO_VAL_EPISODES` | `2` |
| Episode | `PROTO_TEST_EPISODES` | `50` |
| Loss | `PROTO_LOSS_MODE` | `hybrid` |
| Loss | `PROTO_SCALE` | `30.0` |
| Loss | `PROTO_MARGIN` | `0.15` |
| Loss | `AAM_SCALE` | `30.0` |
| Loss | `AAM_MARGIN` | `0.15` |
| Loss | `HYBRID_PROTO_WEIGHT` | `0.8` |
| Loss | `HYBRID_AAM_WEIGHT` | `0.2` |
| Optimizer | `PROTO_LR` | `1e-4` |
| Optimizer | `PROTO_WEIGHT_DECAY` | `0.01` |
| Evaluation | `EVAL_SEED` | `36` |

Kieu training: episodic few-shot learning. Moi episode chon `5` speaker, moi speaker co `5` support samples va `15` query samples.

## 6. Cau hinh ECAPA-TDNN khong ProtoNet

Entrypoint: `train_ecapa_tdnn_no_protonet.py`

| Nhom | Tham so | Gia tri |
| --- | --- | --- |
| Kien truc | `ECAPA_CHANNELS` | `512` |
| Kien truc | `ECAPA_EMBEDDING_DIM` | `192` |
| Training | `ECAPA_EPOCHS` | `50` |
| Training | `ECAPA_BATCH_SIZE` | `128` |
| Training | `ECAPA_LR` | `1e-4` |
| Training | `ECAPA_WEIGHT_DECAY` | `0.01` |
| Training | `ECAPA_NUM_WORKERS` | `0` |
| Loss | `AAM_SCALE` | `30.0` |
| Loss | `AAM_MARGIN` | `0.15` |
| Evaluation | `ECAPA_MAX_EVAL_PAIRS` | `20000` |

Kieu training: supervised batch training voi AAM-Softmax, khong lay episode ProtoNet.

## 7. Cau hinh ECAPA-TDNN + ProtoNet

Entrypoint: `main.py`

| Nhom | Tham so | Gia tri |
| --- | --- | --- |
| Backbone | `BACKBONE` | `ecapa` |
| Kien truc | `EMBEDDING_DIM` | `192` |
| Kien truc | `ecapa_channels` | `512` |
| Episode | `N_WAY` | `5` |
| Episode | `K_SHOT` | `5` |
| Episode | `N_QUERY` | `15` |
| Episode | `N_EPISODES` | `500` |
| Episode | `N_VAL_EPISODES` | `2` |
| Episode | `N_TEST_EPISODES` | `50` |
| Loss | `TRAINING_LOSS_MODE` | `hybrid` |
| Loss | `PROTO_SCALE` | `30.0` |
| Loss | `PROTO_MARGIN` | `0.15` |
| Loss | `AAM_SCALE` | `30.0` |
| Loss | `AAM_MARGIN` | `0.15` |
| Loss | `HYBRID_PROTO_WEIGHT` | `0.8` |
| Loss | `HYBRID_AAM_WEIGHT` | `0.2` |
| Optimizer | `PROTO_LR` | `1e-4` |
| Optimizer | `PROTO_WEIGHT_DECAY` | `0.01` |
| Training | `TRAIN_AUGMENT` | `True` |
| Training | `REQUIRE_CUDA` | `True` |
| Evaluation | `EVAL_SEED` | `36` |

Kieu training: episodic few-shot learning voi ECAPA-TDNN backbone. `main.py` khong dung checkpoint khoi tao ban dau cho backbone.

