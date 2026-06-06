# Tài Liệu Nền Cho Khoá Luận Tốt Nghiệp

## 1. Mở đầu

Dự án hiện tại tập trung vào bài toán nhận dạng và kiểm chứng người nói (speaker recognition / speaker verification) từ tín hiệu tiếng nói ngắn. Về mặt ứng dụng, hệ thống hướng tới khả năng học biểu diễn đặc trưng giọng nói, đăng ký người dùng mới từ một số lượng mẫu nhỏ, so khớp truy vấn với tập người dùng đã đăng ký, và triển khai suy luận trên môi trường biên thông qua ONNX hoặc RKNN.

Xét dưới góc độ học thuật, codebase này đã có đủ các thành phần cốt lõi của một hệ thống nghiên cứu hoàn chỉnh: mô hình học sâu, pipeline huấn luyện, pipeline fine-tune theo miền dữ liệu nội bộ, quy trình tiền xử lý âm thanh, công cụ đánh giá, và hướng triển khai trên thiết bị tài nguyên hạn chế. Vì vậy, repo có thể được sử dụng như một baseline khả thi để phát triển thành khoá luận tốt nghiệp, thay vì xây dựng đề tài từ đầu.

Mục tiêu của tài liệu này là tổng hợp hiện trạng kỹ thuật của dự án, chỉ ra các kết quả đã có, xác định những khoảng trống còn tồn tại, và đề xuất các hướng mở rộng phù hợp để chuyển hoá hệ thống hiện tại thành một đề tài khoá luận có mục tiêu, phạm vi và giá trị đánh giá rõ ràng.

## 2. Tổng quan hệ thống hiện tại

### 2.1. Kiến trúc mô hình

Thành phần học sâu chính của dự án nằm trong [`model/ECAPATDNN.py`](/home/d/Projects/Protonet_SR/model/ECAPATDNN.py). Backbone sử dụng kiến trúc ECAPA-TDNN với các thành phần tiêu biểu gồm:

- khối TDNN 1 chiều trên trục thời gian;
- các khối ECAPARes2SE để học đặc trưng đa tỉ lệ và tăng cường chú ý theo kênh;
- cơ chế multi-layer feature aggregation;
- attentive statistics pooling để gom đặc trưng theo thời gian;
- lớp chiếu cuối cùng để sinh embedding đã được chuẩn hoá L2.

Phần học metric được tổ chức theo hướng Prototypical Network trong [`PrototypicalNetwork/train.py`](/home/d/Projects/Protonet_SR/PrototypicalNetwork/train.py). Trong mỗi episode, hệ thống chọn `n_way` speaker, lấy `k_shot` mẫu hỗ trợ và `n_query` mẫu truy vấn, sau đó:

- sinh embedding cho từng mẫu;
- tính prototype cho từng lớp từ tập support;
- tính độ tương đồng cosine giữa truy vấn và prototype;
- tối ưu hàm mất mát prototypical có bổ sung các ràng buộc pull/push giữa positive và hardest negative.

Thiết kế này phù hợp với bối cảnh few-shot speaker recognition, trong đó hệ thống cần thích nghi tốt với số lượng mẫu đăng ký thấp và có thể mở rộng theo hướng open-set verification.

### 2.2. Các workflow chính trong repo

Các script hiện có phản ánh rõ các luồng xử lý chính của hệ thống:

- [`main.py`](/home/d/Projects/Protonet_SR/main.py): huấn luyện mô hình ECAPA-TDNN với Prototypical Network trên tập LibriSpeech `train-clean-100`. Checkpoint mặc định được ghi vào `output/ECAPATDNN_protonet_model.pth`, đồng thời sinh đồ thị quá trình học và DET curve.
- [`finetune.py`](/home/d/Projects/Protonet_SR/finetune.py): fine-tune từ checkpoint có sẵn trên dữ liệu cục bộ trong `audio_data/` hoặc một thư mục dữ liệu khác. Luồng này cho phép thích nghi mô hình theo miền tiếng nói mục tiêu.
- [`enroll_lite.py`](/home/d/Projects/Protonet_SR/enroll_lite.py): đăng ký speaker bằng cách gom embedding từ nhiều file âm thanh, trung bình hoá và chuẩn hoá để lưu vào kho embedding. Script hỗ trợ hai định dạng đầu ra: `.pt` làm store chính và `.npy` làm định dạng legacy cho môi trường C/C++.
- [`speaker_cli.py`](/home/d/Projects/Protonet_SR/speaker_cli.py): CLI phục vụ hai thao tác triển khai là `enroll` và `verify`, phù hợp với môi trường container hoặc runtime đơn giản.
- [`evaluate_random_speaker_split.py`](/home/d/Projects/Protonet_SR/evaluate_random_speaker_split.py): đánh giá kịch bản chọn ngẫu nhiên một nhóm speaker để enroll, dùng các clip còn lại của nhóm này làm genuine trials và dùng toàn bộ clip của nhóm không enroll làm impostor trials.
- [`evaluate_progressive_enrollment.py`](/home/d/Projects/Protonet_SR/evaluate_progressive_enrollment.py): đánh giá theo hướng tăng dần số lượng clip đăng ký trên mỗi speaker để quan sát độ ổn định khi enrollment ít hoặc nhiều mẫu.
- [`export_to_rknn.py`](/home/d/Projects/Protonet_SR/export_to_rknn.py): xuất checkpoint PyTorch sang ONNX và RKNN; script này cũng xử lý chuẩn bị dữ liệu calibration khi lượng tử hoá INT8 cho nền tảng Rockchip.
- [`test_onnx.py`](/home/d/Projects/Protonet_SR/test_onnx.py): kiểm thử suy luận ONNX bằng cách so sánh truy vấn với một file `.wav` tham chiếu hoặc một `.pt` embedding store.

Xét ở mức hệ thống, repo hiện đã bao phủ gần trọn vòng đời của một hệ thống speaker recognition:

1. huấn luyện baseline trên tập dữ liệu công khai;
2. fine-tune trên dữ liệu đích;
3. enroll người dùng;
4. verify hoặc identify bằng cosine similarity;
5. đánh giá theo nhiều protocol;
6. xuất mô hình để triển khai trên edge device.

### 2.3. Đầu vào và đầu ra quan trọng của từng workflow

Các giao diện công khai trong repo hiện tương đối rõ ràng:

| Thành phần | Đầu vào chính | Đầu ra chính |
| --- | --- | --- |
| `main.py` | thư mục LibriSpeech, tham số episode learning | checkpoint `.pth`, đồ thị huấn luyện, DET curve |
| `finetune.py` | checkpoint sẵn có, thư mục dữ liệu nội bộ | checkpoint fine-tuned, đồ thị huấn luyện, DET curve |
| `enroll_lite.py` | checkpoint và danh sách file audio của một speaker | store `enrolled_speakers.pt` hoặc ma trận `.npy` |
| `speaker_cli.py verify` | audio truy vấn, store speaker đã enroll | kết quả xác nhận hoặc top-k so khớp |
| `evaluate_random_speaker_split.py` | thư mục dataset theo speaker, checkpoint, ngưỡng quyết định | file JSON tổng hợp và JSON chi tiết |
| `evaluate_progressive_enrollment.py` | dataset theo speaker, checkpoint | CSV chi tiết và JSON summary |
| `export_to_rknn.py` | checkpoint hoặc ONNX, tham số export, tập calibration | ONNX, RKNN, bộ calibration `.npy` nếu cần |
| `test_onnx.py` | ONNX model, audio truy vấn, audio/store tham chiếu | điểm tương đồng và quyết định match |

Điểm đáng chú ý là hệ thống đang dùng cùng một trục biểu diễn embedding để phục vụ cả ba mục tiêu: huấn luyện few-shot, enrollment/verification trên host, và chuẩn bị mô hình triển khai trên edge.

## 3. Tiền xử lý dữ liệu

### 3.1. Pipeline tiền xử lý âm thanh hiện tại

Phần tiền xử lý lõi được cài đặt chủ yếu trong [`utils/data_preprocessing.py`](/home/d/Projects/Protonet_SR/utils/data_preprocessing.py). Xét theo logic hiện có trong code, pipeline tiền xử lý chung bao gồm các bước sau:

1. đọc tín hiệu âm thanh ở dạng mono;
2. đưa tín hiệu về sample rate mục tiêu, mặc định là `16 kHz`;
3. chuẩn hoá độ dài bằng cách cắt giữa hoặc đệm `0` tới thời lượng mục tiêu;
4. biến đổi sang mel-spectrogram với số dải mel mặc định là `80`;
5. chuyển đổi sang miền dB;
6. chuẩn hoá đặc trưng theo từng mẫu bằng công thức `(mel - mean) / (std + 1e-8)`.

Trong các luồng huấn luyện và fine-tune, đối tượng `SpeakerDataset` còn hỗ trợ chuyển đổi trực tiếp từ batch waveform sang batch mel trên thiết bị tính toán bằng `torchaudio.transforms.MelSpectrogram` và `AmplitudeToDB`, sau đó chuẩn hoá theo từng mẫu trong batch. Điều này giúp thống nhất tiền xử lý giữa training và evaluation episode.

### 3.2. Chunking cho enrollment, inference và calibration

Repo không chỉ xử lý audio theo một cửa sổ cố định duy nhất. Hàm `audio_chunking()` trong [`utils/data_preprocessing.py`](/home/d/Projects/Protonet_SR/utils/data_preprocessing.py) cho phép:

- chia waveform dài thành nhiều đoạn chồng lấn;
- padding đoạn cuối nếu thiếu mẫu;
- tùy chọn trả về waveform chunks hoặc trả thẳng về mel tensors đã chuẩn hoá.

Luồng này được dùng trực tiếp trong:

- [`enroll_lite.py`](/home/d/Projects/Protonet_SR/enroll_lite.py): tách một file dài thành nhiều chunk trước khi lấy embedding, sau đó trung bình hoá các embedding chunk để tạo đại diện speaker ổn định hơn;
- [`export_to_rknn.py`](/home/d/Projects/Protonet_SR/export_to_rknn.py): chuyển audio calibration thành nhiều tensor `.npy` có kích thước đúng với input ONNX/RKNN;
- một phần của CLI verification trong [`speaker_cli.py`](/home/d/Projects/Protonet_SR/speaker_cli.py), nơi audio truy vấn có thể được quy về nhiều khúc ngắn trước khi gom embedding.

Như vậy, tiền xử lý trong dự án không chỉ là bước “chuẩn bị đầu vào”, mà còn là một thành phần ảnh hưởng trực tiếp tới chất lượng embedding, độ ổn định khi enrollment, và chất lượng calibration trong bước lượng tử hoá mô hình.

### 3.3. Augmentation trong huấn luyện

Chiến lược tăng cường dữ liệu được định nghĩa trong [`utils/data_augmentation.py`](/home/d/Projects/Protonet_SR/utils/data_augmentation.py). Các phép biến đổi hiện có gồm:

- `waveform_dropout`: chèn các khoảng im lặng ngẫu nhiên trong miền thời gian;
- `frequency_dropout`: loại bỏ ngẫu nhiên một số dải tần;
- `reverberation`: chập tín hiệu với RIR lấy từ bộ `RIRS_NOISES`;
- `gaussian_noise`: cộng nhiễu Gaussian theo khoảng SNR chỉ định;
- `noise_reverberation`: kết hợp reverberation và noise;
- `shifting`: dịch thời gian ngẫu nhiên;
- `speed_change`: thay đổi tốc độ phát.

Trong `main.py` và `finetune.py`, augmentation được kích hoạt ở mức xác suất, mặc định với `augmentation_probability = 0.3`, và có thể sử dụng thêm thư mục RIR/noise trong `rirs_noises/RIRS_NOISES/real_rirs_isotropic_noises`. Đây là một điểm quan trọng nếu phát triển thành khoá luận, vì hiệu quả của augmentation theo miền dữ liệu thực tế hoàn toàn có thể trở thành một câu hỏi nghiên cứu độc lập.

### 3.4. Khác biệt giữa các luồng tiền xử lý

Mặc dù các bước cốt lõi khá nhất quán, từng workflow lại có mục tiêu khác nhau:

- Huấn luyện từ đầu (`main.py`): ưu tiên chia dữ liệu thành các split speaker rời nhau, chuẩn hoá waveform/mel về kích thước cố định, và áp dụng augmentation để học embedding tổng quát.
- Fine-tune (`finetune.py`): kế thừa logic training nhưng nhắm tới tập dữ liệu nhỏ hơn, gần miền triển khai hơn, vì vậy nhạy hơn với lựa chọn speaker, số mẫu tối thiểu, và cấu hình augmentation.
- Enrollment (`enroll_lite.py`): dùng chunking để gom nhiều đoạn của cùng một speaker thành một embedding đại diện, giảm độ lệch do từng câu nói riêng lẻ.
- Verification (`speaker_cli.py`, `test_onnx.py`): chú trọng tính nhất quán giữa tiền xử lý của truy vấn và tiền xử lý của reference store hoặc reference audio.
- Export/calibration (`export_to_rknn.py`): không tối ưu cho phân loại trực tiếp mà tối ưu cho việc tạo các tensor đầu vào hợp lệ và mang tính đại diện khi build RKNN lượng tử hoá.

Vì vậy, nếu tiếp tục phát triển thành khoá luận, phần tiền xử lý không nên chỉ được xem là chi tiết cài đặt phụ trợ, mà cần được xem như một biến số thực nghiệm có ảnh hưởng đến kết quả cuối cùng.

## 4. Dữ liệu và thực nghiệm hiện có

### 4.1. Nguồn dữ liệu

Repo hiện phản ánh hai nguồn dữ liệu chính:

- Dữ liệu công khai LibriSpeech `train-clean-100`: dùng làm nguồn huấn luyện baseline cho `main.py`. Theo logic trong `utils/paths.py` và `QUICKSTART.md`, repo hỗ trợ dò tìm thư mục LibriSpeech theo một số vị trí mặc định hoặc thông qua biến môi trường `LIBRISPEECH_ROOT`.
- Dữ liệu nội bộ `audio_data/`: dùng cho fine-tune và đánh giá các tình huống triển khai thực tế hơn.

Thư mục `audio_data/` hiện có:

- `13` speaker;
- `237` file `.wav`;
- số lượng file trên mỗi speaker dao động từ `15` đến `25`.

Phân bố hiện tại như sau:

| Speaker | Số file |
| --- | ---: |
| spk01 | 20 |
| spk02 | 19 |
| spk03 | 15 |
| spk04 | 15 |
| spk05 | 15 |
| spk06 | 15 |
| spk07 | 15 |
| spk08 | 19 |
| spk09 | 19 |
| spk10 | 20 |
| spk11 | 19 |
| spk12 | 25 |
| spk13 | 21 |

Phân bố này cho thấy dữ liệu nội bộ chưa lớn, chưa cân bằng tuyệt đối, và một số speaker chỉ vừa đủ ngưỡng để tham gia các protocol few-shot có `k_shot + n_query` tương đối cao.

### 4.2. Artifact đầu ra hiện có

Thư mục `output/` hiện chứa các artifact quan trọng sau:

- checkpoint huấn luyện và fine-tune: `ECAPATDNN_protonet_model.pth`, `ECAPATDNN_protonet_finetuned.pth`;
- đồ thị huấn luyện và DET curve: `ECAPATDNN_protonet_curves.png`, `ECAPATDNN_protonet_det_curve.png`, `ECAPATDNN_protonet_finetuned_curves.png`;
- nhiều file JSON đánh giá trên các protocol khác nhau, bao gồm:
  - `audio_data_eval_5random_threshold_070.json`;
  - `audio_data_eval_5random_threshold_sweep.json`;
  - `audio_data_eval_spk03_to_spk07_threshold_sweep.json`;
  - `audio_data_eval_spk03_to_spk07_detailed.json`;
  - `audio_data_eval_spk03_to_spk07_softmax_threshold_sweep.json`;
  - `checkpoint_eval_latest_threshold_050.json`;
  - `checkpoint_eval_latest_threshold_070.json`;
  - `checkpoint_eval_latest_threshold_080.json`.

Tập artifact này cho thấy dự án không dừng ở mức xây mô hình, mà đã đi tới bước đo lường hành vi của hệ thống khi thay đổi speaker enroll, ngưỡng quyết định, hoặc chiến lược hậu xử lý xác suất.

## 5. Kết quả hiện tại cần trích dẫn

### 5.1. Kịch bản chọn ngẫu nhiên 5 speaker để enroll

File `output/audio_data_eval_5random_threshold_070.json` phản ánh một kịch bản đánh giá trong đó:

- ngưỡng quyết định là `0.70`;
- tập speaker được enroll gồm `spk02`, `spk06`, `spk07`, `spk08`, `spk11`;
- tập speaker không enroll gồm `spk01`, `spk03`, `spk04`, `spk05`, `spk09`, `spk10`, `spk12`, `spk13`.

Các chỉ số chính như sau:

- độ chính xác genuine tổng thể: `59/62`, tương đương `95.16%`;
- genuine accept rate: `54.84%`;
- false accept rate trên impostor trials: `14/150`, tương đương `9.33%`.

Ở mức speaker, có sự chênh lệch rõ rệt:

- `spk11` đạt accuracy `100%` và accept rate `100%`;
- `spk06` đạt accuracy `100%` nhưng accept rate chỉ `10%`;
- `spk08` đạt accuracy `85.71%`, thấp hơn các speaker còn lại trong nhóm enroll.

Điều này cho thấy việc dự đoán đúng speaker chưa đồng nghĩa với việc hệ thống sẵn sàng “chấp nhận” truy vấn ở ngưỡng đang đặt. Nói cách khác, chất lượng embedding và hiệu ứng calibration ngưỡng vẫn còn là vấn đề đáng nghiên cứu.

Một chi tiết quan trọng khác là false accept tập trung mạnh vào một số cặp speaker:

- phần lớn false accept rơi vào target `spk07` (`11` trường hợp) và `spk06` (`3` trường hợp);
- phía impostor, `spk03` gây ra `9` false accept và `spk05` gây ra `4` false accept.

Mẫu hình này gợi ý rằng sự tương đồng liên-speaker trong miền dữ liệu nội bộ có thể đang là yếu tố chi phối sai số nhiều hơn bản thân độ chính xác closed-set.

### 5.2. Kịch bản sweep ngưỡng với nhóm `spk03` đến `spk07`

File `output/audio_data_eval_spk03_to_spk07_threshold_sweep.json` cho thấy độ nhạy rất rõ theo ngưỡng quyết định cosine:

| Threshold | Genuine accept rate | False accept rate |
| --- | ---: | ---: |
| 0.60 | 88.00% | 4.94% |
| 0.70 | 60.00% | 1.85% |
| 0.75 | 38.00% | 0.62% |
| 0.80 | 14.00% | 0.00% |

Kết quả này phản ánh một đánh đổi điển hình của bài toán open-set verification:

- khi giảm ngưỡng, hệ thống dễ chấp nhận genuine hơn nhưng tăng nguy cơ false accept;
- khi tăng ngưỡng, hệ thống gần như loại bỏ được false accept nhưng làm giảm mạnh khả năng chấp nhận đúng.

Đối với một khoá luận, đây là một điểm rất có giá trị, vì nó mở ra nhu cầu xây dựng tiêu chí chọn ngưỡng dựa trên mục tiêu sử dụng thực tế thay vì đặt thủ công một giá trị cố định.

### 5.3. Kết quả trên checkpoint “latest” với tập nhỏ

Hai file `output/checkpoint_eval_latest_threshold_070.json` và `output/checkpoint_eval_latest_threshold_080.json` cho thấy cùng một checkpoint nhưng thay đổi ngưỡng đã làm thay đổi mạnh hành vi chấp nhận:

Ở ngưỡng `0.70`:

- genuine accuracy: `30/30`, tương đương `100%`;
- genuine accept rate: `50%`;
- false accept rate: `8/30`, tương đương `26.67%`.

Ở ngưỡng `0.80`:

- genuine accuracy: vẫn `30/30`, tương đương `100%`;
- genuine accept rate: giảm còn `10%`;
- false accept rate: giảm còn `3/30`, tương đương `10%`.

Diễn giải học thuật của kết quả này là: mô hình có thể duy trì năng lực phân biệt speaker trong bài toán closed-set rất tốt, nhưng độ tin cậy của score tuyệt đối cho open-set decision chưa ổn định. Do đó, nếu đề tài khoá luận tập trung vào verification, phần hậu xử lý ngưỡng và chuẩn hoá score nên được xem là một trục nghiên cứu trọng tâm.

## 6. Khoảng trống và hạn chế hiện tại

Từ hiện trạng repo và các kết quả đã có, có thể rút ra một số hạn chế cốt lõi.

### 6.1. Quy mô và cấu trúc dữ liệu còn hạn chế

Dữ liệu nội bộ `audio_data/` mới dừng ở `13` speaker với `237` file, trong đó nhiều speaker chỉ có `15` mẫu. Quy mô này đủ để làm baseline và minh hoạ ý tưởng, nhưng còn nhỏ nếu mục tiêu là kết luận chắc chắn về khả năng tổng quát hoá hoặc so sánh nhiều cấu hình một cách có ý nghĩa thống kê.

### 6.2. Hệ thống nhạy với ngưỡng quyết định

Các file sweep ngưỡng cho thấy kết quả verification thay đổi mạnh khi chỉnh threshold. Đây là dấu hiệu cho thấy:

- score cosine chưa được hiệu chỉnh tốt cho open-set decision;
- một giá trị threshold cố định có thể không phù hợp cho mọi tập speaker hoặc mọi điều kiện thu âm;
- độ chính xác phân lớp chưa phản ánh đầy đủ chất lượng hệ thống verification.

### 6.3. Tiền xử lý và enrollment có ảnh hưởng lớn nhưng chưa được chuẩn hoá thành protocol nghiên cứu

Repo hiện đã có nhiều lựa chọn tiền xử lý như duration mục tiêu, chunking, overlap, augmentation, RIR và noise. Tuy nhiên, các biến này mới chủ yếu tồn tại ở mức tham số kỹ thuật, chưa được tổ chức thành một protocol thực nghiệm chuẩn để trả lời các câu hỏi kiểu:

- chunking dài hay ngắn thì phù hợp hơn cho dữ liệu nội bộ;
- augmentation nào có ích, augmentation nào gây nhiễu;
- số lượng clip enroll tối thiểu là bao nhiêu để hệ thống đủ ổn định;
- sự khác biệt giữa mel pipeline khi train và khi export có ảnh hưởng gì tới deployment hay không.

### 6.4. Phần edge deployment đã có nền nhưng chưa được lượng hoá đầy đủ

Luồng ONNX và RKNN cho thấy repo đã hướng tới triển khai thực tế, đặc biệt trên nền tảng Rockchip hoặc Raspberry Pi. Tuy nhiên, hiện vẫn chưa có một khối đánh giá học thuật hoàn chỉnh cho triển khai, ví dụ:

- sai lệch giữa PyTorch và ONNX/RKNN;
- ảnh hưởng của quantization tới cosine score;
- độ trễ, bộ nhớ, thông lượng trên thiết bị mục tiêu;
- đánh đổi giữa độ chính xác và khả năng chạy thời gian thực.

## 7. Định hướng phát triển thành khoá luận

Từ baseline hiện có, có thể phát triển thành khoá luận theo một số hướng cụ thể sau.

### 7.1. Tối ưu ngưỡng cho bài toán open-set speaker verification

**Mục tiêu.** Xây dựng phương pháp chọn ngưỡng quyết định phù hợp hơn cho bài toán verification thay vì cố định thủ công.

**Thay đổi cần làm.**

- mở rộng script đánh giá để sinh thêm các chỉ số như FAR, FRR, EER, DET;
- tách tập validation chuyên cho threshold tuning;
- so sánh threshold toàn cục với threshold thích nghi theo điều kiện hoặc theo speaker group.

**Chỉ số đánh giá.**

- false accept rate;
- false reject rate;
- equal error rate;
- genuine accept rate tại các mốc FAR cố định.

**Giá trị.** Hướng này bám sát kết quả đang có trong `output/` và phù hợp nếu mục tiêu khoá luận là nâng chất lượng verification ở góc nhìn vận hành thực tế.

### 7.2. Đánh giá tác động của tiền xử lý và augmentation theo miền dữ liệu nội bộ

**Mục tiêu.** Xác định cấu hình tiền xử lý và augmentation phù hợp nhất cho tập `audio_data/` hoặc các tập ghi âm thực tế cùng miền.

**Thay đổi cần làm.**

- thiết kế ma trận thí nghiệm cho duration, chunk size, overlap và số chunk dùng để gom embedding;
- bật hoặc tắt riêng từng augmentation như noise, RIR, speed perturbation, shifting, dropout;
- so sánh hiệu quả trước và sau fine-tune theo cùng một protocol đánh giá.

**Chỉ số đánh giá.**

- genuine accuracy;
- genuine accept rate;
- false accept rate;
- độ ổn định embedding giữa nhiều clip của cùng speaker.

**Giá trị.** Đây là hướng có tính kỹ thuật rõ ràng, tận dụng trực tiếp logic đã có trong `utils/data_preprocessing.py` và `utils/data_augmentation.py`, đồng thời có khả năng tạo ra đóng góp thực nghiệm có ý nghĩa.

### 7.3. Fine-tune theo domain nội bộ và chuẩn hoá protocol đánh giá

**Mục tiêu.** Xây dựng quy trình fine-tune và đánh giá có thể lặp lại, nhằm chứng minh mức cải thiện của mô hình khi chuyển từ dữ liệu công khai sang dữ liệu đích.

**Thay đổi cần làm.**

- định nghĩa rõ split train/validation/test trên dữ liệu nội bộ;
- lặp lại thí nghiệm với nhiều seed hoặc nhiều speaker split;
- so sánh checkpoint gốc với checkpoint fine-tuned trên cùng protocol.

**Chỉ số đánh giá.**

- accuracy closed-set;
- FAR/FRR hoặc EER cho verification;
- độ biến thiên kết quả theo seed và theo lựa chọn speaker enroll.

**Giá trị.** Hướng này giúp biến các script đánh giá rời rạc hiện tại thành một protocol nghiên cứu nghiêm ngặt hơn, phù hợp với yêu cầu của khoá luận.

### 7.4. Benchmark triển khai edge với ONNX và RKNN

**Mục tiêu.** Đánh giá khả năng triển khai mô hình trên thiết bị biên mà vẫn giữ được chất lượng nhận dạng chấp nhận được.

**Thay đổi cần làm.**

- đối chiếu embedding hoặc cosine score giữa PyTorch, ONNX và RKNN;
- đo độ trễ suy luận, bộ nhớ sử dụng và kích thước mô hình;
- đánh giá ảnh hưởng của lượng tử hoá INT8 tới verification score.

**Chỉ số đánh giá.**

- độ lệch cosine giữa các backend;
- inference latency;
- model size;
- FAR/FRR trước và sau lượng tử hoá.

**Giá trị.** Hướng này gắn với tính ứng dụng của dự án và tạo ra cầu nối tốt giữa nghiên cứu thuật toán và triển khai thực tế.

## 8. Kết luận

Repo hiện tại đã vượt qua mức của một bản demo đơn giản. Hệ thống đã có backbone học sâu rõ ràng, pipeline huấn luyện và fine-tune, cơ chế enrollment/verification, các script đánh giá đa dạng, và cả lộ trình xuất mô hình cho triển khai edge. Quan trọng hơn, các artifact trong `output/` đã cho thấy những vấn đề thực sự đáng nghiên cứu, đặc biệt là sự đánh đổi giữa genuine accept rate và false accept rate, ảnh hưởng của tiền xử lý, và tính nhạy với cấu hình enrollment.

Vì vậy, dự án hoàn toàn có thể được sử dụng làm baseline cho khoá luận tốt nghiệp. Bước tiếp theo hợp lý là chọn một trục nghiên cứu chính, ưu tiên một trong bốn hướng sau: tối ưu ngưỡng verification, phân tích tiền xử lý và augmentation, fine-tune theo domain nội bộ với protocol chuẩn, hoặc benchmark triển khai edge. Khi đã khoanh rõ một trục, phần còn lại của repo có thể đóng vai trò nền tảng thực thi và đối chứng cho toàn bộ khoá luận.
