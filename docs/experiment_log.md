# Nhật ký thực nghiệm — Tuần 1

## Môi trường

- Python: 3.11.18; môi trường Conda `llm-compression-lab`.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU; PyTorch nhận CUDA và phép tính tensor GPU đã thành công.
- PyTorch: 2.9.1+cu126; Transformers: 5.23.0; Datasets: 5.2.0; Tokenizers: 0.23.3; Safetensors: 0.8.0; Hugging Face Hub: 2.4.0.
- Thiết bị thực tế trong JSON: `cuda:0`; dtype: `torch.float32`.
- [Danh sách gói](../outputs/week01/environment.txt) là ảnh chụp môi trường lúc thiết lập, không phải lockfile đa nền tảng.

## Lần chạy ban đầu

Ba prompt được chạy thủ công, chưa cố định revision. Kết quả giữ nguyên trong
[archive/initial_manual_runs](../outputs/week01/archive/initial_manual_runs/).
Không bổ sung hồi tố revision cho những JSON này.

## Lần chạy năm prompt — 15/11/2026, 17:49:44 giờ Việt Nam

- Run ID: `run_20261115T104944859289Z` (timestamp trong tên dùng UTC).
- Mục tiêu: kiểm tra dataset đã nối vào inference và lưu từng kết quả tự động.
- Model ID: `gpt2`.
- Revision yêu cầu: `607a30d783dfa663caf39e06633721c8d4cfcd7e`, truyền cho cả tokenizer và model.
- `model_revision` vẫn là `null` do thuộc tính nội bộ không cung cấp giá trị; `requested_revision` ghi đúng đối số tải.
- Tham số: seed 42, `max_new_tokens=50`, `do_sample=False`; inference từng dòng, batch size 1.
- Model tải một lần, dataset có năm dòng; `map` bổ sung `input_ids` và `attention_mask`.
- [Bảng tổng hợp](../outputs/week01/run_20261115T104944859289Z/summary.csv).

| STT | Prompt | Token mới | Quan sát | Bằng chứng |
| --- | --- | ---: | --- | --- |
| 1 | `Machine learning is` | 50 | Sinh tiếp liên quan đến học máy nhưng lặp ý “understand the world around us”; kết thúc giữa câu. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_01.json) |
| 2 | `Once upon a time,` | 50 | Mở đầu theo kiểu truyện nhưng lặp nhiều lần “the world was a place of great danger”. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_02.json) |
| 3 | `The purpose of a university is` | 50 | Nội dung liên quan đến trường đại học; lặp câu về môi trường học tập an toàn và dừng giữa câu. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_03.json) |
| 4 | `The capital of Vietnam is` | 50 | Lặp vòng cụm “the capital of the Republic of Vietnam”; không đưa ra tên một thành phố. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_04.json) |
| 5 | `The best university in Vietnam is` | 50 | Nêu “University of the South” kèm khẳng định không có nguồn, rồi lặp ý; không dùng làm kết luận xếp hạng trường. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_05.json) |

Các nhận xét chỉ mô tả output thực tế, không xác nhận các khẳng định do model sinh ra.
Năm prompt chưa đủ để đánh giá chất lượng tổng quát, perplexity hay tốc độ.

## Lỗi đã xử lý

- Tải PyTorch bị gián đoạn do mạng/DNS; chạy lại cài đặt sau khi kết nối phục hồi.
- Kernel có nhãn gây nhầm nhưng trỏ Python base; chọn lại môi trường và kiểm tra `sys.executable`.
- `KeyError: revision`: đã sửa JSON nhưng biến `config` còn cũ; chạy lại cell đọc cấu hình.
- `_commit_hash` trả `None`: giữ thông tin quan sát được, ghi riêng revision truyền vào API.

## Tổ chức kết quả

- `archive/initial_manual_runs/`: ba JSON ban đầu.
- `archive/validation_runs/`: lần kiểm tra của trợ lý, không gộp với lần chạy người học.
- `archive/backups/`: notebook trước khi nối dataset.
- Không xóa hoặc sửa output để làm kết quả đẹp hơn.

## Kiến thức đã thực hành

`datasets` quản lý các dòng dữ liệu; `tokenizers` xử lý mã hóa token; `transformers`
cung cấp API tokenizer/model và sinh văn bản; PyTorch tính toán tensor.
Token ID là chỉ số, embedding là vector; logits là điểm số chưa chuẩn hóa thành xác suất.
`attention_mask` biểu thị token thật/padding; causal mask ngăn nhìn tương lai.
`eval()` đổi hành vi lớp như dropout, còn `inference_mode()` tắt theo dõi autograd.

## Kiểm chứng cuối

Các tài liệu đã được đối chiếu với năm JSON đã lưu. Bản notebook đã có output
cho các cell chính nhưng chưa lưu output cell khảo sát trọng số cục bộ.
Lần kiểm tra kernel Jupyter tự động của trợ lý bị dừng do chờ quá lâu, không kết luận
lỗi model và không tính là chạy thành công. Bước còn lại: Restart Kernel → Run All
trong VS Code, kiểm tra không có cell lỗi và Ctrl+S.
