# Week 01 Report — Hugging Face và inference GPT-2

## Objective

Làm quen `transformers`, `tokenizers`, `datasets`; chạy model có sẵn và hiểu cấu trúc file,
đặc biệt `safetensors`. Tuần này chưa huấn luyện, quantization hay distillation.

## Environment and Model

Python 3.11.18 trong Conda `llm-compression-lab`; PyTorch 2.9.1+cu126;
Transformers 5.23.0; GPU RTX 4060 Laptop, inference `cuda:0` với `torch.float32`.
Chi tiết phiên bản ở [nhật ký](../docs/experiment_log.md).

Model ID `gpt2`, revision yêu cầu `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
Theo [model card](https://huggingface.co/openai-community/gpt2), bản này có khoảng
124 triệu tham số, học dự đoán token tiếp theo trên văn bản tiếng Anh và có giấy phép MIT.
Thông tin model card là mô tả nguồn; không phải số đo tự thực hiện.

## Method

Notebook đọc cấu hình, khởi tạo tokenizer, tạo dataset năm prompt và tokenize bằng `map`.
Model được tải một lần với revision cố định, chuyển lên GPU và đặt `eval()`.
Mỗi dòng dataset được chuyển thành tensor `[1, T]`; `generate` chạy trong
`torch.inference_mode()`, greedy decoding, seed 42, tối đa 50 token mới.
Mỗi kết quả được ghi JSON cùng một bảng CSV của lần chạy.

Luồng: văn bản → token IDs → embedding và các lớp model → logits → chọn token mới
→ nối vào chuỗi → decode. Logits có shape `[B, T, V]`; lấy vị trí cuối để dự đoán
sau prompt. Với “Machine learning is”, quan sát ban đầu có IDs `[37573, 4673, 318]`,
logits `[1, 3, 50257]`, token tiếp theo ID 257 (`' a'`), khớp bước đầu của `generate`.

## Results

Lần chạy người học `run_20261115T104944859289Z` có năm JSON, mỗi prompt sinh 50 token mới.

| STT | Prompt | Token mới | Quan sát | Bằng chứng |
| --- | --- | ---: | --- | --- |
| 1 | `Machine learning is` | 50 | Sinh tiếp liên quan đến học máy nhưng lặp ý “understand the world around us”; kết thúc giữa câu. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_01.json) |
| 2 | `Once upon a time,` | 50 | Mở đầu theo kiểu truyện nhưng lặp nhiều lần “the world was a place of great danger”. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_02.json) |
| 3 | `The purpose of a university is` | 50 | Nội dung liên quan đến trường đại học; lặp câu về môi trường học tập an toàn và dừng giữa câu. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_03.json) |
| 4 | `The capital of Vietnam is` | 50 | Lặp vòng cụm “the capital of the Republic of Vietnam”; không đưa ra tên một thành phố. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_04.json) |
| 5 | `The best university in Vietnam is` | 50 | Nêu “University of the South” kèm khẳng định không có nguồn, rồi lặp ý; không dùng làm kết luận xếp hạng trường. | [JSON](../outputs/week01/run_20261115T104944859289Z/inference_05.json) |

Model sinh tiếp theo ngữ cảnh nhưng thường lặp ý và dừng giữa câu ở giới hạn token.
Các khẳng định trong output không được coi là thông tin đã kiểm chứng.
Đây là khảo sát định tính với năm prompt, chưa phải benchmark chất lượng hay hiệu năng.

## Model Files

Snapshot chứa config, generation config, các file tokenizer và `model.safetensors`.
File trọng số có 548105171 byte (522.71 MiB), chứa 160 tensor được lưu, không phải
160 tham số. Một tensor có nhiều phần tử; một số tensor là buffer.
Config mô tả kiến trúc; tokenizer quy định ánh xạ văn bản–ID; safetensors lưu tensor.
Dung lượng file không phải VRAM lúc inference. Tổng dung lượng snapshot chưa đo chính xác.
Xem [ghi chú model](../docs/model_files.md) để biết đường dẫn và bảng file cục bộ.

## Lessons Learned and Next Steps

Các lỗi đã gặp gồm gián đoạn tải do DNS, chọn nhầm kernel, biến cấu hình chưa được
nạp lại và thuộc tính revision nội bộ không có giá trị. Ghi riêng `requested_revision`
giúp mô tả chính xác cách tải, thay vì suy diễn metadata không có.

Notebook là sản phẩm thực hành có thể chạy bằng Restart Kernel → Run All khi môi trường
và model/cache sẵn sàng. Khảo sát Hub cần mạng và có thể được bỏ qua nếu mất kết nối;
việc bỏ qua được in rõ, không được coi là khảo sát Hub thành công.
Các file dưới `outputs/` đang bị Git ignore; cần kèm chúng khi nộp sản phẩm nếu yêu cầu số liệu thô.

Bước tiếp theo của lộ trình là học/cài Transformer nhỏ ở tuần 2, tập trung embedding,
causal self-attention, MLP, residual và normalization.

## Verification status

Đã đối chiếu năm JSON và các liên kết tài liệu. Notebook đã lưu output của các cell
chính đến khảo sát Hub; cell đọc trọng số cục bộ chưa lưu output trong bản kiểm tra.
Lần kiểm tra tự động qua kernel Jupyter của trợ lý bị dừng vì chờ quá lâu,
không được tính là Run All thành công. Cần người học Restart Kernel → Run All
trong VS Code rồi Ctrl+S để chốt bước tái lập trước khi đánh dấu tuần 1 hoàn tất.
