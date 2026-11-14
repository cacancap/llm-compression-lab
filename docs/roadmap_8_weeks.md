# Lộ trình học tập Project I trong 8 tuần

## Nguồn và cách sử dụng

Kế hoạch gốc: **ProposalPlan_ProjectI_202416345.pdf**, Nguyễn Xuân Sơn,
MSSV 202416345, học phần IT3150 Project I, giảng viên hướng dẫn ThS. Đỗ Tuấn Anh.
PDF gồm 2 trang; mục 2 quy định nội dung từng tuần, mục 3 nêu kết quả dự kiến.
File gốc trên máy: `D:/XStudying_At_Hust/FifthSemester/Courses/IT3150_ProjectI/ProposalPlan_ProjectI_202416345.pdf`.
Đường dẫn này là nguồn tham chiếu cục bộ, không nằm trong repository và có thể không tồn tại trên máy khác.

Các mục **Theo PDF** dưới đây tóm tắt yêu cầu gốc. Các mục **Đề xuất thực hành**
là cách triển khai để học và kiểm chứng, không phải yêu cầu bổ sung của giảng viên.
Tuần 2–8 mới là định hướng: model, dataset, cấu hình và thư viện phải được kiểm tra
trước khi chốt. Không coi các kết quả dự kiến là kết quả đã đạt được.

Bạn đã biết CNN, PyTorch cơ bản và backpropagation. Vì vậy, tập trung học cách xử lý
chuỗi token, attention, mục tiêu dự đoán token tiếp theo và phương pháp nén mô hình.
Đọc [hướng dẫn tuần 1](week01_plan.md) để bắt đầu; dùng [nhật ký](experiment_log.md)
để ghi tiến độ. Chỉ đánh dấu hoàn thành khi có sản phẩm và giải thích được cách làm.

## Chuỗi thực nghiệm

```text
Tuần 1: sử dụng model có sẵn để hiểu inference
                          |
Tuần 2–3: tự xây model nhỏ -> huấn luyện -> checkpoint gốc
                                             |
Tuần 4–5: lượng tử hóa --------------------> benchmark
                                             |
Tuần 6–7: teacher -> student distill -> lượng tử hóa -> so sánh
                                             |
Tuần 8: tổng hợp bằng chứng -> đề cương nghiên cứu tiếp theo
```

GPT-2 dùng ở tuần 1 không có nghĩa phải huấn luyện lại GPT-2 từ đầu ở tuần 3.
RTX 4060 Laptop 8 GB phù hợp bài inference nhập môn; khả năng huấn luyện còn phụ
thuộc số tham số, độ dài chuỗi, batch size, dtype, gradient và trạng thái optimizer.

## Tuần 1 — Hệ sinh thái Hugging Face

**Theo PDF:** dùng `transformers`, `datasets`, `tokenizers`; tải và chạy inference;
hiểu cấu trúc file lưu trữ, đặc biệt `safetensors`. Sản phẩm: notebook và ghi chú.

**Đề xuất thực hành:** dùng GPT-2 124M; quan sát token IDs và logits; sinh tiếp ba
prompt; tạo dataset nhỏ; khảo sát file trọng số. Đọc hướng dẫn tuần 1 theo từng bước.

- [ ] Notebook có output và chạy lại từ kernel mới.
- [ ] Có ba kết quả inference, ghi chú model và nhật ký môi trường.
- [ ] Giải thích được text → token IDs → logits → token mới → text.

## Tuần 2 — Cài đặt kiến trúc Transformer nhỏ

**Theo PDF:** cài kiến trúc quy mô nhỏ theo nanoGPT/nanochat, hiểu thành phần qua
code; sản phẩm là model chạy được forward pass.

**Đề xuất thực hành:** đọc code tham khảo rồi cài lần lượt token/position embedding,
causal self-attention, MLP, residual, normalization và đầu ra vocabulary. So sánh
tensor ảnh `[B, C, H, W]` quen thuộc với chuỗi token `[B, T]` và embedding `[B, T, D]`.
Ghi nguồn code tái sử dụng. Chọn kiến trúc sau khi kiểm tra hướng tích hợp quantization
của tuần 4; chưa chốt số tầng hoặc kích thước hidden.

- [ ] Forward trả logits `[B, T, V]`, không có NaN/Inf.
- [ ] Giải thích được shape và vai trò của từng khối.
- [ ] Kiểm tra causal mask: thay token tương lai không làm đổi logits ở vị trí trước đó khi ở chế độ đánh giá.

## Tuần 3 — Huấn luyện và đánh giá

**Theo PDF:** huấn luyện trên dữ liệu nhỏ, đánh giá sinh văn bản, đối chiếu attention
và backpropagation với code. Sản phẩm: model đã huấn luyện và báo cáo ngắn.

**Đề xuất thực hành:** chọn dữ liệu có giấy phép rõ; tách train/validation trước khi
tạo cửa sổ token. Với chuỗi `[t0, t1, t2, t3]`, input là `[t0, t1, t2]`, target là
`[t1, t2, t3]`. Kiểm tra overfit một batch nhỏ trước khi train đầy đủ. Ghi loss,
lưu checkpoint, tải lại rồi sinh văn bản; chọn TensorBoard hoặc W&B sau khi thử môi trường.

- [ ] Có cấu hình, cách chia dữ liệu và đường train/validation loss.
- [ ] Checkpoint tải lại được; có mẫu sinh văn bản và nhận xét hạn chế.
- [ ] Giải thích được cross-entropy, `backward()` và bước optimizer trong bài toán token.

## Tuần 4 — Post-training quantization

**Theo PDF:** lượng tử hóa model đã huấn luyện tuần 2–3; các công cụ được nêu gồm
bitsandbytes, AutoGPTQ và AutoAWQ. Mục kết quả dự kiến có đề cập GPTQ/AWQ.

**Đề xuất thực hành:** kiểm tra tình trạng hỗ trợ thư viện, Windows/CUDA, kiến trúc
và định dạng checkpoint trước khi chọn công cụ. Thử trên một phần nhỏ trước;
nếu phương pháp cần calibration, dùng dữ liệu riêng từ train, không dùng tập test.
Giữ bản gốc và ghi rõ thuật toán, số bit, phần được lượng tử hóa và cấu hình.
Nếu cần đổi công cụ, giải thích cách vẫn đáp ứng mục tiêu PDF; thay đổi phạm vi
GPTQ/AWQ cần trao đổi với giảng viên.

- [ ] Model lượng tử hóa tải và chạy được; model gốc vẫn giữ nguyên.
- [ ] Có mô tả phương pháp, môi trường và calibration nếu có.

## Tuần 5 — Benchmark

**Theo PDF:** đo perplexity và tốc độ trước/sau quantization; sản phẩm là bảng,
biểu đồ so sánh perplexity, tốc độ và dung lượng model.

**Đề xuất thực hành:** cố định dữ liệu đánh giá, tokenizer, độ dài chuỗi, batch size
và quy tắc sinh. Perplexity đo trên token thật bằng loss, không đo từ ba câu sinh
tuần 1. Đo tốc độ sau warm-up và đồng bộ GPU khi bấm giờ; lặp nhiều lần, báo cáo
cách tổng hợp và số token thực sự sinh. Phân biệt dung lượng checkpoint với VRAM.

- [ ] Có script, cấu hình, số liệu thô, bảng và biểu đồ.
- [ ] Ghi được đánh đổi chất lượng/tốc độ/dung lượng, kể cả khi không nhanh hơn.

## Tuần 6 — Knowledge distillation cơ bản

**Theo PDF:** học lý thuyết Hinton et al., cài soft-label distillation bằng PyTorch
và KL-divergence; sản phẩm là script chạy được trên model nhỏ.

**Đề xuất thực hành:** chọn teacher/student sau khi xác nhận tài nguyên; dùng cùng
tokenizer và vocabulary để so sánh phân phối token trực tiếp. Teacher ở chế độ
đánh giá, không cập nhật gradient. Học temperature, softmax/log-softmax, KL loss
và cách kết hợp với loss nhãn thật; chỉ chốt hệ số sau bài thử nhỏ.

- [ ] Script chạy, loss hữu hạn, gradient cập nhật student mà không cập nhật teacher.
- [ ] Lưu checkpoint student và giải thích được thông tin soft-label cung cấp.

## Tuần 7 — Kết hợp distillation và quantization

**Theo PDF:** lượng tử hóa model đã distill, so sánh với model chưa distill về
chất lượng và hiệu năng; sản phẩm là báo cáo thực nghiệm kết hợp.

**Đề xuất thực hành:** so sánh student cùng kiến trúc được train bằng nhãn thật
với student distill, cùng hai bản lượng tử hóa tương ứng. Ghi ngân sách train,
dữ liệu và cấu hình để phân biệt ảnh hưởng của distillation với khác biệt kiến trúc.
Dùng lại quy trình benchmark tuần 5; teacher có thể làm mốc tham khảo riêng.

- [ ] Có bảng so sánh các điều kiện và log nguồn của từng kết quả.
- [ ] Kết luận dựa trên số đo, không mặc định kết hợp luôn tốt hơn.

## Tuần 8 — Báo cáo và đề cương

**Theo PDF:** tổng hợp 8 tuần, báo cáo tổng kết, xây đề cương cho đồ án chuyên ngành;
công cụ/sản phẩm trình bày có văn bản và slide.

**Đề xuất thực hành:** trình bày câu hỏi, phương pháp, môi trường, dữ liệu, số liệu,
hạn chế và hướng tiếp theo. Mỗi biểu đồ truy ngược được tới log. Đề cương nêu câu
hỏi nghiên cứu cụ thể, baseline, thí nghiệm và nguồn lực cần có.

- [ ] Báo cáo và slide thống nhất với số liệu thực nghiệm.
- [ ] Có đề cương tiếp theo và hướng dẫn tái lập thí nghiệm.

## Quy tắc lưu sản phẩm

- Notebook phục vụ học và quan sát; mã dùng lại dần chuyển vào `src/llm_compression_lab/` khi cần.
- Cấu hình ở `configs/`, kết quả từng tuần ở `outputs/`, báo cáo ở `reports/`.
- Dữ liệu và checkpoint lớn lưu cục bộ trong `data/`, `models/` hoặc cache; không đưa vào Git.
- Mỗi run ghi phiên bản thư viện, revision model, seed, thiết bị, dtype và cấu hình.
- Không đánh dấu hoàn thành chỉ vì tạo được file; cần chạy, lưu bằng chứng và giải thích.

## Tài liệu nghiên cứu được PDF dẫn

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [Quantization and Training of Neural Networks for Integer-Only Inference](https://arxiv.org/abs/1712.05877).
- [GPTQ](https://arxiv.org/abs/2210.17323).
- [AWQ](https://arxiv.org/abs/2306.00978).
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).

Đọc theo tuần tương ứng; chưa cần đọc hết trước khi bắt đầu tuần 1.
