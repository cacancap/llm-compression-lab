# Kế hoạch tuần 1

## Mục tiêu và đầu ra

Notebook chạy inference cơ bản và ghi chú cấu trúc file mô hình Hugging Face.

## Checklist

- [x] Tạo khung thư mục và notebook khởi đầu.
- [ ] Tạo môi trường Python riêng và cài các thư viện.
- [ ] Chạy `scripts/check_environment.py`, ghi phiên bản Python, thư viện, GPU.
- [ ] Chọn model phù hợp tài nguyên máy và điền `configs/week01.json`.
- [ ] Đọc model card, ghi kiến trúc, số tham số, giấy phép và revision.
- [ ] Chạy tokenization; quan sát input IDs và attention mask.
- [ ] Tải model, chạy inference, giải mã đầu ra.
- [ ] Thử ít nhất ba prompt và ghi nhận kết quả.
- [ ] Tìm hiểu vai trò của `datasets`, `tokenizers`, `transformers`.
- [ ] Kiểm tra danh sách file model và hoàn thành `model_files.md`.
- [ ] Lưu notebook có kết quả, nhật ký thực nghiệm và báo cáo ngắn.

## Hoàn thành khi

Notebook chạy được từ đầu đến cuối trong môi trường đã ghi nhận;
giải thích được luồng text → token IDs → model → token IDs → text;
phân biệt được cấu hình, tokenizer và trọng số mô hình.

Tuần 1 chưa triển khai quantization hay distillation.
