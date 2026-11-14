# LLM Compression Lab

Dự án Project I — Quantization & Distillation.

## Tuần 1

Làm quen `transformers`, `datasets`, `tokenizers`; tải và chạy inference một
mô hình có sẵn; tìm hiểu cấu trúc file mô hình, đặc biệt là `safetensors`.
Sản phẩm: notebook inference cơ bản kèm ghi chú cấu trúc file mô hình.

## Đọc và thực hành theo thứ tự

1. Đọc [lộ trình 8 tuần](docs/roadmap_8_weeks.md) để hiểu mục tiêu và sản phẩm từng giai đoạn.
2. Làm theo [hướng dẫn tuần 1](docs/week01_plan.md), bắt đầu từ môi trường và CUDA.
3. Mở [notebook tuần 1](notebooks/week01_huggingface_basics.ipynb), chạy từng phần theo hướng dẫn.
4. Điền [ghi chú model](docs/model_files.md) và [nhật ký](docs/experiment_log.md); lưu báo cáo vào `reports/`.

Hướng dẫn chọn GPT-2 124M để nhập môn và ưu tiên GPU RTX 4060 Laptop 8 GB sau
khi kiểm tra CUDA. Cấu hình thực hành vẫn chưa điền model, notebook chưa có output
được lưu; cập nhật tài liệu không có nghĩa đã hoàn thành bài thực hành.

| Đường dẫn | Mục đích |
| --- | --- |
| `configs/week01.json` | Cấu hình model và inference |
| `notebooks/week01_huggingface_basics.ipynb` | Bài thực hành tuần 1 |
| `src/llm_compression_lab/` | Mã Python dùng lại ở các tuần sau |
| `scripts/check_environment.py` | Kiểm tra thư viện và GPU |
| `docs/roadmap_8_weeks.md` | Lộ trình và đầu ra 8 tuần |
| `docs/week01_plan.md` | Hướng dẫn từng bước và checklist tuần 1 |
| `docs/model_files.md` | Ghi chú các file mô hình |
| `docs/experiment_log.md` | Nhật ký thực nghiệm |
| `data/` | Dữ liệu cục bộ |
| `models/` | Mô hình tải về nếu cần lưu riêng |
| `outputs/week01/` | Kết quả thực nghiệm |
| `reports/` | Báo cáo môn học |

## Bắt đầu trên Windows PowerShell

Mở PowerShell tại thư mục gốc dự án. Làm theo mục **Bước 1** trong
[hướng dẫn tuần 1](docs/week01_plan.md) để chọn Python, cài PyTorch có CUDA,
kiểm tra GPU và mở Jupyter. Hướng dẫn phân biệt lệnh trong terminal với code trong cell.

Lần tải model đầu cần Internet. Chỉ đổi `device` sang `cuda` khi kiểm tra thành công.
Phiên bản trong `requirements.txt` chưa khóa; lưu phiên bản thực tế sau lần chạy thành công.

Không đưa dữ liệu lớn, trọng số mô hình, token truy cập hoặc `.venv` vào Git.
