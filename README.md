# LLM Compression Lab

Dự án Project I — Quantization & Distillation.

## Tuần 1

Làm quen `transformers`, `datasets`, `tokenizers`; tải và chạy inference một
mô hình có sẵn; tìm hiểu cấu trúc file mô hình, đặc biệt là `safetensors`.
Sản phẩm: notebook inference cơ bản kèm ghi chú cấu trúc file mô hình.

Khung này được tạo theo mục tiêu tuần 1. Danh sách đường dẫn đã trao đổi trước
chưa được truy xuất lại; model và dataset chưa được chốt.

| Đường dẫn | Mục đích |
| --- | --- |
| `configs/week01.json` | Cấu hình model và inference |
| `notebooks/week01_huggingface_basics.ipynb` | Bài thực hành tuần 1 |
| `src/llm_compression_lab/` | Mã Python dùng lại ở các tuần sau |
| `scripts/check_environment.py` | Kiểm tra thư viện và GPU |
| `docs/week01_plan.md` | Checklist tuần 1 |
| `docs/model_files.md` | Ghi chú các file mô hình |
| `docs/experiment_log.md` | Nhật ký thực nghiệm |
| `data/` | Dữ liệu cục bộ |
| `models/` | Mô hình tải về nếu cần lưu riêng |
| `outputs/week01/` | Kết quả thực nghiệm |
| `reports/` | Báo cáo môn học |

## Bắt đầu trên Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m jupyterlab
```

Điền `model_id` trong `configs/week01.json` bằng ID của một mô hình
causal language model tương thích `AutoModelForCausalLM`, rồi mở notebook.
Lần chạy đầu cần Internet để tải model; chưa tải model hay cài thư viện khi tạo khung.
Notebook mặc định dùng CPU; chỉ đổi sang `cuda` sau khi kiểm tra môi trường.
Phiên bản thư viện chưa khóa; ghi phiên bản thực tế vào nhật ký sau lần chạy thành công.

Không đưa dữ liệu lớn, trọng số mô hình, token truy cập hoặc `.venv` vào Git.
