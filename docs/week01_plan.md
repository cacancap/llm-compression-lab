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

## Cách dùng hướng dẫn

Đọc [lộ trình 8 tuần và nguồn PDF](roadmap_8_weeks.md) để phân biệt yêu cầu gốc
với đề xuất thực hành dưới đây. Checklist phía trên được giữ nguyên; chỉ đánh dấu
khi chính bạn đã chạy và lưu bằng chứng. Tài liệu này chưa chứng nhận môi trường
hay notebook đã chạy thành công.

Lệnh `powershell` chạy tại thư mục gốc repository. Code `python` chạy trong cell
của [notebook tuần 1](../notebooks/week01_huggingface_basics.ipynb).
Tên mục và ID cell bên dưới đối chiếu với notebook hiện có; ID nằm trong file
JSON, không nhất thiết hiển thị trong Jupyter. **Có sẵn** nghĩa là chạy cell hiện
có; **Bổ sung** nghĩa là bạn tự thêm cell hoặc sửa đúng đoạn được chỉ dẫn khi học.
Việc viết tài liệu này không thay đổi notebook hoặc cấu hình.

## Bước 1 — Môi trường riêng và GPU

**Mục tiêu:** notebook dùng đúng Python và PyTorch nhận GPU.
**Nơi thao tác:** PowerShell, sau đó một cell mới ở đầu notebook.
**Kiến thức:** driver NVIDIA, CUDA runtime của PyTorch và Python environment là
ba thành phần khác nhau. Dòng CUDA của `nvidia-smi` không chứng minh đã cài CUDA
Toolkit hay PyTorch có CUDA. Dùng wheel PyTorch dựng sẵn cho bài này không cần tự
cài Toolkit để biên dịch code.

### 1.1. Kiểm tra trước khi cài

```powershell
Get-Location
python --version
python -c "import sys; print(sys.executable)"
nvidia-smi
```

Thư mục làm việc phải chứa `README.md`, `requirements.txt`, `configs/` và
`notebooks/`. Dùng Python 64-bit; khuyến nghị Python 3.11 hoặc 3.12 cho đường
hướng dẫn này. Nếu `python` trỏ tới bản khác, chọn interpreter 3.11/3.12 đã cài
hoặc cài một bản tương ứng trước khi tạo môi trường. Không ghi đè môi trường
`.venv` đang dùng; nếu đã có, kiểm tra interpreter của nó trước.

Kiểm tra ngày 14/11/2026 trên máy: RTX 4060 Laptop, 8188 MiB VRAM, driver 560.94,
`nvidia-smi` báo CUDA 12.6. Đây là ảnh chụp trạng thái, hãy ghi kết quả của lần chạy
hiện tại vào nhật ký.

### 1.2. Tạo môi trường và cài thư viện

Khi `python` đã trỏ tới interpreter phù hợp, chạy từng lệnh:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py
```

Đây là **một cấu hình đề xuất để bắt đầu**, chưa được cài hoặc kiểm thử trong lần
viết tài liệu. PyTorch công bố wheel 2.9.1 CUDA 12.6 cho Windows/Linux trên
[trang phiên bản chính thức](https://pytorch.org/get-started/previous-versions/#v2-9-1).
Chỉ cài `torch` vì bài này không dùng `torchvision`/`torchaudio`. Cài bản CUDA trước
để `requirements.txt` không tự chọn một bản torch khác trên môi trường trống;
không thêm `--upgrade` vào lệnh cài requirements.

Driver 560.94 vượt mức driver Windows 560.76 được NVIDIA liệt kê cho CUDA 12.6 GA.
Đây là căn cứ chọn `cu126`, không phải cam kết mọi tổ hợp thư viện đều chạy được.
Xem [CUDA 12.6 release notes, bảng driver](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)
và [bộ chọn cài đặt PyTorch](https://pytorch.org/get-started/locally/).
Không tự đổi sang CUDA 13 với driver đang có. Nếu đổi phiên bản, đối chiếu lại
hướng dẫn chính thức và ghi cấu hình mới.

### 1.3. Kiểm tra CUDA thực sự chạy được

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe -c "import torch; x = torch.ones(2, device='cuda'); print(x + 1); print(torch.cuda.get_device_name(0))"
```

Chỉ chạy lệnh thứ hai nếu lệnh đầu trả `True`. Kết quả mong đợi là tensor `[2., 2.]`
trên GPU và tên RTX 4060. Đây mới là phép thử tối thiểu, chưa phải inference model.

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name llm-compression-lab --display-name "Python (llm-compression-lab)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Chọn kernel **Python (llm-compression-lab)**. **Bổ sung** cell đầu notebook:

```python
import sys
print(sys.executable)
```

Đường dẫn phải trỏ tới `.venv\Scripts\python.exe` của dự án.

**Lỗi thường gặp:** `No matching distribution` cần kiểm tra phiên bản/kiến trúc
Python và wheel; `torch.version.cuda is None` là bản torch không có CUDA;
terminal thấy CUDA nhưng notebook không thấy thường do khác kernel. Nếu lỗi
driver hoặc DLL, ghi đầy đủ thông báo trước khi sửa. Nếu muốn tiếp tục bài học
trong lúc xử lý GPU, chọn `device: cpu` ở bước 2 và ghi rõ đang chạy CPU.

- [ ] Lưu phiên bản, kết quả CUDA, GPU và interpreter notebook vào [nhật ký](experiment_log.md).

## Bước 2 — Chọn model và đọc cấu hình

**Mục tiêu:** xác định mình đang chạy model nào và vì sao phù hợp.
**Nơi thao tác:** [configs/week01.json](../configs/week01.json), cell đọc cấu hình
`week01-01`, mục **1. Môi trường** (`week01-03`) đều **có sẵn**.

**Kiến thức:** GPT-2 bản này có 124M tham số, là causal language model sinh tiếp
văn bản tiếng Anh, không phải chatbot được huấn luyện làm theo yêu cầu.
Model đủ nhỏ cho bài inference một prompt ngắn trên GPU 8 GB; chưa cần nén.
Xem [model card](https://huggingface.co/openai-community/gpt2) để ghi kiến trúc,
giấy phép MIT, ngôn ngữ và giới hạn. Việc chọn model cho tuần 1 không quyết định
kiến trúc tự xây ở tuần 2.

Sau khi kiểm tra CUDA thành công, tự điền file cấu hình:

```json
{
  "model_id": "openai-community/gpt2",
  "device": "cuda",
  "prompt": "Machine learning is",
  "max_new_tokens": 50,
  "seed": 42
}
```

Nếu GPU chưa sẵn sàng, dùng `"device": "cpu"`. Chạy cell đọc config rồi cell
môi trường. `seed` kiểm soát nguồn ngẫu nhiên; chưa đủ để bảo đảm kết quả giống
từng bit giữa mọi phiên bản/phần cứng. Bài này dùng greedy decoding để dễ quan sát.

**Kết quả mong đợi:** config hiển thị đúng model và thiết bị; import thành công.
**Lỗi thường gặp:** JSON thiếu dấu phẩy/dùng nháy đơn; chưa lưu file; đổi config
nhưng chưa chạy lại cell nên biến `config` vẫn giữ giá trị cũ.

- [ ] Điền cấu hình, đọc model card; ghi revision thực tế ở bước 7.

## Bước 3 — Tokenization và hai loại mask

**Mục tiêu:** hiểu cách câu chữ trở thành input cho model.
**Nơi thao tác:** mục **2. Tokenizer**, cell `week01-05` **có sẵn**.

**Kiến thức:** tokenizer chia văn bản thành token rồi ánh xạ thành số nguyên.
Token không nhất thiết là một từ hoàn chỉnh; ID không phải embedding hay xác suất.
Model sẽ tra embedding từ các ID này. Trong bài này:

- `input_ids` có shape `[B, T]`: số câu trong batch và số token.
- `attention_mask` do tokenizer trả về thường có 1 cho token thật, 0 cho padding.
- **Causal mask** ngăn vị trí hiện tại nhìn token tương lai; model áp dụng bên trong.
  Một câu không padding có attention mask toàn 1 vẫn bị ràng buộc causal.

Chạy cell có sẵn rồi **bổ sung** ngay sau nó:

```python
print("Input shape:", tuple(encoded["input_ids"].shape))
print("Attention mask:", encoded["attention_mask"].tolist())
print("Round trip:", tokenizer.decode(encoded["input_ids"][0]))
```

**Kết quả mong đợi:** batch size bằng 1, chuỗi token giải mã lại thành prompt.
Quan sát sự khác nhau giữa token dạng ký hiệu và văn bản đã decode.
**Lỗi thường gặp:** hiểu token ID như giá trị mang nghĩa lớn/nhỏ; nghĩ một từ luôn
là một token; tưởng attention mask toàn 1 cho phép model nhìn tương lai.

- [ ] Ghi một ví dụ token/ID và giải thích được hai loại mask.

## Bước 4 — Forward pass, logits và inference

**Mục tiêu:** nối kiến thức CNN đã biết với bài toán dự đoán token tiếp theo.
**Nơi thao tác:** mục **3. Model và inference**, cell `week01-07` **có sẵn**.
Chạy cell này trước; nó đã tải model, chuyển lên `device`, đặt `eval()` và sinh văn bản.

**Kiến thức:** nếu CNN phân loại trả logits `[B, C]`, language model trả logits
`[B, T, V]`: một vector điểm số cho vocabulary tại mỗi vị trí chuỗi. Logits chưa
phải xác suất. Logits ở vị trí cuối dự đoán token tiếp theo prompt.

**Bổ sung** cell ngay sau cell inference:

```python
with torch.inference_mode():
    forward_output = model(**inputs)

logits = forward_output.logits
print("Logits shape:", tuple(logits.shape))
next_token_id = logits[0, -1].argmax().item()
print("Next token ID:", next_token_id)
print("Next token text:", repr(tokenizer.decode([next_token_id])))
print("Parameter count:", sum(p.numel() for p in model.parameters()))
print("Device:", next(model.parameters()).device)
print("Dtype:", next(model.parameters()).dtype)
```

`argmax` chọn điểm số lớn nhất; không cần softmax để tìm vị trí lớn nhất.
Một forward pass cung cấp dự đoán; `generate()` lặp dự đoán rồi nối token mới,
có thể dùng cache để tránh tính lại toàn bộ phần trước.

| Khái niệm | Ý nghĩa |
| --- | --- |
| Training | Tính loss với target, backpropagate, cập nhật trọng số bằng optimizer |
| Inference | Dùng trọng số hiện có để dự đoán, không cập nhật chúng |
| `model.eval()` | Chuyển hành vi các lớp như dropout sang đánh giá; không tự tắt gradient |
| `torch.inference_mode()` | Tắt việc theo dõi autograd và một số overhead; không tự gọi `eval()` |

**Kết quả mong đợi:** logits có shape `[1, T, V]`, device khớp config; hiểu được
vì sao inference thường kết hợp cả hai lệnh trên. Notebook chưa chỉ định dtype,
nên ghi giá trị thực tế, không mặc định rằng chạy GPU tự chuyển sang FP16.
**Lỗi thường gặp:** model trên GPU nhưng input còn ở CPU; chạy cell forward trước
khi `model`/`inputs` được tạo; gọi `backward()` trong `inference_mode()`.

- [ ] Giải thích shape, logits vị trí cuối và sự khác nhau giữa `eval()`/`inference_mode()`.

## Bước 5 — Ba prompt và cách lưu đúng từng lần

**Mục tiêu:** có kết quả thực tế, tránh lẫn output của prompt cũ với cấu hình mới.
**Nơi thao tác:** config và các cell tokenizer, inference, **5. Lưu kết quả**
(`week01-11`) **có sẵn**.

Dùng lần lượt:

```text
Machine learning is
Once upon a time,
The purpose of a university is
```

Giữ seed 42, `max_new_tokens=50`, `do_sample=False`. Với mỗi prompt: sửa và lưu
config → chạy lại cell đọc config/môi trường → tokenizer → inference → cell
quan sát bổ sung → cell lưu kết quả. Không cần restart kernel giữa ba prompt;
cell hiện có sẽ gọi tải model lại, thường dùng cache đã tải.

**Bổ sung** sau inference, trước khi lưu:

```python
prompt_length = inputs["input_ids"].shape[1]
new_ids = output_ids[0, prompt_length:]
new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
print("New token count:", new_ids.numel())
print("Continuation:", new_text)
```

Cell lưu hiện có ghi `generated_text` gồm cả prompt, config, revision, dtype và
phiên bản thư viện. **Bổ sung** ngay sau khi tạo dict `result` và trước
`output_path.write_text(...)` trong cùng cell:

```python
result["continuation"] = new_text
result["new_token_count"] = new_ids.numel()
result["do_sample"] = False
```

**Kiến thức:** 50 là giới hạn số token mới, không phải 50 từ và không bảo đảm
đủ 50 token nếu model gặp điều kiện dừng. Greedy decoding có thể sinh văn bản lặp.
Ba prompt giúp quan sát hành vi, chưa phải thí nghiệm đo chất lượng tổng quát.

**Kết quả mong đợi:** ba file `inference_*.json` riêng trong [outputs/week01](../outputs/week01/),
mỗi file khớp prompt, có phần sinh mới và số token. Ghi nhận lặp từ, câu bị ngắt,
nội dung khó hiểu; không chỉnh output để làm kết quả đẹp hơn.
**Lỗi thường gặp:** chỉ chạy lại cell lưu nên lưu output cũ; đổi `prompt` trong
biến mà không đồng bộ config; nhầm phần prompt được in lại là token model mới sinh.

- [ ] Mở lại ba file, đối chiếu prompt/output và ghi nhận xét vào nhật ký.

## Bước 6 — Làm quen `datasets`

**Mục tiêu:** hiểu cách quản lý và biến đổi dữ liệu trước khi huấn luyện.
**Nơi thao tác:** **bổ sung** cell sau mục **5. Lưu kết quả**, trước phần ghi chú cuối.
Chạy sau khi tokenizer đã được khởi tạo.

```python
from datasets import Dataset

prompt_dataset = Dataset.from_dict({
    "text": [
        "Machine learning is",
        "Once upon a time,",
        "The purpose of a university is",
    ]
})
tokenized_dataset = prompt_dataset.map(
    lambda batch: tokenizer(batch["text"]),
    batched=True,
)
print(prompt_dataset)
print(tokenized_dataset.column_names)
print(tokenized_dataset[0])
```

**Kiến thức:** `datasets` quản lý bảng dữ liệu và phép biến đổi; `tokenizers` xử lý
chia/mã hóa token; `transformers` cung cấp model và API kết nối tokenizer/model.
Các thư viện phối hợp với nhau. `batched=True` ở đây áp dụng hàm theo nhóm dòng,
chưa tạo batch tensor có padding để đưa vào model.

**Kết quả mong đợi:** dataset có 3 dòng, thêm `input_ids` và `attention_mask`.
**Lỗi thường gặp:** yêu cầu tensor cho các chuỗi dài khác nhau mà chưa padding;
nhầm ba prompt này là bộ dữ liệu đủ để huấn luyện hoặc đánh giá LLM.

- [ ] Giải thích được vai trò từng thư viện và tác dụng của `map`.

## Bước 7 — Cấu trúc file, revision và `safetensors`

**Mục tiêu:** biết model nằm ở đâu và các file chứa gì.
**Nơi thao tác:** mục **4. Khảo sát file trên Hugging Face**, cell `week01-09`
**có sẵn**; điền [model_files.md](model_files.md). Có thể chạy mục này trước bước 5
theo thứ tự notebook; phần đọc tensor dưới đây chạy sau khi model tải xong.

Cell liệt kê file trên Hub không đồng nghĩa đã tải tất cả file đó. Ghi riêng SHA
của repository và SHA model đã tải. Nếu khác nhau, không coi chúng là một phiên bản.
Để tái lập một lần chạy sau này, bổ sung `revision="<recorded commit SHA>"` bằng
SHA thật vào cả `AutoTokenizer.from_pretrained` và `AutoModelForCausalLM.from_pretrained`;
không dán nguyên placeholder. Cấu hình hiện tại chưa có trường revision.

**Bổ sung** sau cell khảo sát file để tìm snapshot đã có cục bộ và chỉ đọc header
trọng số; lệnh này không tải thêm file:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open

loaded_revision = getattr(model.config, "_commit_hash", None)
if loaded_revision is None:
    raise RuntimeError("Loaded revision is missing; inspect the cache manually")
snapshot_dir = Path(snapshot_download(
    repo_id=config["model_id"],
    revision=loaded_revision,
    local_files_only=True,
))
print("Local snapshot:", snapshot_dir)
local_files = [path for path in snapshot_dir.rglob("*") if path.is_file()]
for path in local_files:
    print(path.relative_to(snapshot_dir), path.stat().st_size)
print("Snapshot logical bytes:", sum(path.stat().st_size for path in local_files))

weight_files = sorted(snapshot_dir.glob("*.safetensors"))
print("Safetensors weight bytes:", sum(path.stat().st_size for path in weight_files))
if weight_files:
    with safe_open(str(weight_files[0]), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        print("Tensor names:", names[:5])
        first_name = names[0]
        print("First tensor shape:", handle.get_slice(first_name).get_shape())
else:
    print("No safetensors file found in this local snapshot")
```

**Kiến thức:** `config.json` mô tả kiến trúc; file tokenizer mô tả cách mã hóa;
`safetensors` lưu tensor trọng số, không chứa code kiến trúc; model card giải thích
nguồn gốc và giới hạn. Đọc tên/shape tensor giúp liên hệ trọng số với lớp trong model.
Tổng kích thước logic của snapshot không nhất thiết bằng dung lượng chiếm đĩa
vì cache có thể dùng liên kết hoặc chia sẻ blob. Nó cũng không phải VRAM khi chạy.

**Kết quả mong đợi:** ghi đúng file có/không có, revision, đường dẫn cache, dung
lượng trọng số và ví dụ tên/shape tensor. Không mặc định mọi model có đủ các file
trong bảng mẫu. Nếu không có safetensors, kiểm tra file thực tế đã tải và cách nạp
model trước khi kết luận; không cần tải toàn bộ định dạng khác trong repository.
**Lỗi thường gặp:** chỉ nhìn file trên Hub rồi ghi là đã có cục bộ; cộng cả các
định dạng trọng số trùng nhau; nhầm `config.json` đủ để tái tạo model đã huấn luyện.

- [ ] Hoàn thiện bảng và bốn câu hỏi trong `model_files.md` bằng dữ liệu thực tế.

## Bước 8 — Nhật ký, báo cáo và chạy lại

**Mục tiêu:** người khác đọc được cách làm, bạn có thể tái lập kết quả.
**Nơi thao tác:** mục **6. Ghi chú sau thực hành** **có sẵn**,
[experiment_log.md](experiment_log.md), và **tạo mới khi học** `reports/week01_report.md`.

Trong PowerShell sau khi môi trường chạy thành công:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 outputs/week01/environment.txt
```

Đây là ảnh chụp phiên bản để tham khảo, chưa thay thế lockfile đa nền tảng.
Ghi thêm Python, driver, `torch.version.cuda`, GPU, device, dtype và revision model.
Mỗi lần chạy có mục tiêu, prompt, tham số, đường dẫn JSON và nhận xét.

**Bổ sung** báo cáo khoảng 1–2 trang theo mẫu nội dung:

```markdown
# Week 01 Report

## Objective
## Environment and Model
## Method
## Results
## Model Files
## Lessons Learned and Next Steps
```

Viết phần giải thích bằng tiếng Việt: mục tiêu, chuỗi text → token → model → text,
bảng ba prompt/kết quả, cấu trúc file, lỗi đã xử lý và câu hỏi còn chưa hiểu.
Nêu rõ model card là nguồn thông tin model; các output và số đo là kết quả của bạn.

Cuối cùng, giữ một prompt trong config rồi **Restart Kernel and Run All** để kiểm
tra notebook không phụ thuộc biến sót từ lần chạy trước. Ba JSON trước đó vẫn giữ;
chạy lại sẽ tạo thêm một JSON. Lưu notebook bằng `Ctrl+S` sau khi chạy xong.

**Kiến thức:** notebook chạy được từng cell chưa chắc chạy lại từ đầu được.
**Kết quả mong đợi:** notebook có output, ba kết quả ban đầu, ghi chú, nhật ký và
báo cáo liên kết được với nhau.
**Lỗi thường gặp:** chỉ lưu code mà chưa lưu output; thiếu biến do sai thứ tự cell;
ghi CUDA/model revision theo dự định thay vì giá trị thực tế.

- [ ] Chạy lại từ kernel mới thành công và cập nhật checklist đầu tài liệu.
- [ ] Có đủ sản phẩm, không đưa token truy cập, `.venv`, cache hoặc trọng số lớn vào Git.

## Tự kiểm tra trước khi chuyển sang tuần 2

1. Token khác từ và embedding như thế nào?
2. Logits `[B, T, V]` nghĩa là gì, vì sao lấy vị trí cuối để sinh token tiếp theo?
3. Attention mask toàn 1 có làm mất causal mask không?
4. `eval()` có tắt gradient không? `inference_mode()` có tự tắt dropout không?
5. Vì sao chỉ có `config.json` chưa đủ để chạy model đã huấn luyện?
6. Vì sao ba đoạn văn sinh ra chưa đủ để kết luận model tốt hoặc xấu?

Nếu còn vướng, ghi câu hỏi vào nhật ký rồi làm lại đúng cell liên quan.

## Bài mở rộng tùy chọn

Sau khi hoàn thành GPT-2, có thể thử
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) để học
`apply_chat_template` và phân biệt completion với hội thoại. Đọc model card và
thử trên notebook riêng; không chỉ thay model ID rồi coi prompt của GPT-2 là
định dạng hội thoại phù hợp. Bài này không phải điều kiện hoàn thành tuần 1.
