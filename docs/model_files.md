# Ghi chú cấu trúc file mô hình

Model ID: **gpt2**  
Cached revision: **607a30d783dfa663caf39e06633721c8d4cfcd7e**  
Revision yêu cầu trong lần chạy mới: **607a30d783dfa663caf39e06633721c8d4cfcd7e**.  
Thuộc tính nội bộ `model.config._commit_hash` trả `None`; JSON giữ `model_revision: null`, không suy diễn giá trị này. Cả tokenizer và model đều nhận `revision` từ cấu hình.  
Weight file: `model.safetensors`  
Weight file size: 548105171 bytes, khoảng 522.71 MiB.  
Stored tensor count: 160.  

## Model card

- Model: GPT-2, bản nhỏ khoảng 124 triệu tham số.
- Kiến trúc: Transformer dùng cho causal language modeling.
- Ngôn ngữ: tiếng Anh.
- Mục tiêu huấn luyện: dự đoán token tiếp theo.
- Giấy phép: MIT.
- Giới hạn: văn bản sinh ra có thể sai hoặc mang thiên lệch;
  không xem output là nguồn thông tin đã được kiểm chứng.
- Nguồn: https://huggingface.co/openai-community/gpt2

Bảng dưới phản ánh snapshot đã kiểm tra, không phải toàn bộ repository trên Hub. “Không thấy” không có nghĩa file không tồn tại trên Hub.

| File | Vai trò | Có trong snapshot cục bộ? |
| --- | --- | --- |
| `config.json` | Cấu hình kiến trúc mô hình | Có |
| `generation_config.json` | Tham số sinh văn bản mặc định | Có |
| `tokenizer.json` | Dữ liệu và cấu hình tokenizer | Có |
| `tokenizer_config.json` | Thiết lập tokenizer | Có |
| `special_tokens_map.json` | Ánh xạ các token đặc biệt | Không thấy |
| `vocab.json`, `merges.txt` | Từ vựng và quy tắc ghép BPE | Có |
| `tokenizer.model` | File tokenizer của một số loại model khác | Không thấy |
| `*.safetensors` | Tensor trọng số; có thể chia thành nhiều shard | Có |
| `model.safetensors.index.json` | Ánh xạ tên tensor sang shard | Không thấy |
| `pytorch_model.bin` | Một định dạng trọng số khác có thể gặp | Không thấy |
| `README.md` | Model card: mô tả, cách dùng, giới hạn | Không thấy |

## Câu hỏi cần trả lời sau thực hành

1. Vì sao chỉ có `config.json` thì chưa đủ để suy luận?  
   config.json mô tả cấu hình kiến trúc, nhưng không chứa các giá trị trọng số đã học. Muốn khôi phục model đã huấn luyện, cần nạp thêm trọng số; để xử lý văn bản, cần tokenizer phù hợp.
2. Vì sao tokenizer cần phù hợp với model?  
   Tokenizer dùng cách mã hoá và ánh xạ các token-ID phù hợp với quá trình huấn luyện model. Vì vậy nếu tokenizer mới ánh xạ token với token ID khác thì model sẽ hiểu sai.
3. `safetensors` lưu gì? Có phải mã nguồn kiến trúc model không?  
   `safetensors` là định dạng lưu tensor, không chứa mã nguồn kiến trúc. File có thể chứa trọng số và buffer; 160 tensor được lưu không đồng nghĩa 160 tham số trainable.
4. Model đang nằm ở cache nào trên máy? Tổng dung lượng bao nhiêu?  
   Model được lưu trong snapshot:
`D:\Dumps\huggingface_cache\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e`

    File trọng số `model.safetensors` có dung lượng 548105171 byte,
    khoảng 522.71 MiB. Chưa đo chính xác tổng dung lượng các file
    trong snapshot. Dung lượng file trên đĩa không đồng nhất với
    VRAM sử dụng khi inference.
