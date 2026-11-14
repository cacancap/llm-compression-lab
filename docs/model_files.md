# Ghi chú cấu trúc file mô hình

Model ID: **Chưa chọn**  
Revision/commit: **Chưa ghi nhận**

Các tên bên dưới là những trường hợp thường gặp; kiểm tra file thực tế của model đã chọn.

| File | Vai trò | Có trong model đã chọn? |
| --- | --- | --- |
| `config.json` | Cấu hình kiến trúc mô hình | TODO |
| `generation_config.json` | Tham số sinh văn bản mặc định | TODO |
| `tokenizer.json` | Dữ liệu và cấu hình tokenizer | TODO |
| `tokenizer_config.json` | Thiết lập tokenizer | TODO |
| `special_tokens_map.json` | Ánh xạ các token đặc biệt | TODO |
| `vocab.json`, `merges.txt`, `tokenizer.model` | Thành phần tokenizer tùy loại | TODO |
| `*.safetensors` | Tensor trọng số; có thể chia thành nhiều shard | TODO |
| `model.safetensors.index.json` | Ánh xạ tên tensor sang shard | TODO |
| `pytorch_model.bin` | Một định dạng trọng số khác có thể gặp | TODO |
| `README.md` | Model card: mô tả, cách dùng, giới hạn | TODO |

## Câu hỏi cần trả lời sau thực hành

1. Vì sao chỉ có `config.json` thì chưa đủ để suy luận?
2. Vì sao tokenizer cần phù hợp với model?
3. `safetensors` lưu gì? Có phải mã nguồn kiến trúc model không?
4. Model đang nằm ở cache nào trên máy? Tổng dung lượng bao nhiêu?
