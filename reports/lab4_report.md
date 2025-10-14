## Task bonus

1. Chuẩn bị & cấu hình

- Đặt đường dẫn đầu vào `INPUT_PATH = "data/UD_English-EWT/en_ewt-ud-train.txt"`.

- Cấu hình siêu tham số Word2Vec (kích thước vector `VECTOR_SIZE`, `WINDOW`, `MIN_COUNT`, `SG` chọn skip-gram, số epoch...).

- Thiết lập logging để biết tiến độ.

2. Đọc dữ liệu theo luồng (streaming)

- `UDConlluSentenceIterator` là một iterator file-backed: mở file và đọc từng dòng, gom token thành một câu, rồi yield danh sách token khi gặp dòng trống.

- Cách này không load toàn bộ file vào bộ nhớ — chỉ giữ một câu tại một thời điểm.

- Class này cũng có logic fallback: nếu file không phải định dạng CoNLL-U thì sẽ xử lý mỗi dòng như một câu (tokenize bằng whitespace). Vì vậy mã khá chắc chắn cho cả hai trường hợp.

3. Xây vocab (build_vocab)

- `model.build_vocab(sentences_iter)` quét iterator một lần để đếm tần suất từ và xây chỉ mục từ vựng.

- Lưu ý: `build_vocab` tiêu thụ iterator (nó lặp qua nó), nên cần tạo lại iterator (mở file lại) để dùng cho training. Trong code trên ta khởi tạo 2 iterator tách biệt.

4. Huấn luyện (train)

- Sau khi `vocab` được xây, gọi `model.train(sentences_iter, total_examples=model.corpus_count, epochs=EPOCHS)`.

- Vì iterator đọc file trực tiếp, quá trình train cũng là memory-efficient.

- `workers` dùng số CPU nhằm parallel hoá (gensim dùng multi-threading).

5. Lưu model

- `model.save(OUTPUT_MODEL)` lưu toàn bộ Word2Vec model (có thể load lại bằng `Word2Vec.load(...)`).

- Nếu chỉ cần embeddings (KeyedVectors), có thể lưu `model.wv.save()` để giảm dung lượng và chỉ load phần embeddings.

6. Minh hoạ sử dụng

- `model.wv.most_similar('word')` để lấy các từ tương tự.

- Analogy: `model.wv.most_similar(positive=['king','woman'], negative=['man'])` tương đương phép: king - man + woman.

- Trước khi gọi cần kiểm tra các từ có tồn tại trong vocab không (đặc biệt khi `min_count > 1`).

- Kết quả minh hoạ:
```
    Từ tương tự 'game':
        m16             0.4564
        THAT            0.3488
        Vietnam         0.3307
        aspects         0.3232
        confidentiality 0.3149
        beyond          0.3146
        systems.        0.3138
        estimated       0.3129
        missiles        0.3123
        Essie           0.3104

    Phép tương tự: king - man + woman = ?
        lies            0.3454
        rent            0.3212
        be.             0.3189
        pub             0.3107
        strategy        0.3082
```

## Advanced task

- Kết quả chương trình: 
```
    Top 5 từ gần nghĩa với 'computer':
    desktop         similarity=0.6887
    uwowned         similarity=0.6830
    computers       similarity=0.6434
    software        similarity=0.6248
    programming     similarity=0.6195
```
