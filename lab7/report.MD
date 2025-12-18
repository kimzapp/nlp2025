# Lab 7: Dependency Parsing với spaCy

## Mục tiêu
- Làm quen với thư viện spaCy cho xử lý ngôn ngữ tự nhiên
- Hiểu về cấu trúc cây phụ thuộc (dependency tree) trong phân tích cú pháp
- Thực hành trực quan hóa và truy xuất thông tin từ cây phụ thuộc
- Áp dụng dependency parsing để trích xuất thông tin có cấu trúc

## 1. Giới thiệu và Cài đặt

### 1.1. Cài đặt spaCy
- Cài đặt thư viện spaCy: `pip install -U spacy`
- Tải mô hình tiếng Anh kích thước trung bình: `spacy download en_core_web_md`
- Mô hình `en_core_web_md` chứa đầy đủ thông tin cho dependency parsing và word vectors

### 1.2. Tải mô hình và khởi tạo
```python
import spacy
nlp = spacy.load("en_core_web_md")
```

## 2. Phân tích câu và trực quan hóa

### 2.1. Phân tích cây phụ thuộc
- Câu ví dụ: "The quick brown fox jumps over the lazy dog."
- Sử dụng `nlp(text)` để phân tích câu thành đối tượng `Doc`
- Mỗi token trong Doc chứa thông tin về:
  - `token.dep_`: Loại quan hệ phụ thuộc
  - `token.head`: Token cha trong cây
  - `token.children`: Các token con

### 2.2. Trực quan hóa với displaCy
```python
from spacy import displacy
displacy.serve(doc, style='dep')
```

**Phân tích câu "The quick brown fox jumps over the lazy dog.":**
1. **Từ gốc (ROOT):** `jumps` - động từ chính của câu
2. **Các từ phụ thuộc của `jumps`:**
   - `fox`: Chủ ngữ (nsubj)
   - `over`: Giới từ bổ nghĩa (prep)
3. **`fox` là head của:** `The`, `quick`, `brown` (các từ bổ nghĩa cho danh từ)

## 3. Truy cập các thành phần trong cây phụ thuộc

### 3.1. Phân tích chi tiết từng token
Câu ví dụ: "Apple is looking at buying U.K. startup for $1 billions"

| Token     | DEP        | HEAD       | HEAD POS | CHILDREN |
|-----------|------------|------------|----------|----------|
| Apple     | nsubj      | looking    | VERB     | []       |
| is        | aux        | looking    | VERB     | []       |
| looking   | ROOT       | looking    | VERB     | [Apple, is, at] |
| at        | prep       | looking    | VERB     | [buying] |
| buying    | pcomp      | at         | ADP      | [startup] |
| U.K.      | compound   | startup    | NOUN     | []       |
| startup   | dobj       | buying     | VERB     | [U.K., for] |
| for       | prep       | startup    | NOUN     | [billions] |
| $         | quantmod   | billions   | NOUN     | []       |
| 1         | compound   | billions   | NOUN     | []       |
| billions  | pobj       | for        | ADP      | [$, 1]   |

## 4. Duyệt cây phụ thuộc để trích xuất thông tin

### 4.1. Tìm chủ ngữ và tân ngữ của động từ
```python
def extract_subject_verb_object(doc):
    for token in doc:
        if token.pos_ == 'VERB':
            verb = token.text
            subject = ''
            obj = ''
            
            for child in token.children:
                if child.dep_ == 'nsubj':
                    subject = child.text
                if child.dep_ == 'dobj':
                    obj = child.text
            
            if subject and obj:
                print(f'Found Triplet: ({subject}, {verb}, {obj})')
```

**Kết quả từ câu "The cat chased the mouse and the dog watched them.":**
- `(cat, chased, mouse)`
- `(dog, watched, them)`

### 4.2. Tìm tính từ bổ nghĩa cho danh từ
```python
def find_adjective_modifiers(doc):
    for token in doc:
        if token.pos_ == 'NOUN':
            adjectives = []
            for child in token.children:
                if child.dep_ == 'amod':  # adjectival modifier
                    adjectives.append(child.text)
            
            if adjectives:
                print(f"Danh từ: '{token.text}' được bổ nghĩa bởi: {adjectives}")
```

**Kết quả từ câu "The big, fluffy white cat is sleeping on the warm mat.":**
- Danh từ: `cat` được bổ nghĩa bởi: `['big', 'fluffy', 'white']`
- Danh từ: `mat` được bổ nghĩa bởi: `['warm']`

## 5. Bài tập tự luyện

### 5.1. Tìm động từ chính của câu

**Hướng tiếp cận:**
- Duyệt lần lượt qua từng token và kiểm tra thuộc tính `dep_`
- Nếu tìm thấy token có `dep_ == 'ROOT'` thì đó chính là động từ chính của câu
- Sử dụng early return để thoát ngay khi tìm thấy (vì mỗi câu chỉ có 1 ROOT)
- Xử lý trường hợp edge case: câu không có ROOT

**Kết quả thực nghiệm:**
- Câu test: "The big, fluffy white cat is sleeping on the warm mat."
- Kết quả: Động từ chính trong câu: `sleeping`

### 5.2. Trích xuất cụm danh từ (Noun chunks)
**Hướng tiếp cận:**
- Tìm tất cả token có `pos_ == 'NOUN'` làm head của cụm danh từ
- Với mỗi danh từ, duyệt qua các `children` để tìm từ bổ nghĩa:
  - `'det'`: Determiner (the, a, an, this, that...)
  - `'amod'`: Adjectival modifier (big, white, beautiful...)  
  - `'compound'`: Compound modifier (noun modifying noun)
- Ghép các thành phần theo thứ tự: [modifiers] + [noun]
- Xử lý trường hợp danh từ cũng có thể là modifier của danh từ khác

**Kết quả thực nghiệm:**
- Câu test: "The big, fluffy white cat is sleeping on the warm mat."
- Các cụm danh từ được trích xuất:
  - `['big', 'fluffy', 'white', 'cat']`
  - `['warm', 'mat']`

### 5.3. Tìm đường đi từ token đến ROOT
**Hướng tiếp cận:**
- Sử dụng thuật toán **DFS (Depth-First Search)** với backtracking
- Từ token xuất phát, thử 2 hướng tìm kiếm:
  1. **Đi xuống:** Duyệt các `children` của token hiện tại
  2. **Đi lên:** Duyệt `head` (parent) của token hiện tại
- Sử dụng `visited` set để tránh cycle (đi vòng lặp vô hạn)
- **Base case:** Khi gặp token có `dep_ == 'ROOT'`, trả về đường đi
- **Recursive case:** Tiếp tục tìm từ các node lân cận chưa thăm

**Chi tiết thuật toán:**
1. Kiểm tra token hiện tại có phải ROOT không
2. Nếu không, thử tìm qua các children trước  
3. Nếu children không dẫn đến ROOT, thử tìm qua parent (head)
4. Sử dụng visited set để đánh dấu các node đã thăm
5. Trả về path dưới dạng list các token text

**Kết quả thực nghiệm:**
- Câu test: "The big, fluffy white cat is sleeping on the warm mat."
- Start token: `white` (index 3)
- **Đường đi tìm được:** `['white', 'cat', 'sleeping']`
- **Giải thích:** white → cat (head) → sleeping (ROOT)

## 6. Phân tích và nhận xét

### 6.1. Độ chính xác của spaCy
- spaCy `en_core_web_md` cho kết quả dependency parsing rất chính xác trên câu tiếng Anh chuẩn
- Các quan hệ phụ thuộc được nhận diện đúng: `nsubj`, `amod`, `prep`, `pobj`...
- ROOT được xác định chính xác (thường là động từ chính của câu)

### 6.2. Ưu điểm của dependency parsing
- **Linh hoạt:** Xử lý được câu có cấu trúc phức tạp, nhiều mệnh đề
- **Hiệu quả:** Thời gian xử lý nhanh, phù hợp cho ứng dụng real-time
- **Đa dụng:** Hỗ trợ nhiều tác vụ downstream như Information Extraction, QA

### 6.3. Hạn chế và thách thức
- **Câu mơ hồ:** Dependency parsing có thể cho kết quả khác nhau với câu có nhiều cách hiểu
- **Domain-specific text:** Hiệu suất giảm với văn bản chuyên ngành (y tế, pháp lý...)
- **Informal text:** Xử lý kém với social media text, typos, abbreviations
