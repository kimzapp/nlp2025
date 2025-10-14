import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split

def main():
    # khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("Spark Word2Vec Demo") \
        .master("local[*]") \
        .getOrCreate()

    print("Spark Session started.")

    # đọc dữ liệu JSON
    # Ví dụ: mỗi dòng trong file là {"text": "some sentence here ..."}
    input_path = r'F:\nlp2025\data\c4-train.00000-of-01024-30K.json'
    df = spark.read.json(input_path)
    print(f"Loaded dataset with {df.count()} rows")

    # tiền xử lý văn bản
    # - Chọn cột text
    # - Chuyển lowercase
    # - Loại bỏ ký tự đặc biệt
    # - Tokenize (chuyển chuỗi thành danh sách từ)
    df_clean = df.select("text") \
        .withColumn("text", lower(col("text"))) \
        .withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", "")) \
        .withColumn("words", split(col("text"), r"\s+")) \
        .filter(col("text").isNotNull())

    # cấu hình và huấn luyện Word2Vec
    word2Vec = Word2Vec(
        vectorSize=100,      # số chiều vector embedding
        minCount=5,          # bỏ từ hiếm
        inputCol="words",
        outputCol="features",
        seed=42
    )

    print("Training Word2Vec model...")
    model = word2Vec.fit(df_clean)
    print("Training completed.")

    # minh hoạ: tìm từ tương tự “computer”
    synonyms = model.findSynonyms("computer", 5)
    print("\nTop 5 từ gần nghĩa với 'computer':")
    for row in synonyms.collect():
        print(f"{row['word']:<15} similarity={row['similarity']:.4f}")

    # dừng Spark
    spark.stop()
    print("Spark Session stopped.")


if __name__ == "__main__":
    main()
