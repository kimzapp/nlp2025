import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # Xác định đường dẫn tương đối đến tệp dữ liệu
    # Giả định tệp này chạy ở thư mục gốc và dữ liệu nằm trong thư mục 'data'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "sentiments.csv")

    # 1. Khởi tạo Spark Session
    # appName là tên ứng dụng của bạn sẽ hiển thị trên Spark UI
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    print("Spark Session đã được khởi tạo.")

    try:
        # 2. Tải và chuẩn bị dữ liệu
        df = spark.read.csv(data_path, header=True, inferSchema=True)

        # Chuyển đổi nhãn -1/1 thành 0/1 (vì LogisticRegression thường mong đợi 0/1)
        df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)

        # Xóa các dòng có giá trị 'sentiment' bị rỗng (null)
        df = df.dropna(subset=["sentiment", "text"])

        print(f"Đã tải dữ liệu, tổng số mẫu: {df.count()}")
        df.printSchema()
        df.show(5)

        # Chia dữ liệu thành 2 tập: huấn luyện (80%) và kiểm thử (20%)
        (trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)
        print(f"Số mẫu huấn luyện: {trainingData.count()}")
        print(f"Số mẫu kiểm thử: {testData.count()}")

        # 3. Xây dựng Pipeline tiền xử lý và đặc trưng hóa
        
        # Stage 1: Tách văn bản thành các từ (tokens)
        tokenizer = Tokenizer(inputCol="text", outputCol="words")

        # Stage 2: Loại bỏ các từ dừng (stop words)
        stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

        # Stage 3: Ánh xạ từ thành vector đặc trưng thô (sử dụng Hashing)
        # numFeatures là một tham số quan trọng, 10000 là giá trị phổ biến
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

        # Stage 4: Tính toán trọng số IDF để "giảm trọng số" các từ phổ biến
        idf = IDF(inputCol="raw_features", outputCol="features")

        # 4. Huấn luyện mô hình (Logistic Regression)
        
        # Stage 5: Mô hình Logistic Regression
        lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

        # Gói tất cả các stage vào một Pipeline duy nhất
        pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

        # 5. Huấn luyện Pipeline
        print("Bắt đầu huấn luyện mô hình...")
        model = pipeline.fit(trainingData)
        print("Huấn luyện hoàn tất.")

        # 6. Đánh giá mô hình trên tập kiểm thử
        predictions = model.transform(testData)

        # Hiển thị một vài dự đoán
        predictions.select("text", "label", "prediction", "probability").show(10)

        # Sử dụng MulticlassClassificationEvaluator để tính toán độ chính xác (accuracy)
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction", 
            metricName="accuracy"
        )
        accuracy = evaluator_acc.evaluate(predictions)
        print(f"Test Set Accuracy = {accuracy * 100:.2f}%")

        # Tính toán F1 Score (một chỉ số tốt hơn cho dữ liệu mất cân bằng)
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction", 
            metricName="f1"
        )
        f1 = evaluator_f1.evaluate(predictions)
        print(f"Test Set F1 Score = {f1:.4f}")

    finally:
        # Luôn dừng Spark Session sau khi hoàn tất
        spark.stop()
        print("Spark Session đã dừng.")

if __name__ == "__main__":
    main()