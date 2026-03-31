# MovieLens Belief 2024 — Collaborative Filtering Comparison
Solo ML project comparing 7 models (SVD, SVD++, KNN, NMF, NCF) on MovieLens dataset.
Best model: SVD++ (RMSE: 0.8500)

# Tóm tắt
Hệ thống gợi ý phim (Recommender Systems) giúp dự đoán sở thích người dùng và đề
xuất nội dung phù hợp. Báo cáo này nghiên cứu bài toán dự đoán đánh giá phim trên bộ
dữ liệu MovieLens Belief 2024 - bộ dữ liệu đặc biệt của GroupLens Research, chứa cả
dữ liệu đánh giá truyền thống lẫn “beliefs” (dự đoán chủ quan của người dùng về phim
chưa xem).

Bảy mô hình được xây dựng và so sánh: SVD, KNN User-Based, KNN Item-Based,
SVD++, NMF, BaselineOnly và NCF (Neural Collaborative Filtering). Các mô hình
được đánh giá bằng ba phương pháp: train/test split (80/20), Cross-Validation 5-Fold và
Time-Series CV, với các chỉ số RMSE, MAE, Precision@K và Recall@K.

Kết quả cho thấy SVD đạt cân bằng tốt nhất giữa độ chính xác và tốc độ. BaselineOnly xác nhận rằng bias người dùng/phim đã giải thích phần lớn phương sai trong
ratings. Time-Series CV phản ánh hiệu suất thực tế chính xác hơn Random CV. Nghiên
cứu cũng cho thấy tiềm năng của máy học trong việc dự đoán tốt hơn trực giác con người
ở bài toán rating phim.
