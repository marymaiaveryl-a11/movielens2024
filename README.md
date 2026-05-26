# MOVIELENS  2024
# Data: MovieLens Beliefs Dataset 2024
The Updated Data Release Ver - Upated on February 8th 2025
The full MovieLens dataset used in this project can be downloaded from the official source at this link:
https://grouplens.org/datasets/movielens/ml_belief_2024/

# Cập nhật Data Release 2 (8/2/2025)
Ngày 8 tháng 2 năm 2025, GroupLens cập nhật bộ dữ liệu với 2 thay đổi quan trọng:
1. Bổ sung phim: Cập nhật file movies.csv để mọi phim trong belief dataset đều
có thông tin tương ứng.
2. Bổ sung ratings: Thêm dữ liệu rating cho một nhóm users mới - bao gồm cả
những users đã thấy giao diện belief elicitation nhưng không trả lời.

This repository only contains the project reports and code for my two university courses: **Machine Learning** and **Data Visualisation**
Each subject include 2 files: code and report

# PART 1 - MACHINE LEARNING
# MovieLens Belief 2024 — Collaborative Filtering Comparison
Solo ML project comparing 7 models (SVD, SVD++, KNN, NMF, NCF) on MovieLens dataset.
Best model: SVD++ (RMSE: 0.8500)

Hệ thống gợi ý phim (Recommender Systems) giúp dự đoán sở thích người dùng và đề
xuất nội dung phù hợp. Báo cáo này nghiên cứu bài toán dự đoán đánh giá phim trên bộ
dữ liệu MovieLens Belief 2024 - bộ dữ liệu đặc biệt của GroupLens Research, chứa cả
dữ liệu đánh giá truyền thống lẫn “beliefs” (dự đoán chủ quan của người dùng về phim
chưa xem).

Bảy mô hình được xây dựng và so sánh: SVD, KNN User-Based, KNN Item-Based,
SVD++, NMF, BaselineOnly và NCF (Neural Collaborative Filtering). Các mô hình
được đánh giá bằng ba phương pháp: train/test split (80/20), Cross-Validation 5-Fold và
Time-Series CV, với các chỉ số RMSE, MAE, Precision@K và Recall@K.

Kết quả cho thấy SVD đạt cân bằng tốt nhất giữa độ chính xác và tốc độ. BaselineOnly 
xác nhận rằng bias người dùng/phim đã giải thích phần lớn phương sai trong
ratings. Time-Series CV phản ánh hiệu suất thực tế chính xác hơn Random CV. Nghiên
cứu cũng cho thấy tiềm năng của máy học trong việc dự đoán tốt hơn trực giác con người
ở bài toán rating phim.

# PART 2 - DATA VISUALISATION
Báo cáo này thực hiện phân tích khám phá dữ liệu (EDA) và trực quan hóa
(Data Visualization) trên bộ dữ liệu MovieLens Belief 2024 - một bộ dữ liệu đặc
biệt của GroupLens Research, chứa cả dữ liệu đánh giá phim truyền thống lẫn “beliefs”
(dự đoán chủ quan của người dùng về phim chưa xem), nhằm bóc tách mối liên hệ giữa
hành vi đánh giá thực tế và kỳ vọng chủ quan (beliefs) của người dùng.

Bộ dữ liệu gồm 5 file CSV với tổng cộng hàng trăm nghìn bản ghi, bao gồm thông tin
đánh giá phim, thể loại, niềm tin dự đoán (beliefs), và lịch sử gợi ý. Quy trình phân tích
bao gồm: giới thiệu và mô tả dữ liệu, xử lý dữ liệu (kiểm tra missing values, duplicates,
tính hợp lý, outliers, feature engineering, scale dữ liệu, phân tích tương quan), vẽ 18 biểu
đồ đa dạng (histogram, boxplot, heatmap, pairplot, bar chart, pie chart, violin plot, KDE
plot, line chart, scatter plot, dashboard tổng hợp) với nhận xét insight sau mỗi biểu đồ.

Kết quả trực quan hóa cho thấy: 
(1) Phân bố rating lệch trái với 4.0 là phổ biến nhất,
(2) Drama/Comedy/Action chiếm đa số, 
(3) Phim cũ có rating cao hơn do survivorship
bias, 
(4) Người dùng có xu hướng dự đoán lạc quan hơn thực tế, 
(5) Ma trận user-item rất thưa (>99%), và 
(6) Phân bố long-tail rõ rệt.


