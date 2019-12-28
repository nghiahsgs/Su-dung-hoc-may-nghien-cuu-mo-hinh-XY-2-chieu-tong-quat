# Sử dụng học máy nghiên cứu mô hình XY 2 chiều tổng quát
## Giới thiệu chung <br>
Học máy phát triển mạnh mẽ trong thập kỷ qua nhờ sự bùng nổ của dữ liệu lớn, học máy có tác động mạnh trong các lĩnh vực khoa học và công nghệ. Vật lý chất rắn không đứng ngoài xu hướng này. Tính từ 2015, số lượng của các bài báo xuất bản sử dụng học máy trong vật lý, cụ thể trong vật lý chất rắn, gia tăng đáng kể. Hiện tại có 1 lượng lớn ứng dụng học máy trong vật lý chất rắn: Phân loại pha cho mô hình spin cổ điển hoặc mô hình mạng lượng tử, tìm trạng thái cơ bản (ground state) của hệ lượng tử, tăng tốc mô phỏng Monte Carlo. <br>



Trong đề tài này, chúng tôi nghiên cứu việc sử dụng học máy nghiên cứu mô hình XY 2 chiều tổng quát. Chúng tôi trình bày 2 vấn đề được của việc phân loại pha của mô hình xy 2 chiều tổng quát. Thứ nhất liệu học máy có phân loại pha tốt hơn các phương pháp khác không. Thứ hai xem xét tại sao nó lại phân loại như vậy, nếu hiệu suất nó tốt hơn thì nó đã dùng đặc trưng gì để phân loại và liệu đặc trưng đó có tương ứng với một đại lượng vật lý nào không. <br>

## Chi tiết đề tài
các file chi tiết về đề tài :
Bản tóm tắt đề tài
Slide trình chiếu:

# Mã nguồn và các code được chạy trong đề tài
## Chuẩn bị dữ liệu
Phần quan trọng nhất của machine learning là dữ liệu. Dữ liệu đã được chúng tôi sinh ra và upload trên gg drive tại đây
https://drive.google.com/drive/folders/1YkGS3MEX7yI9D-rcxcqHoeP0lVmlkq2M?usp=sharing

## Code huấn luyện model nằm ở thư mục /codes/createModel
Trong thư mục này có những file sau là quan trọng:
### utils.py: File này chứa các hàm quan trọng
### ml_trainning.py:  File này để sinh ra model deep learning. Khi chạy file nhận 3 tham số. Tham số 1	là loại dữ liệu đầu vào (xy, angle hoặc vortex). Tham số thứ 2 là thư mục trỏ đến dữ liệu đầu vào (vd C:/nghiahsgs/Delta0.2/L12). Tham số  thứ 3 là nơi bạn muốn lưu model của mình (vd C:/nghiahsgs/trained_model)
### post_process_on_test.py: File này sẽ tính tất cả các đại lượng của tập test , gồm label, helicity ... sau đó nén lại thành 1 file h5 (Bạn đọc có thể mở code ra xem)
### plot_on_test.py File này sẽ dùng để vẽ các kết quả trên tập test. Do quá trình làm cần vẽ rất nhiều kết quả nên file này là file phức tạp nhất. Có khoảng 10 chức năng trong đó.
## Tất cả các model đã huấn luyện được lưu tại /codes/model
Do có rất nhiều model và dữ liệu khá nặng nên mình chỉ upload vài model ví dụ. Bạn đọc có thể sinh lại rất cả các model học sâu bằng cách chạy code trong thư mục /codes/createModel


## Code để chạy ra kết quả SHAP
https://drive.google.com/drive/folders/1WdbocrwWUMdxqM7UR6sBMck7_WE8x_FW?usp=sharing

Mọi ý kiến đóng góp hoặc muốn copy thêm dữ liệu xin hãy liên hệ nghiahsgs@gmail.com =))
