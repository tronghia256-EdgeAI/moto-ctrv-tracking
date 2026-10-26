# Phạm vi paper — quyết định được lưu

## Quyết định của Nghĩa ngày 12/10/2026

**Bỏ phân loại mức độ tai nạn `minor/severe` khỏi paper hiện tại.**
Đây là quyết định phạm vi, không phải kết quả thực nghiệm mới.

- Không yêu cầu GT severity, train EfficientNet-B0 severity, confusion matrix
  `none/minor/severe` hoặc claim đóng góp phân loại mức độ.
- Giữ detector 5 class: `Accident`, `bus`, `car`, `motorcycle`, `truck`.
  Không đổi class ID hoặc nhãn detector vì quyết định này.
- Tập trung GT tracking, C1/C2, đánh giá C3 có bằng chứng và latency bao gồm detector.
- Nếu giữ ứng dụng cảnh báo, chỉ đánh giá có/không tai nạn theo sự kiện: precision,
  recall, F1, cảnh báo sai/giờ video và độ trễ phát hiện. Cần clip dương/âm tính,
  onset/end, nhóm sự kiện/camera và protocol test độc lập; ảnh box đơn lẻ chưa đủ.
- Không coi C3 là đã được kiểm chứng chỉ vì bỏ severity. Nhánh precision cần
  ablation có/không C3 trên cùng detector và protocol; chưa có thì ghi untested.
- Severity chỉ được mở lại khi Nghĩa yêu cầu thay đổi phạm vi.

## Phân biệt bản thảo và code hiện có

`code/detection/accident_classifier.py` và nhánh alert trong `code/pipeline.py`
vẫn là code legacy dựa trên severity. Tác vụ cập nhật phạm vi paper không thay
runtime, weights hoặc cảnh báo đang dùng. Không mô tả binary alert là đã chạy
chỉ bằng cách đổi nhãn trong hình/bản thảo. Nếu làm nhánh này sau, cần thiết kế,
triển khai, test và chấm event-level riêng; không gửi cảnh báo thật khi benchmark.

Bản thảo JRTIP ở `paper/jrtip/main.tex`; skeleton cũ trong `paper/sections/`
được đồng bộ phạm vi để tránh khôi phục yêu cầu severity ở phiên sau.
Spec dữ liệu hiện hành: [ACCIDENT_DATA_SPEC.md](ACCIDENT_DATA_SPEC.md).

## ROI do Nghĩa chọn cho GT (12/10/2026)

Cho phép polygon ROI theo camera để giới hạn vùng gán nhãn quan sát được.
Giữ GT đầy đủ và ROI riêng; chấm GT và predictions bằng cùng quy tắc tâm đáy
box trong polygon, bao gồm biên. Chốt trước khi xem metric, giữ cố định giữa
variants và khai báo polygon/hash/counts trong báo cáo. ROI không chứng minh
chất lượng vùng xa hoặc test độc lập và không tự hoàn tất human review.

Yêu cầu tiếp theo cùng ngày: tái tạo GT mixed/review chỉ trong ROI, giữ full
GT/lịch sử làm provenance. Runtime phải mask pixel ngoài ROI, loại detection
trước tracker và bỏ track ngoài khỏi state; C1 brightness chỉ tính pixel ROI.
Các kết quả full-frame cũ không tự trở thành kết quả ROI runtime. Đóng băng ROI
cho phiên review và dùng cùng scope/checkpoint cho mọi ablation.
