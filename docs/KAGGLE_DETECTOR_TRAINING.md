# Đưa dataset Accident / Vehicle lên Kaggle

## Hai file cần dùng

1. **Dataset ZIP:**
   `data/processed/kaggle/caap_accident_vehicle_dev_20261011.zip`.
   Upload file này thành **Private Kaggle Dataset**.
2. **Notebook:** `notebooks/CAAP_YOLO11_Kaggle.ipynb`.
   Import vào Kaggle Notebook rồi gắn dataset ở trên bằng **Add Input**.

Gói này dùng snapshot sau bước cách ly 13 ảnh lỗi; 4.967 ảnh KEEP có nhãn
YOLO detection 5 class. Nhóm gần giống/tên nguồn/camera được giữ cùng split
train/val phát triển, seed 42, mục tiêu 80/20. Không có test độc lập.
Không upload trực tiếp bộ `accident_vehicle/` gốc: nhãn có polygon lẫn box,
đường dẫn YAML cũ không phù hợp và split cũ có nhóm xuyên tập.

ZIP gồm ảnh/nhãn train/val, YAML, manifest/hash, report, danh sách ảnh bỏ qua,
helper kiểm tra và `yolo11n.pt` pretrained đầu vào. Không chứa ảnh cách ly,
journal review, checkpoint segmentation cũ hoặc kết quả train đã xóa.
Tên bản sao là ID ngắn; manifest giữ ánh xạ tới tên nguồn.

## Chạy notebook

- Chọn **Accelerator → GPU**, bật **Internet**, giữ Dataset và Notebook private.
- Chạy các cell theo thứ tự. Notebook tự tìm gói đã giải nén hoặc ZIP trong input;
  nếu gắn nhiều gói CAAP thì điền `PACKAGE_ROOT` để chọn rõ ràng.
- Cài `ultralytics==8.4.92`, giữ PyTorch CUDA của Kaggle; không dùng môi trường
  CPU trên laptop. Mọi dữ liệu làm việc và output nằm ở `/kaggle/working`.
- Xem overlay nhãn trước khi train. Mặc định YOLO11n detection, 640 px, batch 16,
  AdamW, 100 epoch tối đa, patience 20, seed 42. Đây là baseline ban đầu;
  chưa bảo đảm hội tụ. Nếu GPU hết VRAM, giảm batch và đổi `RUN_NAME` cho lượt mới.
- `SMOKE=True` chạy 1 epoch, ảnh 320, batch 4 trên split đầy đủ để kiểm tra nhanh.
- Cell cuối tạo `<RUN_NAME>_outputs.zip` chứa best/last checkpoint, log, cấu hình,
  protocol và metric từng class. Dùng **Save Version** và tải output về để giữ kết quả.

Dataset và checkpoint đầu vào được kiểm tra hash; split/nhãn/inventory sai sẽ
dừng trước training. Run đang tồn tại không bị ghi đè âm thầm. Notebook lưu
phiên bản Python/PyTorch/CUDA/Ultralytics, GPU, tham số và hash snapshot.
Seed và deterministic không bảo đảm số giống tuyệt đối giữa mọi phần cứng.

## Nếu chưa thấy epoch

Epoch chỉ xuất hiện ở cell **4. Train** sau cài thư viện, nạp/check dữ liệu,
kiểm tra GPU và khởi tạo model. Bản notebook cập nhật in tiến độ copy mỗi
500 file, kiểm tra mỗi 250 ảnh và thông báo `Starting training now`.
Nếu đang ở copy/check ảnh, training chưa bắt đầu; không cần upload ZIP lại.
Nếu dừng lâu hoặc báo lỗi, lấy 5–10 dòng cuối output của cell đang chạy để
phân biệt lỗi cài đặt, chưa gắn dataset, thiếu GPU hoặc lỗi khởi tạo model.
Chạy các cell theo thứ tự hoặc Run All; cài package thành công chưa phải
đã chạy cell training.

## Tiếp tục một lượt bị ngắt

Gắn output cũ làm input. Điền `RESUME_CHECKPOINT` tới `last.pt` hoặc periodic
checkpoint còn optimizer/epoch; cần `training_protocol.json` ở cạnh thư mục
`weights/` và đúng snapshot/class. Cell train dùng `resume=True`.
Checkpoint đã hoàn tất bị strip optimizer không dùng để resume; nếu fine-tune,
tạo lượt mới với protocol riêng. Đường dẫn dataset runtime được tạo lại cố định
ở `/kaggle/working/caap_data.yaml`.

## Phạm vi kết quả

Gói mang nhãn **DEVELOPMENT_ONLY_NOT_PAPER_EVALUATION** vì 533 mục review
và nguồn/ảnh trước augmentation chưa được xác nhận tại thời điểm chuẩn bị.
Val dùng chọn checkpoint và phân tích lỗi; chưa dùng số mAP này làm kết quả
test độc lập cho paper. Hoàn tất [workflow clean](ACCIDENT_DATASET_REVIEW_WORKFLOW.md),
kiểm tra overlap benchmark rồi chốt evaluation riêng trước khi báo kết quả.
Bộ này không có timeline tai nạn; phân loại mức độ nằm ngoài
[phạm vi paper](PAPER_SCOPE.md).

Không tự thay `weights/yolo11_vn.pt` bằng model mới. Khi đưa detector đã kiểm
tra vào CAAP, dùng cùng checkpoint cho mọi biến thể C1/C2/C3; code ánh xạ tên
`Accident, bus, car, motorcycle, truck` sang class ID của pipeline.

## Tạo lại gói sau khi sửa dữ liệu

```powershell
.\.venv\Scripts\python.exe -X utf8 -m code.scripts.package_kaggle_detection --output data/processed/kaggle/caap_accident_vehicle_dev_VERSION
```

Đổi VERSION cho snapshot mới; script không train, không upload và không ghi
đè gói cũ. Ảnh, nhãn, ZIP và checkpoint được `.gitignore` giữ trên máy.

Tài liệu đối chiếu: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks),
[Ultralytics Train](https://docs.ultralytics.com/modes/train).
