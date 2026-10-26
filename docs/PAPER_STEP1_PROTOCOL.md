# Bước 1: chốt protocol development và kiểm tra chéo GT ROI

Ba phiên GT tracking đã được Nghĩa hoàn tất, nhưng so sánh tracker còn có
nguy cơ thiên lệch từ tracker sinh prelabels. Bước này bổ sung một lượt kiểm
tra các điểm bất đồng, không tự thay đổi những quyết định đã lưu.

## Chuẩn bị và mở cửa sổ

```powershell
.\.venv\Scripts\python.exe -X utf8 -m code.scripts.paper_gt_audit prepare
```

Mở **`review_gt_audit.cmd`**. Chương trình hiển thị từng mục riêng, ẩn tên
tracker và segment, tự lưu quyết định/ghi chú. Tên người review là Nghĩa.
Tick **Xem toàn khung** nếu crop không đủ ngữ cảnh.

Phiên đã chuẩn bị ngày 12/10/2026: `roi_241dc106947b1723`, **148 mục**
(92 box, 56 identity), ban đầu 0 verdict. Checkpoint trong `ket_qua_train/`
có SHA256 `85bbd3e66e228612d8402d9f51e911ffa6982d6be3e24a5ca4d3e91861b226a1`;
đã mở kiểm tra task detect, đúng năm class và metric validation mAP50
0.62161 / mAP50-95 0.42693, khớp kết quả Kaggle đã cung cấp.

| Câu hỏi | Y | N | D | ? |
|---|---|---|---|---|
| Có xe máy thật và box đỏ bám hợp lý vào xe? | Có | Không có xe hoặc box sai | Xe đã có box xanh khác | Chưa rõ |
| Hai box đỏ trước/sau có cùng một xe? | Cùng xe | Hai xe khác nhau | Không dùng | Chưa rõ |

Chọn **?** nếu ảnh không đủ để kết luận, ghi lý do. Không đoán tên tracker.
Không bấm phím N theo thói quen của cửa sổ GT cũ: trong audit này **N là
câu trả lời phủ định**, không phải “chưa duyệt”.

Sau khi duyệt hết, bấm **Hoàn tất → tổng hợp điểm cần xử lý**. Chương trình
ghi `audit_result.json`; những lỗi GT được xác nhận và câu trả lời `?` phải
được xác minh/sửa trong phiên GT tương ứng trước khi chốt phiên cuối. Một
tracker mắc ID switch thật không tự có nghĩa GT sai. Công cụ phân biệt các
trường hợp này trong bản tổng hợp; chưa tự thêm/xóa box hoặc gộp/tách ID.

## Mẫu và bảo toàn các lượt trước

Audit mới dùng cả **Mixed, Freeflow, Dense trong ROI**, so với hai track dump
đã lưu của Baseline và C2. Với mỗi segment/tracker, lấy ngẫu nhiên tối đa
10 run box dư và 10 run GT không được tracker khớp; lấy tối đa 10 sự kiện
ID switch. Nếu ít hơn thì lấy hết. Seed `20261012`, trộn thứ tự trước khi
duyệt; lấy cùng cỡ mẫu cho hai tracker, không chọn theo kết quả tốt/xấu.

Phải trả lời hết mẫu cố định; không dừng theo tỷ lệ lỗi tạm thời. Chỉ khi
hoàn tất mới tổng hợp lỗi. Đây là **audit mẫu phục vụ phát hiện lỗi**, không
đủ để chứng nhận toàn bộ GT không còn thiếu xe, hoặc chứng minh tracker tốt
hơn. Muốn suy rộng tỷ lệ lỗi cần thiết kế phân tích đúng đơn vị run/stratum;
không coi các frame liền kề là quan sát độc lập.

Mẫu cũ tại `data/annotations/seed_bias_audit/` vẫn giữ nguyên. Audit mới nằm
dưới `data/annotations/paper_gt_audit/roi_<hash>/`, được gitignore vì chứa ảnh,
tọa độ và quyết định cục bộ. `active.json` chọn phiên hiện tại. Không copy
verdict cũ sang mẫu mới. Tạo lại với cùng đầu vào sẽ dùng lại phiên đã có;
GT/checkpoint/cấu hình đổi thì cần tạo phiên mới, giữ phiên cũ để truy vết.

### Audit đã lưu xong

Nghĩa đã lưu **148/148** mục (119 Y, 18 D, 11 N, không có `?`).
Kết quả chỉ ra **34 mục cần xử lý GT**: 25 box dư của tracker được xác nhận
có xe thật, 3 box GT bị đánh dấu sai/trùng, 6 cặp identity được đánh dấu
hai xe khác nhau. Mục lặp được nhóm để hỗ trợ điều hướng; không tự sửa GT.

Danh sách kèm ảnh và frame MOT nằm tại
`data/annotations/paper_gt_audit/roi_241dc106947b1723/issues.md`;
receipt đầy đủ ở `audit_result.json` cùng thư mục. Receipt audit ban đầu ghi
`HUMAN_RECONCILIATION_REQUIRED`. Lượt sửa 32 nhóm đã hoàn tất ngày 13/10/2026,
đã xuất GT mới và chấm lại; xem [mốc sau sửa](GT_RECONCILED_20261013.md).
Một verdict ở frame mẫu không xác nhận toàn bộ run
hoặc tự quyết định ID để nối box mới vào track.

Khi tổng hợp, chỉ một hash code của `audit_disagreements.py` đã thay đổi do
việc pin nguồn của mẫu cũ. Đã lấy bản gốc từ commit `c5e7def`, xác minh hash
đúng với protocol và so sánh AST: ngoài khai báo/import `REVIEWS`, mọi câu
lệnh thực thi đều giữ nguyên. GT, ảnh, key và verdict của audit không đổi.
`complete_paper_gt_audit.py` ghi bằng chứng này trong receipt; nó từ chối
mọi thay đổi GT, ảnh, logic helper hoặc đầu vào khác. Không sửa hash protocol
gốc để bỏ qua kiểm tra và không yêu cầu gán nhãn lại mẫu đã hoàn tất.

Tái lập việc tổng hợp cho phiên này:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m code.scripts.complete_paper_gt_audit --source-ref c5e7def
```

## Những gì được ghi nhận trong protocol

`protocol.json` ghi đường dẫn và hash của checkpoint `ket_qua_train/best.pt`, hồ sơ train Kaggle,
cấu hình ablation hiện tại, code tracker/detector/model, GT/ROI/nhật ký review,
hai track dump dùng cho audit và ảnh nguồn được lấy mẫu. Nó đối chiếu
checkpoint ở root (nếu còn) với `ket_qua_train/best.pt`, và GT với các bản xuất đã chốt.
Công cụ từ chối hoàn tất nếu đầu vào đổi hoặc hai cửa sổ ghi đè nhau.

Protocol đề xuất cho lượt chạy mới:

- Nhóm hiện tại: **development**, ba đoạn cùng camera NKKN.
- Detector: checkpoint đã xác minh; thử input 640 và 448 riêng biệt.
- Mọi tracker nhận cùng detection cache, giữ sàn confidence 0.1 để các
  thuật toán có thể dùng detection confidence thấp.
- Cùng ROI: mask pixel trước detector, lọc detection bằng tâm đáy trước
  tracker, bỏ state active/lost khi track ra ngoài ROI.
- Cấu hình hiện tại là snapshot ứng viên. Runner còn cần kiểm tra hỗ trợ ROI
  và ghi **cấu hình đã resolve** cho từng lượt chạy; không coi config cũ tự
  trở thành config ROI hoặc kết quả của detector mới.
- Chỉ tune trên development. Khóa tham số trước khi chấm camera/video test
  độc lập; khai báo mọi thay đổi bằng phiên protocol mới.

## Khi nào bước 1 hoàn tất?

1. Hash/checkpoint/GT/ROI khớp và audit được lưu đủ.
2. Các lỗi GT hoặc mục chưa rõ đã được xử lý; nếu sửa GT, xuất phiên mới và
   lưu liên kết từ audit sang phiên GT mới. Không tự coi audit đã hoàn tất
   là các lỗi đã được sửa.
3. Nghĩa xác nhận nguồn, quyền sử dụng footage và nguồn dataset detector;
   đăng ký video/camera độc lập hoặc giới hạn rõ nghiên cứu ở development.
4. Protocol cuối xác định tham số/runner thực tế, danh sách ablation và
   footage test trước khi chạy test.

**Trạng thái hiện tại:** đã ghi nhận đầu vào development, hoàn tất 148 mục
audit và lượt xử lý 32 nhóm; đã xuất/chấm GT mới. Nguồn/quyền sử dụng,
runner/cấu hình ROI cho checkpoint mới và protocol test độc lập còn chờ.
Không đánh dấu paper-ready hoặc test-independent.

Phạm vi paper giữ theo [PAPER_SCOPE.md](PAPER_SCOPE.md), không có
`minor/severe`. Mốc GT trước audit ở [GT_COMPLETION_20261012.md](GT_COMPLETION_20261012.md).
