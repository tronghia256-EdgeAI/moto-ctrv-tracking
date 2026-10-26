# Sửa GT sau audit đã hoàn tất

**Trạng thái 13/10/2026:** 32/32 nhóm đã được xử lý; các bản GT đã xuất và
kiểm tra. Xem [receipt và kết quả chấm lại](GT_RECONCILED_20261013.md).

Nghĩa đã duyệt 148 mục audit. Các verdict báo 34 điểm cần xử lý; hai cặp
lặp/chồng vị trí được nhóm để điều hướng thành **32 nhóm**, gồm Dense 16,
Freeflow 6 và Mixed 10. Gộp nhóm không tự chứng minh hai box là cùng xe.

Mở **`review_gt_reconcile.cmd`**. Cửa sổ này chuyển giữa ba segment, chỉ
hiển thị những điểm cần sửa. Nó dùng bản sao GT đã hoàn tất, giữ nguyên
GT cũ, 148 câu trả lời audit và các pointer review trước đó.

## Cách xử lý một mục

1. Bấm **V**. Editor mở tại frame trọng điểm, có các thao tác box/ID quen thuộc.
2. Bấm **Phát từ đầu đoạn** để xem toàn đoạn quanh lỗi; có thể chọn 25 fps.
   Lượt xem 600 frame của phiên cũ không thay thế lượt xem đoạn sửa này.
3. Sửa theo loại lỗi:
   - Xe thiếu/chưa khớp GT: box **cam** chỉ là gợi ý từ tracker, chưa phải GT.
     Chọn đúng ID trên frame trước rồi bấm **Gợi ý → xe đang chọn** nếu xe
     đã có track. Nếu là xe mới, bấm **Gợi ý → xe mới**. Kiểm tra và kéo sửa
     box; dùng cùng ID cho các frame tiếp theo. Không tự áp dụng toàn run
     chỉ từ verdict đã xác nhận một ảnh mẫu.
   - Box GT sai/trùng: chọn đúng box và chỉnh/xóa trong frame liên quan.
   - ID nối hai xe: xem vị trí chuyển xe. Nếu đúng frame đề xuất, đóng V
     rồi **A** để tách tại frame đó. Nếu cần tách ở frame khác, dùng nút
     **Tách ID từ frame này** trong V, đóng V rồi **E**.
4. **Lưu và về review**, rồi **E** xác nhận sửa thủ công. Box và nhật ký
   chỉnh sửa tự lưu; E xác nhận mục đó đã được xử lý.

Ảnh ở cửa sổ chính là bằng chứng audit **trước sửa**. V hiển thị GT hiện tại.
Công cụ kiểm tra điểm mẫu có GT khớp sau khi thêm box, box sai có chỉnh/xóa,
hoặc cặp ID đã được tách. Chỉ đổi cách viết số hoặc confidence không giải
quyết một box sai vị trí.

Nếu xem toàn đoạn cho thấy GT cũ hợp lý và verdict audit trước đó nhầm,
điền lý do cụ thể rồi bấm **Giữ GT + lý do [K]**. Quyết định audit gốc không
bị ghi đè; bản sửa ghi nhận phán quyết sau khi xem thêm ngữ cảnh. Nút
**Đặt lại mục này** xóa quyết định xác nhận, giữ các chỉnh sửa để kiểm tra lại.

## Xuất và chấm lại

Khi đủ 32 mục được xử lý hoặc có lý do giữ GT, bấm **Hoàn tất → xuất GT mới
và chấm lại**. Chương trình kiểm tra cả ba bộ trước khi xuất; những cửa sổ
còn thiếu lượt xem, sửa chưa xử lý đúng điểm mẫu hoặc KEEP thiếu lý do sẽ
bị chặn. Dữ liệu cũ vẫn giữ nguyên.

Các bản GT mới được lưu dưới `experiments/gt_qa/mixed_reviewed/`. Receipt
liên kết ba bản xuất nằm cạnh collection của lượt sửa, được chọn bằng
`latest_release.json`; nó không ghi đè receipt GT cũ vốn được audit ràng buộc.
Lượt chấm dùng bảy track dump cũ với cùng ROI. Thí nghiệm với checkpoint
`ket_qua_train/best.pt` và masking ROI vẫn là bước kế tiếp riêng.

Lượt sửa nằm trong các bundle `mixed_review/roi_reviews/reconcile_*`;
`data/annotations/paper_gt_reconciliation/active.json` trỏ tới collection.
Nếu GT hoặc câu trả lời audit gốc đổi, tạo phiên mới; không sửa nguồn của
phiên hiện tại để bỏ qua kiểm tra hash.

Chuẩn bị lại bằng CLI nếu cần:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m code.scripts.gt_audit_reconcile prepare
```

Mọi đoạn hiện tại vẫn thuộc **development**. Hoàn tất lượt sửa không xác
nhận toàn bộ GT không còn lỗi hoặc tạo ra test camera độc lập. Không có
nhãn `minor/severe`; phạm vi theo [PAPER_SCOPE.md](PAPER_SCOPE.md).
