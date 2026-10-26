# GT sau audit đã xuất và kiểm tra — 13/10/2026

Nghĩa đã hoàn tất **32/32 nhóm**: 25 quyết định EDIT, 7 KEEP có ghi chú
sau khi xem lại ngữ cảnh. Các cửa sổ sửa đã được xem đầy đủ theo nhật ký
riêng; lượt xem 600 frame của phiên trước không được dùng để xác nhận sửa.
Đây là hoàn tất lượt adjudication mẫu trên development, không chứng nhận
toàn bộ GT không còn lỗi hoặc khả năng tổng quát trên camera độc lập.

| Segment | EDIT / KEEP | Box trước → sau | ID trước → sau |
|---|---|---:|---:|
| dense | 15 / 1 | 21.651 → 21.749 | 341 → 344 |
| freeflow | 3 / 3 | 2.501 → 2.503 | 47 → 48 |
| mixed | 7 / 3 | 9.145 → 9.168 | 153 → 154 |

Tổng **33.420 box**, thêm 190 và xóa 67 so với GT trước lượt sửa.
Không cộng số ID giữa các đoạn thành số xe duy nhất trong video.

## Các bản GT phải dùng cho bước thí nghiệm tiếp theo

Các thư mục sau nằm dưới `experiments/gt_qa/mixed_reviewed/`; tên dùng UTC,
nên bắt đầu bằng 20261012 dù ngày hoàn tất ở Việt Nam là 13/10/2026.

| Segment | Thư mục | SHA256 `gt_roi.txt` |
|---|---|---|
| dense | `20261012T170037Z_74193458` | `f4338ee06a1b878c07b80e681ed00721f2a57c97b508028d0b91db81dc927a50` |
| freeflow | `20261012T170822Z_855ce785` | `89a6a7de2c4ed047b234810e3042bfd226766872cbabfd7dc0386742aa9de07a` |
| mixed | `20261012T170849Z_77c2177b` | `64e1720ee4cb5b31f776c6f3a4980dac83b1acb465bfb4f4bc652ec60611bc0f` |

Receipt tổng hợp:
`data/annotations/paper_gt_reconciliation/reconcile_90026162a7963db7/reconciled_e880ead8.json`.
`latest_release.json` cùng thư mục chọn receipt này. Giữ nguyên collection
GT cũ và 148 verdict audit gốc để truy vết; các viewer/pointer review cũ
không tự đổi sang bản sau sửa. Các lượt chạy mới phải chọn GT từ receipt
sau sửa này, không lấy nhầm [mốc trước audit](GT_COMPLETION_20261012.md).

Đã kiểm tra cả ba bộ: strict review, nhật ký xem các cửa sổ sửa, snapshot
khớp quyết định/nhật ký hiện tại, hash GT và ROI, số box, tâm đáy mọi box
trong ROI, hash cả bảy track dump và các metric hữu hạn. GT nguồn và audit
gốc vẫn giữ nguyên.

## Chấm lại track dump cũ trên GT mới

IDF1 cho motorcycle, IoU=0.5:

| Segment | Baseline | C2: CTRV + gate + NSA | C1 + C2 + C3 |
|---|---:|---:|---:|
| dense | 0.8662 | 0.9109 | 0.9004 |
| freeflow | 0.9324 | 0.9621 | 0.8941 |
| mixed | 0.9865 | 0.9495 | 0.9581 |

C2 vẫn cải thiện Dense và Freeflow nhưng giảm trên Mixed. Không mô tả là
cải thiện đồng đều. Kết quả dùng track dump cũ trong `experiments/gt_ablation/`,
lọc cả GT và predictions bằng cùng ROI. Chưa phải inference/tracking mới
với `ket_qua_train/best.pt` và mask pixel ROI. MOTA, IDSW và các class/variant
khác nằm trong `summary.md`/`review.json` của từng bản xuất.

## Điều chỉnh kiểm tra box trùng

Ở Mixed, mục audit #25 được lưu KEEP kèm ghi chú đã sửa nhưng E bị chặn.
Nhật ký cho thấy reviewer giữ box ID 1992 và xóa box trùng ID 2033 tại
MOT 4, đồng thời thêm các frame tiếp theo. Đó là một thao tác sửa duplicate
hợp lệ. Validator trước đây chỉ nhận việc sửa/xóa box đỏ được audit;
đã bổ sung nhận việc gỡ bản trùng còn lại, khi số box motorcycle khớp vùng
audit giảm từ nhiều box xuống một. Không đổi verdict KEEP hay ghi chú đã
lưu để làm đẹp thống kê; receipt giữ nguyên lịch sử thao tác thực tế.

Bước kế tiếp là chạy cache detection và các ablation bằng checkpoint mới,
cùng ROI và các GT sau sửa nêu trên. Test độc lập và xác nhận nguồn/quyền
sử dụng vẫn là phần chưa chốt. Phạm vi theo [PAPER_SCOPE.md](PAPER_SCOPE.md).
