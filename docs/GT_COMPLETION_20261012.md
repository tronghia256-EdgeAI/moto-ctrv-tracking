# GT trong ROI đã được Nghĩa duyệt — 12/10/2026

Cả ba phiên đã có quyết định cho toàn bộ hàng chờ và nhật ký hiển thị đủ
600/600 frame mỗi phiên. Kiểm tra nguồn ảnh, tính nhất quán class/ID, các
chỉnh sửa thủ công và điều kiện xuất nghiêm ngặt đều đạt. Đây là xác nhận
hoàn tất workflow; không chứng nhận không còn lỗi nhãn hoặc độ đồng thuận
giữa nhiều người gán nhãn.

| Segment | Mục đã duyệt | APPLY / KEEP / EDIT | Box sau review | ID sau review |
|---|---:|---|---:|---:|
| mixed | 79/79 | 3 / 69 / 7 | 9.145 | 153 |
| freeflow | 54/54 | 9 / 32 / 13 | 2.501 | 47 |
| dense | 200/200 | 27 / 154 / 19 | 21.651 | 341 |

Tổng cộng 333 mục, 1.800 frame và 33.297 box. Không cộng số ID của các
segment thành số xe duy nhất trong toàn video.

Các phiên vẫn giữ tại `experiments/gt_qa/mixed_review/roi_reviews/`:

- mixed: `roi_57b24ac4c01d`, đoạn video `[2500, 3100)`.
- freeflow: `roi_f5578e885dec`, đoạn video `[6400, 7000)`.
- dense: `roi_6f8320de9fb3`, đoạn video `[10700, 11300)`.

Phạm vi là ROI do Nghĩa chọn, đóng băng theo manifest của từng phiên;
box được giữ khi tâm đáy nằm trong polygon, bao gồm biên. Bản nguồn,
các quyết định và lịch sử chỉnh sửa được giữ lại. Dữ liệu GT/ảnh/video
và các artifact xuất bản vẫn nằm trong các đường dẫn được gitignore.

## Cách sử dụng kết quả chấm lại

Các bản xuất hoàn tất nằm trong `experiments/gt_qa/mixed_reviewed/`;
mỗi bản có `gt_roi.txt`, `review.json`, `summary.md` và snapshot của
quyết định, chỉnh sửa, manifest, ROI và nguồn. Receipt tổng hợp nằm tại
`data/processed/gt_diagnostics/completed_roi_collection.json`.

| Segment | Thư mục bản xuất | SHA256 của `gt_roi.txt` |
|---|---|---|
| mixed | `20261012T094827Z_7779f93a` | `35c38f38b0786d05fd8952c7a3e0ccc1e773286d3fa05159c82345508d5fe669` |
| freeflow | `20261012T112802Z_e7771a10` | `a89ab6dbe0b24d0e01da8edc300b935480915f01201d3502b5643b971bc25442` |
| dense | `20261012T112827Z_7b620b49` | `c712eb30c3a4da2573dabc6972c82852f109745a2b040a939d65eee1f42a2741` |

Đã đối chiếu hash snapshot với phiên review hiện tại, kiểm tra số box và
mọi tâm đáy box thuộc ROI, hash GT, hash bảy track dump, metric hữu hạn và
hash engine chấm. Cả ba bản xuất đạt kiểm tra này.

Lượt chấm này dùng bảy `tracks_*.json` đã lưu trong `experiments/gt_ablation/`.
GT và predictions được lọc bằng cùng ROI, metric dùng engine hiện có
`code/evaluation/metrics.py` với IoU=0.5. Đây là chấm lại track cũ trên GT mới;
không phải chạy inference/tracking mới với `best.pt` và pixel mask ROI.

IDF1 cho motorcycle trong lượt chấm track cũ này:

| Segment | Baseline | CTRV + gate + NSA (C2) | C1 + C2 + C3 |
|---|---:|---:|---:|
| mixed | 0.9828 | 0.9449 | 0.9534 |
| freeflow | 0.9293 | 0.9651 | 0.8908 |
| dense | 0.8640 | 0.9091 | 0.8977 |

C2 cải thiện Freeflow và Dense nhưng giảm trên Mixed; không mô tả là
cải thiện đồng đều cả ba chế độ. Số đầy đủ theo class/variant và MOTA,
ID switches có trong từng `summary.md`/`review.json`.

Cả ba đoạn thuộc cùng camera/video NKKN, đã dùng trong phát triển phương pháp.
GT bắt đầu từ prelabels của tracker và đã được Nghĩa kiểm tra, nên cần nêu
provenance này; không coi chúng là test độc lập hoặc tự suy ra không còn
thiên lệch từ tracker sinh prelabels. Kết quả giữa các segment phải được
báo cáo đầy đủ, kể cả trường hợp variant không cải thiện baseline.

Bước tiếp theo là chạy lại các ablation bằng cùng checkpoint detector,
ROI và chính sách bỏ detection/track ngoài ROI; ghi hash checkpoint,
cấu hình, ROI, phiên GT và track dump. Sau đó đánh giá tracking và latency.
Test từ camera/video độc lập vẫn cần chuẩn bị riêng.

Phạm vi paper theo [PAPER_SCOPE.md](PAPER_SCOPE.md): không làm nhãn
`minor/severe`. Không dùng việc hoàn tất GT tracking để xác nhận kết quả
cảnh báo tai nạn theo sự kiện.
