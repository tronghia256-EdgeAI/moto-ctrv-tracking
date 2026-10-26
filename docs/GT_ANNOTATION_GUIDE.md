# Hướng dẫn label Ground Truth (GT) bằng CVAT cloud

**Cho:** Nghĩa + Lành · **Mục tiêu trước mắt: CHỈ đoạn `dense`** (~1.5–2h tổng)
· **Deliverable:** `data/annotations/NKKN_gt/dense/gt/gt.txt` (gitignored — KHÔNG commit)

## Trạng thái: `dense` ĐÃ XONG (28/7/2026)

Kết quả trong `experiments/gt_ablation/evidence.md`. Việc label đã trả tiền
ngay: GT phát hiện 3 thứ mà proxy không thể thấy — C1 đang **có hại** trên
dense (FP 1388 → 6255), nhánh MAPPO của C3 **không lọc gì** (τ=0.40 nằm dưới
score thấp nhất 0.435), và gating nếu không có NSA thì **làm IDsw tệ hơn**.
C2 đầy đủ: IDF1 xe máy 0.844 → 0.933, IDsw −85%.

## Việc tiếp theo: `mixed` — ĐÃ CHUẨN BỊ XONG, sẵn sàng label

**Luồng review local mới (07/10/2026):** mở `review_gt_mixed.cmd`. Có ảnh so sánh,
nối/tách ID, sửa/thêm/xóa box, đổi class, lưu tự động và audit đủ 600 frame.
Không cần upload sang CVAT để dùng luồng này. Xem
[hướng dẫn review trên máy](GT_REVIEW_WORKFLOW.md).
Phần CVAT bên dưới là lựa chọn khác; xuất GT CVAT tạo dataset mới, không ghi đè
snapshot nguồn của app local đang review.

```
data/annotations/prelabels/mixed_images.zip        ← upload làm task data
data/annotations/prelabels/mixed_annotations.zip   ← import đè lên
data/annotations/prelabels/mixed/review_queue.md   ← 131 chỗ, ~48 phút
```

**Pre-labels của `mixed` seed từ baseline, KHÔNG phải từ ctrv_gate_nsa** —
khác `dense`, và đây là chủ ý. Hai lý do, một công đôi việc:

1. **Test được C1.** `dense` không bao giờ tụt xuống dưới ngưỡng mật độ
   (0/600 frame) nên C1 giữ nguyên một trạng thái suốt — không thể chứng
   minh "thích ứng" gì cả. `mixed` chuyển trạng thái 264/336 frame, mật độ
   trung bình 30.0 đúng ngay ngưỡng 30. Đây là đoạn duy nhất trả lời được
   câu hỏi C1 có tác dụng không.
2. **Đổi nguồn pre-labels để kiểm tra độ nhạy với seed.** GT của `dense` seed
   từ method được đánh giá; mixed seed từ baseline. Đổi seed không tự loại bỏ
   annotation bias: cả hai vẫn cần human review, gồm vật thể detector bỏ sót.
   Các sensitivity cũ là các kịch bản thử, không phải giới hạn đảm bảo cho bias.

Vì seed từ baseline (yếu hơn) nên sẽ có nhiều chỗ phải sửa hơn `dense` —
67 MERGE thay vì 37. Đó là bình thường, không phải lỗi.

## Tại sao làm việc này

Mọi số liệu proxy (re-id events −60%) là **không có GT** — đủ để phát triển,
không đủ để đưa vào paper. Có GT là có MOTA/IDF1 thật cho bảng ablation
chính, và là thước đo duy nhất không bị Goodhart (gate sweep đã chứng minh
proxy có thể bị game).

Mô hình làm việc: **tracker đã label sẵn ~95% đúng** (pre-labels), người
chỉ **sửa chỗ sai** — nhanh hơn label từ đầu 5–10×.

## Phạm vi: dense trước, phần còn lại tuỳ sức

Đoạn `dense` (frames 10700–11300, ~47 xe/frame) là nơi track fragmentation
xảy ra — đúng chỗ C2 claim cải thiện, nên **1 đoạn này đủ cho bảng MOTA
sơ bộ của paper**. `mixed` và `freeflow` (đã chuẩn bị sẵn y hệt) chỉ làm
thêm sau nếu còn thời gian — chất lượng 1 đoạn hơn 3 đoạn vội.

## Dữ liệu đã chuẩn bị sẵn (sinh bởi `python -m code.scripts.export_prelabels`)

```
data/annotations/prelabels/
├── dense_images.zip          ← 600 frames, upload làm task data
├── dense_annotations.zip     ← pre-labels MOT 1.1, import đè lên task
└── (mixed_*, freeflow_* — để dành làm sau)
```

Lưu ý riêng tư: chỉ upload frames giao thông thường này lên CVAT cloud
(workspace riêng, không public). **Dataset accident và model weights
tuyệt đối không upload đi đâu.**

## Bước 1 — Tài khoản (~3 phút, 1 lần)

1. Mở `https://app.cvat.ai` → **Sign up** (miễn phí).
2. (Tuỳ chọn) tạo Organization và mời người kia để 2 người cùng thấy task.

## Bước 2 — Tạo Project + Labels (1 lần)

1. **Projects → +**, tên `NKKN GT`.
2. Thêm đúng 5 labels, **tên phải khớp từng ký tự** (thứ tự không quan
   trọng, tên thì có): `car`, `motorcycle`, `bus`, `truck`, `accident`

## Bước 3 — Tạo Task + nạp pre-labels (~5 phút)

1. **Tasks → +**, tên `dense`, Project = `NKKN GT`.
2. Phần **Select files**: kéo thả `dense_images.zip` → **Submit & Open**.
3. Trong trang task: **Actions → Upload annotations → "MOT 1.1"** →
   chọn `dense_annotations.zip`.
4. Mở job → thấy boxes phủ sẵn lên xe là thành công.

## Bước 4 — Sửa pre-labels (phần việc chính)

**Đừng tua playback từ đầu tới cuối.** Có sẵn danh sách chỗ đáng ngờ:

```bash
python -m code.scripts.gt_review_queue --segment dense
# -> data/annotations/prelabels/dense/review_queue.md   (checklist)
#    data/annotations/prelabels/dense/review_queue.csv  (bản máy đọc)
```

File `review_queue.md` liệt kê **137 chỗ cần nhìn** thay vì 600 frame,
mỗi dòng có sẵn số frame để gõ thẳng vào ô frame của CVAT và thao tác
cần làm. Ước lượng ~57 phút. Mở file đó, làm từ trên xuống, tick `[x]`
khi xong — **mục MERGE làm trước**, đó đúng là thứ C2 claim cải thiện.

Ba điều cần nhớ về danh sách này:

- Nó **không phải GT** và không quyết định đúng/sai. Nó chỉ nói "chỗ này
  đáng ngờ" dựa trên hình dạng của track (chết giữa đường, box nhảy,
  hai ID chồng nhau…). Nhìn ảnh rồi tự quyết định.
- Bỏ qua một dòng **không làm hỏng GT**; sửa bừa một dòng thì có. Thấy
  pre-label đã đúng thì tick và đi tiếp.
- Nó không bắt được **xe bị miss hoàn toàn** (không có box nào từ đầu) —
  detector không thấy thì danh sách cũng không thấy. Nên sau khi làm hết
  checklist, tua nhanh 1 lượt playback để bắt loại này.

Số frame trong file là **số của CVAT** (đếm từ 0), gõ thẳng được. Lần đầu
nên đối chiếu 1 box để chắc mapping đúng.

3 loại lỗi cần sửa, theo thứ tự quan trọng với paper:

| Lỗi | Nhận ra thế nào | Sửa thế nào trong CVAT |
|---|---|---|
| **ID switch / track vỡ** (quan trọng nhất — đây là thứ C2 claim cải thiện) | Một xe đi qua mà số ID trên box đổi giữa chừng, hoặc box biến mất rồi hiện lại với ID mới | Chọn track thứ nhất → phím `M` (merge) → chọn track thứ hai → `M` lần nữa → hai track thành một ID. Ngược lại, 2 xe bị dính chung 1 ID → dùng **Split** (icon kéo trên sidebar) tại frame đổi xe |
| **Miss (xe không có box)** | Xe rõ ràng trong hình mà không box nào bám | Vẽ box mới ở **Track mode** (không phải Shape mode) → CVAT tự interpolate các frame sau, chỉnh lại khi lệch |
| **Box lệch / sai class** | Box trùm cả 2 xe, box hụt nửa xe, xe máy bị gán car… | Kéo cạnh box; đổi class trong sidebar phải |

### Quy ước label (đọc kỹ — nhất quán quan trọng hơn "đúng tuyệt đối")

- **Xe máy = xe + người ngồi trên xe** (cả người ôm sau). Đây là quy ước
  detector đã train, giữ nguyên.
- **Che khuất (occlusion):** còn thấy ≥ ~30% xe thì vẫn giữ box, ước
  lượng phần bị che. Khuất hẳn sau xe bus/cây ≥ vài frame → để track
  đứt tại đó là đúng (đừng vẽ box đoán mò); khi xe hiện lại, **nối lại
  cùng ID cũ bằng merge** — chính những chỗ này đo khả năng re-id.
- **Vùng xa (gần vạch sang đường phía trên):** xe nhỏ hơn ~8 px hoặc
  không phân biệt nổi class bằng mắt → **bỏ qua, không label**. GT chỉ
  tính vật thể người nhìn còn xác định được.
- **Xe đậu bên lề:** có label (detector thấy chúng, bỏ ra sẽ thành FP oan).
- **accident:** clip này không có tai nạn; đừng gán class accident.

Nhớ **Save (Ctrl+S)** thường xuyên — cloud không tự lưu từng thao tác.

## Bước 5 — Export GT đã sửa

**Actions → Export task dataset → "MOT 1.1"** → tải về, giải nén thành:

```
data/annotations/NKKN_gt/dense/gt/gt.txt
```

## Bước 6 — Kiểm tra (Nghĩa chạy, ~1 phút)

```bash
# 1. GT tự chấm chính nó phải ra MOTA = IDF1 = 1.000
python -m code.scripts.import_gt \
    --gt data/annotations/NKKN_gt/dense/gt/gt.txt \
    --pred data/annotations/NKKN_gt/dense/gt/gt.txt

# 2. Chấm thử baseline với GT thật (số MOTA/IDF1 đầu tiên của paper!)
python -m code.scripts.import_gt \
    --gt data/annotations/NKKN_gt/dense/gt/gt.txt \
    --pred-tracks experiments/c2_final/tracks_baseline.json \
    --segment 10700 11300
```

Segment ↔ frame tuyệt đối (cho `--segment`):
`dense 10700 11300` · `mixed 2500 3100` · `freeflow 6400 7000`

## Nếu kẹt

- Hotkeys CVAT đầy đủ: `https://docs.cvat.ai/docs/manual/` (mục
  "Shortcuts"). Ba phím dùng nhiều nhất: `F`/`D` (tiến/lùi frame),
  `M` (merge track), `N` (vẽ box mới).
- Muốn riêng tư tuyệt đối (không dùng cloud): CVAT chạy local được qua
  Docker (`github.com/cvat-ai/cvat`, `docker compose up -d`) — quy trình
  từ Bước 2 trở đi giống hệt.
