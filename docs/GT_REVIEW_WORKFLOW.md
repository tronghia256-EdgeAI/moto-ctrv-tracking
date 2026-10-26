# Review GT trên máy

**Cập nhật 13/10/2026:** 32 nhóm sửa sau audit đã hoàn tất, ba bản GT mới đã
xuất/chấm/kiểm tra. Dùng [receipt GT sau sửa](GT_RECONCILED_20261013.md) cho
thí nghiệm tiếp theo; các mốc dưới đây giữ làm lịch sử.

**Thí nghiệm sau GT:** [runner tự động](PAPER_EXPERIMENT_WORKFLOW.md) tạo
cache bằng `ket_qua_train/best.pt`, chạy ablation/đối chứng trong ROI, chấm
cả ba đoạn và đo latency ở cùng độ phân giải. GT hiện tại vẫn là development.

**Cập nhật 12/10/2026:** Nghĩa đã hoàn tất ba phiên Mixed, Freeflow và Dense
trong ROI (333 mục, 1.800 frame). Xem [receipt và bước tiếp theo](GT_COMPLETION_20261012.md).

**Kiểm tra chéo trước thí nghiệm mới:** mở `review_gt_audit.cmd` để duyệt
ảnh bất đồng từng mục và tự lưu. Xem [bước 1: protocol và audit](PAPER_STEP1_PROTOCOL.md).

**Sau khi audit đã xong:** mở `review_gt_reconcile.cmd` để sửa 32 nhóm có
vấn đề trên bản sao GT. Xem [hướng dẫn sửa GT sau audit](GT_AUDIT_RECONCILIATION.md).

**Dataset Accident / Vehicle:** mở `review_accident_dataset.cmd` để duyệt cụm
ảnh gần giống, sửa box và xuất dataset detector. Xem
[hướng dẫn làm sạch dataset](ACCIDENT_DATASET_REVIEW_WORKFLOW.md).

**Train thử detector:** mở `train_detector_trial.cmd` để kiểm tra YOLO11n
trên CPU. Xem [workflow train thử](DETECTOR_TRIAL_WORKFLOW.md); kết quả này
chỉ dùng phát triển, chưa phải evaluation cho paper.

**Train trên Kaggle:** upload ZIP đã chuẩn bị và import
`notebooks/CAAP_YOLO11_Kaggle.ipynb`. Xem
[hướng dẫn Kaggle](KAGGLE_DETECTOR_TRAINING.md).

## Mixed: nối/tách ID và sửa nhãn trực tiếp

### ROI cho vùng GT quan sát được

Mở **`review_gt_roi.cmd`**, hoặc nút **Điều chỉnh ROI** trong phiên GT mixed mới.
Click các đỉnh theo thứ tự vòng quanh vùng cần chấm, kéo đỉnh để chỉnh. Chọn frame
MOT khác để kiểm tra cùng polygon trên đoạn video. **Vẽ lại**, **Bỏ đỉnh cuối**,
**Toàn khung** giúp chỉnh nhanh. Điền lý do phạm vi, rồi bấm **Lưu ROI**.
Vùng tối nằm ngoài ROI; box xanh nằm trong theo quy tắc **tâm đáy box nằm trong
polygon**, bao gồm biên. Polygon tự cắt hoặc thiếu đỉnh sẽ không được lưu.

ROI lưu riêng tại `experiments/gt_qa/mixed_review/roi.json`, có ràng buộc manifest,
kích thước frame, tác giả, lý do và lịch sử thay đổi; không ghi đè GT hoặc các
quyết định trước đó. File được gitignore. Muốn trở lại chấm toàn khung, bỏ tick
**Áp dụng ROI khi xuất/chấm** rồi lưu.

Khi hoàn tất, giữ `gt_reviewed.txt` đầy đủ và xuất thêm `gt_roi.txt` nếu ROI bật.
Chấm tất cả variants bằng cùng ROI trên **cả GT và dự đoán**; giữ tọa độ box và
ID gốc, không cắt ảnh/box. Báo cáo lưu polygon, rule, hash và số box nằm ngoài.
Editor V tô tối ngoài ROI, giúp tập trung xem vùng có thể gán nhãn.

Theo yêu cầu mới của Nghĩa, GT mixed dùng **hàng chờ chỉ trong ROI**. Mở
`review_gt_mixed.cmd` để vào phiên ROI đang active. Bản toàn khung cũ giữ nguyên;
source của phiên mới đã kế thừa các sửa box/ID của bạn rồi lọc toàn bộ 600 frame.
Các đề xuất xa nằm ngoài ROI không còn trong queue mới. Chỉ KEEP cũ với case và
nhãn trong ROI không thay đổi được kế thừa; các đề xuất mới vẫn chờ người duyệt.
Audit vẫn đi qua thời gian của cả đoạn nhưng chỉ kiểm tra đối tượng trong ROI.

Tạo một phiên ROI từ bản full đang có (ROI phải được lưu và bật):

```powershell
.\.venv\Scripts\python.exe -X utf8 -m code.scripts.gt_roi_scope
```

Phiên mới nằm dưới `mixed_review/roi_reviews/`, kèm `parent_snapshot/` và
`scope_provenance.json`; launcher chọn bằng `roi_active.json`. ROI của phiên
đã lọc được đóng băng để tránh đổi phạm vi mà queue vẫn cũ. Muốn đổi vùng,
chỉnh ROI ở bản full và tạo phiên mới; giữ cả phiên cũ để đối chiếu.

Đã áp dụng cùng ROI cho ba segment **cùng camera NKKN**:

| Segment | Box trong ROI khi tạo phiên | Hàng chờ mới | Mở |
|---|---:|---:|---|
| mixed | 9.095 / 15.234 | 79 (23 KEEP kế thừa) | `review_gt_mixed.cmd` |
| dense | 21.719 / 27.513 | 200 | `review_gt_roi_dense.cmd` |
| freeflow | 2.464 / 7.249 | 54 | `review_gt_roi_freeflow.cmd` |

Dense kế thừa GT đã tách ID từ lượt review hoàn tất trước đó; hàng chờ ROI mới
là audit rộng hơn về box/class/ID nên không có nghĩa kết quả tách cũ bị xóa.
Các vùng/camera khác cần ROI riêng, không dùng polygon NKKN chỉ vì ảnh cùng kích thước.
Để chuẩn bị lại collection camera sau khi chốt nguồn:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m code.scripts.gt_roi_scope --all-camera-segments
```

Runtime ROI: dùng `configs/mixed_roi.yaml` cho đúng camera NKKN/800×450.
YOLO nhận ảnh đã che pixel ngoài ROI; detection ngoài ROI bị loại trước tracker,
track có tâm đáy ra ngoài bị loại khỏi cả active/lost state, C1 đo độ sáng chỉ
trong ROI. CNN vẫn chạy trên tensor khung đầy đủ; không mặc định coi ROI làm
latency giảm theo tỷ lệ diện tích. Replay detection cũng được lọc trước tracking.
Mọi ablation cần cùng ROI và checkpoint; chạy lại dump cho protocol runtime mới,
không coi các track full-frame cũ là đã chạy detector/tracker theo ROI.

Chốt ROI dựa trên vùng quan sát/gán nhãn được trước khi xem metric, dùng cùng
ROI cho mọi ablation và báo rõ trong paper. ROI không chứng minh tracking tốt
ở vùng xa hoặc làm evaluation trở thành test độc lập.

Nếu app GT đã mở từ trước khi cập nhật code, đóng và mở lại `review_gt_mixed.cmd`
trước khi tiếp tục để vào queue ROI mới. Không cần xóa tiến độ review cũ.

Mở **`review_gt_mixed.cmd`** tại thư mục gốc repo, hoặc:

```powershell
.venv\Scripts\python.exe -X utf8 -m code.scripts.gt_mixed_app
```

Tên người review mặc định là **Nghĩa**. Công cụ chuẩn bị và mở lại dữ liệu tại
`experiments/gt_qa/mixed_review/`; luôn giữ lại quyết định và chỉnh sửa đã lưu.
Không thay đổi kết quả dense đã review trước đó.

Bản mixed có **181 mục** từ pre-labels baseline:
13 điểm tách ID, 67 cặp nối ID, 15 cặp box trùng, 42 track ngắn, 7 track cần
xác nhận class, 7 điểm có thể thiếu box và 30 đoạn audit bao phủ đủ 600 frame.
Danh sách được tính lại từ dữ liệu; queue 131 mục cũ chưa bao gồm các điểm tách
ID và lượt kiểm tra toàn đoạn này.

### Duyệt nhanh

| Phím | Làm gì |
|---|---|
| **A** | Chấp nhận đề xuất đang ghi trên nút: tách, nối, gộp box trùng hoặc xóa track không có xe thật. |
| **K** | Giữ nguyên; đọc câu hỏi vì ý nghĩa cùng/khác xe tùy loại mục. |
| **U** | Để trống, chuyển sang mục chưa duyệt tiếp theo. |
| **V** | Mở video để xem kỹ hoặc sửa nhãn trực tiếp. |
| **E** | Xác nhận đã sửa thủ công; cần có thay đổi được lưu cho mục này. |
| **N**, **← / →** | Tìm mục chưa duyệt hoặc xem lại quyết định cũ. |

Đọc câu hỏi phía trên ảnh trước khi bấm A. **GHOST**: A nghĩa là *không có xe
thật*, xóa toàn track. **DUP**: A xác nhận hai track cùng xe xuyên suốt, gộp ID
và giữ box A ở frame có cả hai box. Nếu box của cặp đó xung đột ở một frame,
công cụ báo frame cần xem lại, giữ quyết định ở trạng thái chưa áp dụng.

**CLASSCHECK**: K nếu nhãn đúng, V nếu cần đổi class; A mở trình xem để kiểm tra,
không tự đoán class. **ORPHAN**: K nếu che khuất/đi vào khung là hợp lý, V nếu
cần thêm hoặc chỉnh box.

### Khi cần sửa box, thêm xe hoặc chọn cặp nối khác

Trong cửa sổ V:

1. **Phát/Dừng**, kéo thanh frame hoặc dùng mũi tên. Mặc định 5 FPS để dễ nhìn;
   có 2/10/25 FPS. Bật/tắt box hoặc zoom 200% và cuộn để xem xe nhỏ.
2. Click box để chọn xe, rồi **Vẽ/sửa box** và kéo khung mới quanh xe. **Xe mới +**
   tạo ID mới: chọn class rồi kéo khung. Xe máy gồm cả người ngồi trên xe.
3. **Sao chép sang frame sau** giúp vẽ tiếp cùng xe; kiểm tra và kéo sửa vị trí
   ở frame mới. Công cụ không tự nội suy qua các frame chưa được xem.
4. **Xóa box frame này** xử lý box không có vật thể. **Đổi class cả ID đang xem**
   áp dụng cho các box của identity hiện tại, gồm các đoạn đã nối; các đoạn đã
   tách thành xe khác giữ class của chúng.
5. **Tách ID từ frame này** chia đoạn đang chọn. Để nối cặp khác: chọn xe đầu,
   **Đánh dấu xe A**, chuyển tới xe tiếp theo, click box rồi **Nối A với xe đang chọn**.
   Chỉ bật tùy chọn box trùng khi xác nhận chúng cùng một xe.
6. **Hoàn tác sửa cuối** bỏ thao tác manual cuối cùng. Nếu thao tác đang được
   một quyết định nối ID sử dụng, cần đưa quyết định đó về U rồi sửa lại.

Mỗi thao tác sửa tự lưu ngay vào `edits.json`. Đóng V sẽ lưu những frame thực tế
đã hiển thị, rồi quay về thẻ review. Bấm E để xác nhận sửa thủ công, hoặc A/K
nếu quyết định theo đề xuất/giữ nguyên. Đóng app và mở lại vẫn tiếp tục được.

Nếu lỗi **multiple classes** xuất hiện khi nối, thông báo mới chỉ rõ ID sau nối,
ID nguồn, class và frame MOT ví dụ đang xung đột trên toàn identity. Nhãn của hai
box ở frame hiện tại không chứng minh mọi frame đều cùng class. Kiểm tra các
frame được nêu; chỉ sửa class khi biết nhãn sai. Chọn dropdown không tự đổi nhãn,
cần bấm **Đổi class cả ID đang xem**. Nếu đã nối/sửa thủ công trong V, quay về
bấm **E**; **A** áp dụng lại đề xuất tự động gốc, có thể khác cặp bạn vừa chọn.
Thao tác bị chặn không được lưu; các chỉnh sửa thành công trước đó vẫn giữ nguyên.

### Lượt audit và hoàn tất mixed

Mỗi mục **AUDIT** bao gồm 20 frame. V để xem cả đoạn và kiểm tra xe thiếu, box,
class, ID; sửa nếu cần. A/K chỉ được lưu sau khi mọi frame trong đoạn đã hiển
thị. Những frame đã xem ở các mục trước được tính, giúp giảm thao tác lặp.
Việc ghi nhận frame đã hiển thị không chứng minh người review đã sửa hết lỗi;
cần giữ các quy ước annotation nhất quán và kiểm tra độc lập khi làm paper.

Khi đủ quyết định và lượt xem, bấm **Hoàn tất → tạo GT và chấm lại**. Công cụ
áp dụng đúng quyết định và edit đã lưu, kiểm tra class nhất quán, ID không bị
trùng trong một frame, điểm tách không bị một chuỗi nối vô tình hủy, rồi chấm
7 ablation trên segment video `[2500, 3100)`.

Run mới nằm tại `experiments/gt_qa/mixed_reviewed/<run>/`: `gt_reviewed.txt`,
`summary.md`, `review.json` và các snapshot `source_gt.txt`, `decisions.csv`,
`edits.json`, `manifest.json`. GT source giữ nguyên; box chỉ thay đổi khi bạn
duyệt xóa box trùng/track sai hoặc sửa nhãn trực tiếp. Báo cáo ghi tên Nghĩa,
hash dữ liệu/ảnh, các thao tác, số box thêm/xóa và metrics. Không xuất run dở dang.

`mixed` dùng cho phát triển/validation vì nằm trong đoạn video từng phát triển
phương pháp. Việc đổi seed từ C2 sang baseline không tự loại bỏ annotation bias;
GT vẫn cần review độc lập và tập test cuối cần xác nhận nguồn dữ liệu train.
Các ảnh, tọa độ và journal nằm trong `.gitignore`; sao lưu cả thư mục review,
không chỉ CSV quyết định, vì journal chứa các box đã chỉnh.

CLI không mở GUI:

```powershell
# Chuẩn bị, hoặc tiếp tục bundle hiện có; không reset quyết định
.venv\Scripts\python.exe -X utf8 -m code.scripts.gt_mixed_review --prepare
# Kiểm tra dữ liệu và tiến độ
.venv\Scripts\python.exe -X utf8 -m code.scripts.gt_mixed_review --check
# Sau khi Nghĩa duyệt đủ
.venv\Scripts\python.exe -X utf8 -m code.scripts.gt_mixed_review --finalize --reviewer "Nghĩa"
```

## Dense: review các điểm tách ID

Luồng này tự động hóa việc review **105 vị trí nghi ngờ gộp nhầm ID trong dense GT**:
mở ảnh, xem đoạn nối, lưu quyết định, tiếp tục phiên cũ, áp dụng các CUT và chấm lại
các track dump. Chạy trên máy, không cần API, tài khoản hay tải ảnh lên dịch vụ ngoài.

## Mở công cụ

Double-click `review_gt.cmd` tại thư mục gốc repo, hoặc chạy PowerShell:

```powershell
.venv\Scripts\python.exe -m code.scripts.gt_review_app
```

Nếu chưa có môi trường, làm phần Quick start trong README; công cụ dùng Tkinter
(đi kèm Python trên Windows) và Pillow trong `requirements.txt`.

## Phần người review cần làm

Nhìn xe trước và sau đoạn mất dấu; dùng đoạn video khi ảnh tĩnh chưa đủ rõ.

| Phím / nút | Ý nghĩa |
|---|---|
| **C — CUT** | Hai xe khác nhau: tách ID từ frame được ghi trên ảnh. |
| **K — KEEP** | Cùng một xe: giữ ID qua đoạn mất dấu. |
| **U — Chưa rõ** | Xóa quyết định hiện tại, để trống và chuyển sang ảnh chưa review tiếp theo. |
| **V — Xem đoạn nối** | Mở các frame trước, trong và sau đoạn nối; phát/dừng hoặc kéo thanh frame. |
| **N — Chưa review** | Tới ảnh chưa có quyết định tiếp theo. |
| **← / →** | Xem lại ảnh trước/sau; có thể đổi quyết định. |
| **Ảnh 100% / Vừa cửa sổ** | Đổi kích thước hiển thị; cuộn khi cần xem chi tiết. |
| **Ctrl+S** | Lưu ghi chú ngay. |

C/K/U/V/N và mũi tên chỉ hoạt động khi không nhập ghi chú hoặc tên người review.
Quyết định lưu ngay vào `experiments/gt_qa/split_candidates/decisions.csv`;
ghi chú tự lưu sau khi ngừng gõ, khi chuyển ảnh và khi đóng cửa sổ.
Mở lại sẽ tiếp tục từ ảnh chưa có quyết định đầu tiên. Trường hợp chưa rõ phải
được xem lại hoặc nhờ một người có chuyên môn adjudicate trước khi hoàn tất.

Khung xanh trong trình xem đoạn nối là box của **ID trong GT gốc**, để định vị xe;
việc ID gốc có mặt trước/sau không chứng minh đó là cùng một xe. Frame MOT bắt đầu
từ 1, frame CVAT từ 0; frame video là chỉ số tuyệt đối từ 0. Ảnh được đọc từ
`data/annotations/prelabels/dense/img1`, phát khoảng 25 FPS. Có thể chọn nguồn ảnh
khác với `--frames-dir`; phải dùng đúng ảnh của segment đang review.

Nếu sửa CSV trong IDE lúc app đang mở, app từ chối lưu đè. Bấm **Tải lại CSV**
để nhận thay đổi bên ngoài. Không chạy đồng thời hai cửa sổ review cùng một CSV.

## Hoàn tất

Khi mọi ảnh có CUT hoặc KEEP, nhập tên người review và bấm **Hoàn tất**. Công cụ:

1. Kiểm tra các rank, ảnh, ID, điểm tách và segment khớp dữ liệu nguồn.
2. Tạo GT mới và kiểm tra toàn bộ box, class, visibility và số dòng giữ nguyên.
3. Chấm tất cả `tracks_*.json` trong `experiments/gt_ablation` trên segment
   `[10700, 11300)`, dùng metric hiện có trong repo với IoU 0.5.
4. Lưu một thư mục kết quả mới tại `experiments/gt_qa/reviewed/<run>/`.

Mỗi run gồm `gt_reviewed.txt`, bản chụp `decisions.csv`, `review.json` (tên người
review, thời gian UTC, hash các đầu vào, thông số và metrics) và `summary.md`.
GT gốc giữ nguyên. Không xuất run dở dang nếu bước kiểm tra/chấm điểm thất bại.
Chấm toàn bộ ablation có thể mất vài phút; giao diện hiển thị variant đang chạy.
Các run nằm trong `.gitignore` vì chứa tọa độ GT; `decisions.csv` nguồn vẫn được
version control. Lưu bản sao các run để làm hồ sơ thực nghiệm của paper.

Review này chỉ xử lý các vị trí **tách ID** đã được heuristic tìm thấy. Nó chưa
kiểm tra box thiếu, sai class, ID cần gộp, các lỗi ngoài danh sách, hay chất lượng
GT ở `mixed` và `freeflow`. Không dùng bảng điểm để quyết định CUT/KEEP hoặc coi
việc hoàn tất danh sách là chứng nhận GT độc lập cho toàn bộ paper.

## Chạy không có giao diện

Kiểm tra dữ liệu, không sửa file:

```powershell
.venv\Scripts\python.exe -m code.scripts.gt_review --check
```

Áp dụng và chấm lại sau khi CSV đã được người review điền đầy đủ:

```powershell
.venv\Scripts\python.exe -m code.scripts.gt_review --finalize --reviewer "Tên người review"
```

Tham số: `--decisions`, `--candidates`, `--gt`, `--tracks-dir`, `--segment START END`;
CLI có `--out-dir` (phải là thư mục chưa tồn tại) và `--skip-evaluation` nếu cần
xuất riêng GT, với trạng thái chưa chấm điểm ghi rõ trong báo cáo.
