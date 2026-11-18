# Báo cáo Day 08 — 2A202603113 - Huỳnh Nhật Quang

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26m.pt`) | 0.3 (mặc định, không chọn riêng) | Detector 6 lớp COCO, chạy nhanh trên Colab T4 (r01: 1.68 s/ảnh, r02: 0.35 s/ảnh). Không lớp nào đạt recall 90% ở ngưỡng ≤ 0.5 trong `threshold_sweep.md` (person tối đa 89%, 28 vật; bicycle 86%, 7 vật; motorbike 79%, 19 vật; car 86%, 7 vật); bus chỉ 1 vật, truck 0 vật trong gold (< 5) nên không được chọn ngưỡng riêng |
| three_wheeler | YOLOE-26s-seg (`yoloe-26s-seg.pt`), prompt: `tuk-tuk`, `cyclo`, `cargo tricycle` | 0.3 (mặc định) | YOLO26 không có lớp này (r01: xe ba bánh ở img_011 bị gọi `truck` 0.77; r04: ở img_013 bị gọi `car` 0.42). YOLOE gõ chữ ra được lớp này (r03: 8 box trên 8 ảnh pilot, 0.19 s/ảnh). Recall tối đa trong sweep 88% (8 vật), không đạt 90% |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` (không dùng `--thr` vì sweep không chọn được ngưỡng riêng cho lớp nào; YOLOE đứng trước để `three_wheeler` thắng `car`/`truck` khi box chồng nhau)
- **Phiên bản SAM bạn chọn:** SAM 2.1 base (`sam2.1_b.pt`). Có sẵn trong bộ `core`, không cần xin quyền như SAM 3; chạy được trên Colab T4 với 2.08 s/ảnh (r04). Lý do thêm của bạn: <<TỰ VIẾT: chất lượng mask bạn thấy ở Part 5. Xem ảnh trong `submission/masks/` và bảng mục 5>>
- Lỗi nào **không sửa được bằng đổi ngưỡng**? Tuk-tuk bị gọi sai lớp: `truck` 0.77 ở img_011 (r01) và `car` 0.42 ở img_013 (r04). Điểm 0.77 vẫn qua ngưỡng 0.5 (r02), nên ngưỡng không đổi được nhãn lớp. Cách xử lý: dùng YOLOE cho `three_wheeler` và ưu tiên YOLOE khi gộp; phần còn lại sửa tay khi review (bảng mục 2, three_wheeler: 6 edited, 7 deleted, 7 added).
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): <<TỰ ĐIỀN: mở CVAT, lọc `conf` cao nhưng bạn đã sửa hoặc xóa (box điểm cao mà sai), và box bạn vẽ thêm mà máy có điểm thấp ở ảnh đó; ghi img_0xx>>
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan; bus 1 vật và truck 0 vật nên không có ngưỡng riêng; các lớp còn lại 7–28 vật nên ngưỡng chỉ là ước lượng thô.

## 2. AI error profile (từ `qc/review_stats.md`, lần review 2 là nhãn nộp)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 461 | 90.6% | 509 box máy |
| edited | 24 | 4.7% | 509 box máy |
| deleted | 24 | 4.7% | 509 box máy |
| added | 80 | 14.2% | 565 đối tượng sau duyệt |

(Đã loại 16 đối tượng quanh các bẫy khỏi bảng này; có 5 trường hợp đổi lớp, nên số vật máy sót thật ≈ 80 − 5 = 75.)

Theo lớp (accepted / edited / deleted / added): person 256/7/7/25 · bicycle 5/1/3/4 · motorbike 129/8/3/23 · car 52/2/2/16 · bus 3/0/0/3 · truck 6/0/1/2 · three_wheeler 10/6/8/7.

Để so sánh, lần review 1 (nhãn cũ, trước khi gán lại): accepted 469/509 = 92.1%; edited 20; deleted 20; added 61/550 = 11.1%.

- Lớp nào máy tệ nhất? Theo tỉ lệ, `three_wheeler`: trong 24 box máy chỉ 10 được giữ nguyên (41.7%), 14 bị sửa hoặc xóa, và 7 vật phải vẽ thêm. Theo số vật sót, `person` (25 added), `motorbike` (23) rồi `car` (16). Vì sao (dẫn ảnh): <<TỰ VIẾT: dẫn img_0xx cụ thể. Nơi xem: CVAT, lọc theo lớp trong Task `batch-review`>>
- So với gold: precision / recall của **máy** 68.2% (45/66; KTC 95%: 56–78%) / 64.3% (45/70; KTC 95%: 53–74%) (mẫu số: 70 đối tượng gold trong 4 ảnh). Số đo lạc quan vì ngưỡng chọn trên chính gold.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy lần 2 (nhãn nộp): 9/12 = 75.0% (KTC 95%: 47–91%; ngưỡng đạt 80%). Theo loại: wrong_class 3/3, extra 3/3, missing 2/3, loose 1/3. Bẫy lọt: img_023 (loose), img_005 (missing), img_014 (loose). Vì sao lọt (bạn đã bỏ bước nào của checklist)? <<TỰ VIẾT: nhìn lại 3 ảnh trên trong CVAT, ghi bạn đã bỏ bước nào của checklist (bước 0 quét, bước 4 lỏng...)>>
- **Lưu ý phương pháp:** lần review 1 cho bắt bẫy 7/12 = 58.3%, bẫy lọt gồm img_023, img_025, img_005, img_014, img_006 (nguồn: `review_stats.md` lần đó). Lần 2 bạn gán lại sau khi đã xem kết quả lần 1, và hai bẫy mới bắt được (img_025, img_006) đều nằm trong danh sách lọt của lần 1. Vì vậy 9/12 không còn là phép đo độc lập về thiên kiến của bạn; số đo lần 1 (7/12) mới phản ánh lần duyệt "mù". <<TỰ ĐIỀN: xác nhận đúng như vậy, và lý do bạn gán lại>>
- Accepted 90.6% (461/509) lần 2 (92.1% lần 1) mà bắt bẫy lần 1 chỉ 58.3% (7/12), trong khi so với gold máy chỉ đạt recall 64.3% (45/70). Điều này nói gì về cách bạn duyệt? <<TỰ VIẾT>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 20 phút / 4 ảnh = 5.0 phút/ảnh; duyệt có pre-label lần 1 50 phút / 24 ảnh = khoản 2 phút/ảnh.
- Kết luận:  có máy đáng dùng, dựa trên accepted 90.6%, recall 68.2% so với gold, bắt bẫy (9/12 lần 1, 9/12 lần 2) và thời gian trên.
- **Ba quy tắc giảm thiên kiến của bạn** 
  1. Quét trước, nhìn nhãn sau: ẩn nhãn máy, tự đếm vật theo lớp trước khi hiện nhãn, làm ở mọi ảnh. Kiểm tra: ghi số đếm của bạn vào một cột trong log rồi đối chiếu với số added.
  2. Soi riêng lớp dễ sai (three_wheeler, xe ở xa) mỗi ảnh một lượt. Kiểm tra: số box three_wheeler bị sửa hoặc xóa ở lần duyệt sau so với 14/24 hiện tại.
  3. Dùng gold làm cân chuẩn: chọn vài ảnh có nhãn gán tay để tự so tỉ lệ sót của mình với của máy. Kiểm tra: recall của máy trên gold.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<phiên bản bạn dùng ở `--sam`>> | <<đếm số `--pos` + `--neg` trong lệnh>> | <<mở ảnh mask xem>> | <<mở ảnh mask xem>> | <<`ls submission/masks`>> |

(Điền sau Part 5: mỗi vật một dòng, tối thiểu 3 dòng.)

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<TỰ VIẾT: đọc `guideline/guideline_v1.0_label_contract.md` mục 4.2 (lỗ thủng, bóng, kính, hai vật chạm nhau) và so với mask của bạn>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<xem `docs/CVAT_AI.md` mục 6; bỏ trống nếu không làm>>

## 6. Calibration (Part 6)

- Số quyết định: <<đếm dòng trong `submission/qc/calibration_decisions.csv`, tối thiểu 3>>; quan trọng nhất (dẫn `CAL-xx`): <<TỰ VIẾT>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<tóm tắt từ file `submission/guidelines/guideline_v1.1_calibrated.md`; nguồn ca: ca `can_review` bạn ghi khi gán, 7 câu hỏi mở ở mục 6 của guideline v1.0, bẫy lọt (img_023, img_005, img_014)>>
- Nguyên nhân gốc: <<TỰ VIẾT: guideline mơ hồ / chưa nắm quy tắc / máy; xem `docs/CALIBRATION_QC.md`>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt). Nguồn: <<`submission/qc/qc_sample.csv` sau khi chạy `qc_sample.py`>>
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%). Nguồn: <<`submission/qc/qc_report_auto.md` sau `qc_report.py`; không sửa tay>>
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<từ cột `error_type` trong `submission/qc/qc_review.csv`>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<TỰ VIẾT dựa trên khoảng tin cậy trong file đó>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<TỰ VIẾT; xem `docs/CALIBRATION_QC.md`>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<TỰ VIẾT, đối chiếu ngưỡng ở `submission/taxonomy.json` mục `rules` và cây quyết định trong `docs/CALIBRATION_QC.md`>>
Nếu Rework / Hold, việc cần làm tiếp: <<TỰ VIẾT>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** dùng Claude (trợ lý AI) để hướng dẫn từng bước thao tác, đọc tài liệu lab, soạn khung `model_log.csv` và điền số liệu vào `REPORT.md` từ các file output của script (`runs.csv`, `threshold_sweep.md`, `review_stats.md`, `time_log.csv`). Không dùng AI để gán nhãn gold hay duyệt ảnh. Kết luận và quy tắc trong báo cáo do tôi tự viết. Cách kiểm tra: <<TỰ ĐIỀN: ví dụ đối chiếu từng số trong báo cáo với file trong `submission/qc/`>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<TỰ ĐIỀN: chỉ bạn biết; không / có, như sau: ...>>
- Review 2 lần: lần 1 ra bắt bẫy 7/12; sau khi xem `review_stats.md` bạn gán lại 24 ảnh, lần 2 bắt bẫy 9/12. Lý do gán lại: <<TỰ ĐIỀN>>. Số đo bẫy lần 2 không còn độc lập (xem mục 3).
- Máy / GPU / CVAT dùng: Linux, RAM dưới 8 GB, không dùng GPU trên máy; model chạy trên Google Colab T4; CVAT v2.81.0 (Docker). Vướng mắc cài đặt: <<TỰ ĐIỀN nếu có (ví dụ lỗi `ython`/venv, `cvat.zip` lỡ commit)>>
- Dùng cache `data/cache/` thay vì tự chạy? Không, tự chạy `prelabel.py` trên Colab T4.
- Lỗi / bug script đã gặp (nếu có): <<TỰ ĐIỀN: bỏ trống hoặc ghi "không" nếu không gặp>>
- Một điều bạn làm khác lần sau: <<TỰ VIẾT>>
