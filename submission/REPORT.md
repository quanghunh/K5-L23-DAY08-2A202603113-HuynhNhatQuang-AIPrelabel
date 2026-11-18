# Báo cáo Day 08 — 2A202603113 - Huỳnh Nhật Quang

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26m.pt`) | 0.3 (mặc định, không chọn riêng) | Detector 6 lớp COCO, chạy nhanh trên Colab T4 (r01: 1.68 s/ảnh, r02: 0.35 s/ảnh). Không lớp nào đạt recall 90% ở ngưỡng ≤ 0.5 trong `threshold_sweep.md` (person tối đa 89%, 28 vật; bicycle 86%, 7 vật; motorbike 79%, 19 vật; car 86%, 7 vật); bus chỉ 1 vật, truck 0 vật trong gold (< 5) nên không được chọn ngưỡng riêng |
| three_wheeler | YOLOE-26s-seg (`yoloe-26s-seg.pt`), prompt: `tuk-tuk`, `cyclo`, `cargo tricycle` | 0.3 (mặc định) | YOLO26 không có lớp này (r01: xe ba bánh ở img_011 bị gọi `truck` 0.77; r04: ở img_013 bị gọi `car` 0.42). YOLOE gõ chữ ra được lớp này (r03: 8 box trên 8 ảnh pilot, 0.19 s/ảnh). Recall tối đa trong sweep 88% (8 vật), không đạt 90% |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` (không dùng `--thr` vì sweep không chọn được ngưỡng riêng cho lớp nào; YOLOE đứng trước để `three_wheeler` thắng `car`/`truck` khi box chồng nhau)
- **Phiên bản SAM bạn chọn:** SAM 2.1 base (`sam2.1_b.pt`). Có sẵn trong bộ `core`, không cần xin quyền như SAM 3; chạy được trên Colab T4 với 2.08 s/ảnh (r04). Lý do thêm của bạn: <<TỰ VIẾT: chất lượng mask bạn thấy ở Part 5>>
- Lỗi nào **không sửa được bằng đổi ngưỡng**? Tuk-tuk bị gọi sai lớp: `truck` 0.77 ở img_011 (r01) và `car` 0.42 ở img_013 (r04). Điểm 0.77 vẫn qua ngưỡng 0.5 (r02), nên ngưỡng không đổi được nhãn lớp. Cách xử lý: dùng YOLOE cho `three_wheeler` và ưu tiên YOLOE khi gộp; phần còn lại sửa tay khi review (bảng mục 2, three_wheeler: 6 edited, 7 deleted, 7 added).
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): <<TỰ ĐIỀN: ảnh bạn nhớ khi review>>
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan; bus 1 vật và truck 0 vật nên không có ngưỡng riêng; các lớp còn lại 7–28 vật nên ngưỡng chỉ là ước lượng thô.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 469 | 92.1% | 509 box máy |
| edited | 20 | 3.9% | 509 box máy |
| deleted | 20 | 3.9% | 509 box máy |
| added | 61 | 11.1% | 550 đối tượng sau duyệt |

(Đã loại 17 đối tượng quanh các bẫy khỏi bảng này; có 3 trường hợp đổi lớp, nên số vật máy sót thật ≈ 61 − 3 = 58.)

Theo lớp (accepted / edited / deleted / added): person 259/5/6/15 · bicycle 5/1/3/4 · motorbike 131/6/3/19 · car 53/2/1/15 · bus 3/0/0/0 · truck 7/0/0/1 · three_wheeler 11/6/7/7.

- Lớp nào máy tệ nhất? Theo tỉ lệ, `three_wheeler`: trong 24 box máy chỉ 11 được giữ nguyên (45.8%), 13 bị sửa hoặc xóa, và 7 vật phải vẽ thêm. Theo số vật sót, `motorbike` (19 added), rồi `person` và `car` (15 mỗi lớp). Vì sao (dẫn ảnh): <<TỰ VIẾT: ảnh cụ thể bạn nhớ>>
- So với gold: precision / recall của **máy** 68.2% (45/66; KTC 95%: 56–78%) / 64.3% (45/70; KTC 95%: 53–74%) (mẫu số: 70 đối tượng gold trong 4 ảnh). Số đo lạc quan vì ngưỡng chọn trên chính gold.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 9/12 = 75%. Theo loại: wrong_class 3/3, missing 2/3, loose 1/3, extra 1/3. Bẫy lọt: img_023 (loose), img_025 (extra), img_005 (missing), img_014 (loose), img_006 (extra). Vì sao lọt (bạn đã bỏ bước nào của checklist)? Không nhìn kỹ được các vật sát nhau.
- Accepted 92.1% mà bắt bẫy, trong khi so với gold máy chỉ đạt recall 68.2%. Điều này nói gì về cách bạn duyệt? Còn nhiều bỏ xót và chưa phân biệt rõ các vật gần giống.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 20 phút / 4 ảnh = 5.0 phút/ảnh; duyệt có pre-label 50 phút / 24 ảnh = khoảng 2 phút/ảnh.
- Kết luận: Máy có đáng dùng nhưng không đáng tin hoàn toàn, dựa trên accepted 92.1%, recall 64.3% so với gold, bắt bẫy 9/12 và thời gian trên.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Quét trước, nhìn nhãn sau: ẩn nhãn máy, tự đếm vật theo lớp trước khi hiện nhãn, làm ở mọi ảnh. Kiểm tra: ghi số đếm của bạn vào một cột trong log rồi đối chiếu với số added.
  2. Soi riêng lớp dễ sai (three_wheeler, xe ở xa) mỗi ảnh một lượt. Kiểm tra: số box three_wheeler bị sửa hoặc xóa ở lần duyệt sau so với 14/24 hiện tại.
  3.Dùng gold làm cân chuẩn: chọn vài ảnh có nhãn gán tay để tự so tỉ lệ sót của mình với của máy. Kiểm tra: recall của máy trên gold.

## 5. Mask SAM cho 3 vật (Part 5)

Phiên bản SAM: SAM 2.1 base (`sam2.1_b.pt`), click mô phỏng Interactor bằng `scripts/sam2_points.py`, chạy trên Colab T4.

| Ảnh (vật) | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 (tuk-tuk, lần A) | sam2.1_b.pt | 1 (1 dương) | Không gồm xe; chỉ có huy hiệu cam trên thân xe | Một click cho mask của bộ phận, không phải cả xe | `tuktuk_img013_A.jpg` |
| img_013 (tuk-tuk, lần B) | sam2.1_b.pt | 3 (3 dương) | Gồm cả xe, **người lái** (áo, chân) và một phần người ngồi bàn phía sau | Mask ăn cả người, trái guideline (người là `person` riêng); gồm cả kính chắn gió | `tuktuk_img013_B.jpg` |
| img_013 (tuk-tuk, lần C) | sam2.1_b.pt | 5 (3 dương + 2 âm: người lái, người ngồi sau) | Người lái vẫn lẫn một phần; người áo sọc đỏ ở sau vẫn còn bị phủ | Click âm loại không sạch: mask bị rỗ, nhiều đốm nhỏ ở vùng áo người lái | `tuktuk_img013_C.jpg` |
| img_003 (xe ba gác chở rau củ) | sam2.1_b.pt | 5 (3 dương + 2 âm: bé trai, người lái) | Gồm cả hàng hoá (rau củ) trong thùng; bé trai gần như bị loại; người lái còn bị phủ một phần | Hàng trong thùng bị gộp vào mask xe; click âm trên người lái chưa loại hết | `bagac_img003.jpg` |
| img_017 (tuk-tuk trái, bị che) | sam2.1_b.pt | 5 (3 dương + 2 âm: cô gái và người đàn ông đi bộ phía trước) | Không gồm hai người đi bộ; chỉ phủ phần xe nhìn thấy | Mask dừng ở chỗ giao với tuk-tuk giữa, không chồng sang xe bên cạnh. Mask tốt nhất trong 3 vật | `tuktuk_img017.jpg` |

Số polygon của mỗi mask: <<TỰ ĐIỀN: lấy từ dòng log in ra khi chạy `sam2_points.py` trong Colab; bỏ trống nếu không còn log>>

- Đối chiếu guideline mục 4.2: **giúp:** quy tắc "polygon bám biên phần nhìn thấy" và "hai vật chạm nhau không chồng lên nhau" khớp với kết quả ở img_017 (xe bị che, mask dừng đúng ở chỗ giao). **Thiếu:** (1) guideline chưa nói cần làm gì khi click âm không loại sạch người, làm mask bị rỗ nhiều đốm nhỏ (img_013 lần C); (2) chưa chốt hàng hoá trong thùng có tính vào mask xe không (img_003, câu hỏi mở số 3) và kính chắn gió có thuộc mask không (img_013 B, C, câu hỏi mở số 5); (3) chưa chốt mask của xe có gồm người lái ngồi trên xe không, trong khi mục 2 chỉ nói box người là `person` riêng.
- Số click ảnh hưởng mask: cùng một vật ở img_013, 1 click ra bộ phận (huy hiệu), 3 click ra cả xe kèm người, thêm 2 click âm vẫn chưa ra mask sạch.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: <<TỰ ĐIỀN: bỏ trống hoặc ghi "không làm" nếu bạn không thử; xem `docs/CVAT_AI.md` mục 6>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** dùng Claude (trợ lý AI) để hướng dẫn từng bước thao tác, đọc tài liệu lab, soạn khung `model_log.csv` và điền số liệu vào `REPORT.md` mục 1–4 từ các file output của script, và ghi nhận xét mask ở mục 5 sau khi xem ảnh trong `submission/masks/` (`runs.csv`, `threshold_sweep.md`, `review_stats.md`, `time_log.csv`). Không dùng AI để gán nhãn gold hay duyệt ảnh. Kết luận và quy tắc trong báo cáo do tôi tự viết. Cách kiểm tra: <<TỰ ĐIỀN: ví dụ đối chiếu từng số với file trong qc/>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<TỰ ĐIỀN: không / có, như sau: ...>>
- Máy / GPU / CVAT dùng: Linux, RAM dưới 8 GB, không dùng GPU trên máy; model chạy trên Google Colab T4; CVAT v2.81.0 (Docker). Vướng mắc cài đặt: <<TỰ ĐIỀN nếu có>>
- Dùng cache `data/cache/` thay vì tự chạy? Không, tự chạy `prelabel.py` trên Colab T4.
- Lỗi / bug script đã gặp (nếu có): <<TỰ ĐIỀN>>
- Một điều bạn làm khác lần sau: <<TỰ VIẾT>>
