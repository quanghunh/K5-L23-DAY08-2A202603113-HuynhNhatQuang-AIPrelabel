# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603113 - Huỳnh Nhật Quang

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Vật ở gần, còn nhận ra từng xe: gán **từng xe**. `ignore_region` chỉ cho cụm **xa** từ 5 vật trở lên không tách được (hoặc cao < 20 px), và không được phủ lên vật ở gần | img_026 | CAL-02 |
| 3 | Hàng nhô ra: box và mask xe có gồm hàng không? | Hàng nằm gọn trong thùng / gắn cố định: **thuộc xe** (box và mask). Hàng nhô ra quá ~20% chiều dài xe: **không** tính vào box, cũng không vào mask | img_003, img_017 | CAL-03 |
| 5 | Kính chắn gió / cửa sổ xe thuộc mask xe không? | **Có**, kính thuộc mask xe. Người nhìn thấy qua kính hoặc khung hở có polygon riêng, vùng chồng thuộc người (vật phía trước) | img_013 | CAL-04 |
| 6 | Box chồng nhau nhiều: gộp hay giữ? | **Giữ** từng box (người ngồi trên xe vẫn là hai box chồng nhau, mục 4.1.4); không gộp. Chỉ vật quá nhỏ / quá xa mới đi vào `ignore_region` | img_026 | CAL-02 |

**Chưa chốt (và vì sao):**
- Câu 1 (xe đạp điện: `bicycle` hay `motorbike`), câu 4 (ngưỡng "còn thấy ≥ 20%" với người trong xe có mui): trong 24 ảnh batch không có box nào được tick `can_review`, nên chưa có ca cụ thể để chốt. Giữ nguyên quy tắc v1.0 (`bicycle` chỉ cho xe đạp đạp chân; 20% theo mục 4.1.2) cho đến khi có ảnh ví dụ.
- Câu 7 (mức `conf` tối thiểu để nhận box máy không cần sửa): chưa tính tương quan giữa `conf` và tỉ lệ accepted theo lớp, nên chưa đủ số liệu đặt ngưỡng. Ngưỡng theo lớp hiện dùng là ngưỡng lọc của Part 3 (`qc/threshold_sweep.md`), không phải quy tắc chất lượng.

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_011, img_013 | Tuk-tuk có mui, hành khách và người lái ngồi trong | YOLO26-m gọi là `truck` (0.77) và `car` (0.42); chỉ 8/24 box `three_wheeler` của máy được giữ nguyên | Gán `three_wheeler`, sửa nhãn máy, không tin nhãn | Mục 2: xe ba bánh có mui hoặc thùng vẫn là `three_wheeler`. Mục 4.3: đếm bánh nhìn thấy trước khi nhận nhãn máy (CAL-01) |
| img_026 | Hàng xe máy đỗ sát nhau, gần và xa camera, ảnh đêm | YOLO26-m gán nhiều xe máy nhưng cụm xa dính liền | Gần: từng xe. Xa, không tách được: một `ignore_region` | Mục 4.1.6: "gần thì gán từng vật, xa thì `ignore_region`" (CAL-02) |
| img_003 | Xe ba gác chở hàng: mask SAM ôm cả hàng trên thùng | SAM 2.1 (click dương) phủ cả hàng và một phần người ngồi | Hàng gọn trong thùng thuộc xe; hàng nhô quá ~20% thì loại ra | Mục 4.1.5 và thêm mục 4.2.5 (CAL-03) |
| img_013 | Tuk-tuk có kính chắn gió, người lái nhìn thấy qua kính | SAM mask B ôm xe + người lái + người ngồi sau | Kính thuộc xe; người có polygon riêng, vùng chồng thuộc người | Mục 4.2.3 và 4.2.4 (CAL-04) |
| img_023, img_014, img_005 | Bẫy để lọt: 2 box lỏng, 1 vật bị thiếu | Bẫy do script tạo, không phải máy | Không đổi guideline; đây là lỗi duyệt của tôi | Không đổi (CAL-05); xử lý bằng checklist |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu 2, 3, 5, 6; làm rõ `three_wheeler` ở mục 2 và 4.3; thêm 4.2.5 cho mask (hàng nhô ra); câu 1, 4, 7 chưa chốt | CAL-01, CAL-02, CAL-03, CAL-04; CAL-05 không đổi guideline |
