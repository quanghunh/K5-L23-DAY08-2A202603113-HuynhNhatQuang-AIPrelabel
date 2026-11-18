# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 509; đối tượng sau duyệt: 566; ngưỡng accepted: IoU >= 0.9
- Đã **loại 16 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 459 | 459/509 = 90.2% | 509 box máy |
| edited (sửa) | 25 | 25/509 = 4.9% | 509 box máy |
| deleted (xoá) | 25 | 25/509 = 4.9% | 509 box máy |
| added (vẽ thêm) | 82 | 82/566 = 14.5% | 566 đối tượng sau duyệt |

Trong số deleted + added, **5 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 77.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 256 | 7 | 7 | 25 |
| bicycle | 5 | 1 | 3 | 4 |
| motorbike | 129 | 8 | 3 | 24 |
| car | 52 | 2 | 2 | 16 |
| bus | 3 | 0 | 0 | 3 |
| truck | 6 | 0 | 1 | 2 |
| three_wheeler | 8 | 7 | 9 | 8 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 70 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 45 | 21 | 25 | 68.2% (KTC 95%: 56–78%) | 64.3% (KTC 95%: 53–74%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 9/12 = 75.0% (KTC 95%: 47–91%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 3/3
  - loose: 1/3
  - extra: 3/3
  - missing: 2/3
  - lọt: img_023.jpg (loose), img_005.jpg (missing), img_014.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
