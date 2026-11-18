# Thông tin học viên

- Họ và tên: Huỳnh Nhật Quang
- MSHV: 2A202603113
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: quanghunh
- Repo bài nộp (URL): K5-L23-DAY08-2A202603113-HuynhNhatQuang-AIPrelabel,
- Commit chốt bài (hash): a3e1919bb64466ee12defbd290983b5d7f97c439
- Máy: Linux, RAM dưới 8 GB, không dùng GPU trên máy (model chạy trên Google Colab T4)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 459 / 25 / 25 (mẫu số: 509 box máy); added: 82 (mẫu số: 566 đối tượng)
- So với gold: precision / recall của **máy** 68.2% (45/66) / 64.3% (45/70)
- Bắt bẫy: 9/12 = 75.0%
- Thời gian: gán tay 5.0 phút/ảnh (20 phút / 4 ảnh); duyệt có pre-label khoảng 2.1 phút/ảnh (50 phút / 24 ảnh)
- Lỗi QC: 0.0% theo ảnh (0/5), 0.0% theo đối tượng (0/107)
- **Quyết định:** Rework — bắt bẫy 9/12 và mẫu QC 5 ảnh quá nhỏ để chứng minh lỗi ≤ 2%
