Bản vá v5 bổ sung auto-tuning cho FIS rules.

File mới:
- `search_fis_rules.py`: grid search 48 cấu hình để tối ưu Full FIS trên checkpoint hiện có.
- `fis_modules_optimal.py`: cấu hình bảo thủ hơn, dùng khi Full vẫn thua `snr_only` trên AWGN.

Quy trình đề nghị:

1. Huấn luyện baseline và FIS như patch v4.
2. Chạy grid search trên checkpoint FIS tốt nhất hiện có.
3. Đọc `best_fis_config.json` và chép các tham số tốt nhất vào `fis_modules.py` hoặc dùng trực tiếp `fis_modules_optimal.py`.
4. Chạy lại `run_paper_sims.py` để xác nhận.

Ví dụ AWGN:

```bash
python /media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/search_fis_rules.py \
  --baseline_ckpt /media/data/students/nguyenquangkhai/ckpts_baseline_awgn/baseline_best.pth \
  --fis_ckpt /media/data/students/nguyenquangkhai/ckpts_fis_awgn/fis_power_best.pth \
  --channel AWGN \
  --snrs 1,4,7,10,13 \
  --budget 1.0 \
  --max_batches 20 \
  --objective full_vs_all \
  --save_dir fis_search_awgn
```

Ví dụ Rayleigh:

```bash
python /media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/search_fis_rules.py \
  --baseline_ckpt /media/data/students/nguyenquangkhai/ckpts_baseline_Rayleigh/baseline_best.pth \
  --fis_ckpt /media/data/students/nguyenquangkhai/ckpts_fis_Rayleigh/fis_power_best.pth \
  --channel Rayleigh \
  --snrs 1,4,7,10,13 \
  --budget 1.0 \
  --max_batches 20 \
  --objective full_vs_all \
  --rayleigh_equalize \
  --save_dir fis_search_rayleigh
```

Nếu muốn thay nhanh sang cấu hình bảo thủ hơn, có thể sửa trong `fis_modules.py`:

```python
self.c = torch.tensor([1.30, 1.20, 1.08, 1.04, 1.00, 0.92])
self.w0 = 0.05
smooth_kernel = 3
alpha_linear = 0.6
```

Lưu ý: search script không retrain, chỉ chọn rule tốt nhất trên backbone/checkpoint hiện có. Đây là cách nhanh nhất để đẩy Full FIS vượt `snr_only` mà không thay đổi kiến trúc.
