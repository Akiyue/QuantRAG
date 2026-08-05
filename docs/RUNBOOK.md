# Runbook — cách chạy thực tế

Tài liệu này trả lời đúng một câu: **gõ gì, khi nào, và nhìn vào đâu trước khi
đi tiếp.**

Chạy trên **server RTX 5000 Ada**. Laptop chỉ để viết code và đọc kết quả.

---

## 0. Nguyên tắc

Có bảy **cổng** (`.gates/`). Stage sau bị chặn cho tới khi cổng trước được ký:

```bash
./run.sh gate <tên>       # ký
rm .gates/<tên>           # thu hồi
./run.sh status           # xem còn thiếu gì
```

Cổng không phải thủ tục hành chính. Mỗi cái ứng với một phán đoán mà chỉ người
làm được, và bỏ qua nó nghĩa là số liệu cuối cùng không bảo vệ được.

**Mọi stage đều resumable.** Bị ngắt giữa chừng thì chạy lại đúng lệnh đó, nó
bỏ qua phần đã xong. Không cần dọn dẹp gì.

---

## Tuần 1 — hạ tầng và dữ liệu

### Ngày 1: môi trường + model

```bash
./run.sh setup                    # venv 3.12, deps, chạy test
./run.sh models                   # tải + lượng tử hóa 3 model × 4 mức
```

`setup` sẽ **dừng nếu test đỏ**. Đó là cố ý — đừng chạy gì tốn kém trên một
pipeline chưa xanh.

`models` mất khá lâu (tải ~15 GB). Cần `llama.cpp` đã build sẵn:

```bash
git clone https://github.com/ggerganov/llama.cpp ../llama.cpp
cmake -B ../llama.cpp/build -DGGML_CUDA=ON ../llama.cpp
cmake --build ../llama.cpp/build -j
```

### Ngày 1 (song song): khảo sát độ phủ tiếng Việt

```bash
./run.sh survey 21919
```

**Không cần GPU.** Chạy được ngay trong lúc model đang tải.

> Đây là phép đo có giá trị thông tin cao nhất trong cả 4 tuần. Nó cho biết
> nhánh tiếng Việt có khả thi không **trước khi** tiêu một giây GPU nào. Nhìn cột
> phần trăm theo quan hệ, giữ những quan hệ phủ ~100%, sửa `configs/relations.yaml`
> nếu cần bỏ bớt.

### Ngày 3–4: dataset

```bash
./run.sh dataset -- --target 500
```

Xong thì **rà tay hai thứ**, đây là công việc thật chứ không phải liếc qua:

1. **Alias** trong `data/facts.jsonl` — bỏ mỹ từ (`Kinh đô ánh sáng`), bỏ suy
   rộng ngữ nghĩa (`tiếng Mỹ` cho English). Label tiếng Anh đã được tự động thêm
   vào alias tiếng Việt; kiểm tra lại là đúng.
2. **Template tiếng Việt** trong `configs/relations.yaml` — bản tôi viết là nháp.
   Đây là câu chữ model bị chấm điểm trên đó.

```bash
./run.sh gate aliases
./run.sh gate templates
```

### Ngày 5: pilot

```bash
./run.sh pilot
```

Chạy 50 fact **hai lần** rồi so sánh. Đọc kỹ dòng cuối:

| Kết quả | Nghĩa là |
|---|---|
| `label disagreement: 0.0000` | Tất định hoàn toàn. Mọi flip về sau là do lượng tử hóa |
| `≤ 0.02` | Ghi con số này vào paper, chỉ coi flip rate cao hơn hẳn nó là kết quả |
| `> 0.02` | **Dừng lại.** QFR không diễn giải được ở mức nhiễu này. Sửa tính tất định trước |

Xem thêm vài prompt bằng mắt: `python scripts/inspect_prompts.py data/facts.jsonl`

```bash
./run.sh gate pilot
```

### Ngày 5: lọc tập known

```bash
./run.sh filter
```

Đọc bảng kích thước. **Nếu `known_all` tiếng Việt < 150**, script sẽ cảnh báo, và
bạn phải quyết định **ngay bây giờ**, không phải tuần 4:

- hạ nhánh VI xuống phân tích phụ và nói rõ trong Limitations, hoặc
- dùng model lớn hơn cho nhánh VI

```bash
./run.sh gate known
```

---

## Tuần 2 — chạy grid

```bash
tmux new -s grid
./run.sh main
```

96,000 lượt, ước ~15–30 giờ GPU. Cứ để chạy, log ở `logs/`.

### Nếu có nhiều GPU

Mỗi model đều vừa một card, nên **đừng chia một model qua hai GPU** — không được
gì mà thêm một biến vào tính tất định. Thay vào đó chạy hai tiến trình, mỗi cái
ghim một card:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_grid.py --pass main \
    --models qwen2.5-0.5b qwen2.5-1.5b &
CUDA_VISIBLE_DEVICES=1 python scripts/run_grid.py --pass main \
    --models qwen2.5-3b &
wait
```

Mỗi (model, precision) ghi ra file riêng nên không đụng nhau. Card nào chạy arm
nào được ghi vào `env.cuda_visible_devices` của từng record, khỏi phải dựng lại
từ lịch sử shell về sau.

Chia thế nào cho cân: 3B nặng gấp khoảng đôi 1.5B, nên `{0.5B, 1.5B}` trên một
card và `{3B}` trên card kia là xấp xỉ đều.

```bash
./run.sh dose
```

> **Bắt buộc chạy, kể cả khi tuần 3 không kịp phân tích.** Đây là nguyên liệu cho
> metric của bản journal. Chạy lại sau 6 tháng thì môi trường đã khác và số liệu
> journal không so sánh được với số liệu conference nữa — chi phí không sửa được.

Tùy chọn, nếu còn thời gian: `python scripts/run_grid.py --pass main --tier B`

### Validate evaluator

```bash
./run.sh review sample                        # sinh results/evaluator_review.csv
# điền cột human_label: TRUE / FAKE / REFUSAL / OTHER
# ĐỪNG nhìn cột auto_label khi quyết định
./run.sh review score
```

Dưới 0.95 thì sửa alias rồi chạy lại các cell generate bị ảnh hưởng.
Trên 0.95 thì `./run.sh gate evaluator` — và con số đó vào paper.

---

## Tuần 3 — XAI và phân tích

### Trước hết: viết paraphrase tay

100 mẫu trong `data/xai_sample.json` cần `evidence_fake_para` viết tay cho cả hai
ngôn ngữ. Nếu để LLM sinh rồi dùng thẳng, SemanticGap sẽ đo chất lượng paraphrase
chứ không đo model.

```bash
./run.sh gate paraphrase
./run.sh xai
```

### Phân tích

```bash
./run.sh analyze
```

**Đọc theo đúng thứ tự này:**

| # | File | Vì sao trước |
|---|---|---|
| 1 | `00_margin_control.md` | Phép kiểm định phủ định. AUC ≥ 0.85 nghĩa là flip chỉ là nhiễu quanh biên và **claim chính sụp** — biết ngay, đừng để tới lúc viết |
| 2 | `05_diagnostics.md` | Agreement generate-vs-scoring, phân bố nhãn. `OTHER` nhiều thường là alias thiếu chứ không phải model kém |
| 3 | `02_flips.md` | Cột **asymmetry** — claim có hướng, sống sót qua null ròng |
| 4 | `03_reliance.md` | ΔP_ctx là metric chính, không phải ΔR thô |
| 5 | `04_interactions.md` | Nơi tín hiệu nằm nếu hiệu ứng ròng null |
| 6 | `xai/12_attribution_vs_flips.md` | So hai cột: nếu margin dự đoán flip tốt ngang A\*, thì attribution không mang thông tin độc lập |

```bash
./run.sh gate margin      # nếu margin control không giết claim
```

Hình nằm ở `results/known_all/figures/` — PDF cho LaTeX, PNG để xem. **Mở PNG ra
nhìn.** Palette đã validate bằng script, còn layout thì chưa ai kiểm — tìm nhãn
đè nhau, chữ bị cắt.

---

## Tuần 4 — viết

Bảng đã ở dạng Markdown và CSV, bê thẳng vào LaTeX được.

Rà từng claim ngược về số liệu. Chạy lại mọi trường hợp bất thường.

---

## Khi có sự cố

| Triệu chứng | Nguyên nhân thường gặp |
|---|---|
| `blocked by gate 'x'` | Đúng như thiết kế. Làm việc mà cổng đó yêu cầu, rồi ký |
| `boundary errors` > 0 ở cuối grid | Tokenizer nuốt ranh giới prompt/đáp án. Những cell đó **không dùng được**, không phải kém chính xác. Kiểm tra template |
| Nhiều nhãn `OTHER` | Alias thiếu, không phải model kém. Sửa alias rồi chạy lại phần generate |
| `no main-pass records` | Chạy `./run.sh main` trước |
| Grid bị ngắt | Chạy lại đúng lệnh cũ, nó tự bỏ qua phần đã xong |
| Bảng rỗng | Không đủ item sau khi lọc scope. Xem lại kích thước split |

## Chạy thử khi chưa có model

Mọi thứ trừ inference thật đều chạy được không cần GPU:

```bash
python scripts/run_grid.py --pass main --mock --facts data/facts.sample.jsonl \
    --out-dir runs/tmp
pytest -q
```
