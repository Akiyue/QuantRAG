# QuantRAG — Kế hoạch nghiên cứu 4 tuần

**Đề tài:** Does Quantization Change What Small Language Models Trust?
Measuring Context–Memory Reliance in Local RAG

**Đích ngắn hạn:** AICI 2027 (conference paper, 4 tuần)
**Đích dài hạn:** journal Q1 (bản mở rộng, 2027)

---

## 0. TL;DR — các quyết định đã chốt

| Hạng mục | Quyết định | Lý do |
|---|---|---|
| Chạy ở đâu | **RTX 5000 Ada 32 GB** làm chính; Colab dự phòng; laptop chỉ để dev | 32 GB dư dả, không có ràng buộc bộ nhớ |
| Stack chính | **llama.cpp / GGUF (CUDA build)** cho mọi mức precision | Một runtime = loại bỏ confound stack; đúng với motivation local deployment |
| Precision ladder | F16 → Q8_0 → Q4_K_M → Q3_K_M | Cùng một quantizer, cùng tokenizer, cùng sampler |
| Robustness check | Thêm **AWQ 4-bit** (họ quantizer thứ hai), báo cáo riêng | Chứng minh hiệu ứng không phải artifact của một phương pháp lượng tử hóa |
| Model | Qwen2.5-Instruct 0.5B / 1.5B / 3B (+7B nếu kịp) | Một family, thang size sạch; tiếng Việt khá nhất nhóm nhỏ |
| Đo lường chính | **Teacher-forced scoring**, generation để bổ trợ | Logprob là đại lượng đúng để đo reliance, không phụ thuộc quirk decoding |
| Headline claim | **Dịch chuyển sự phân xử context ↔ memory** ($\Delta R$, $P_{\text{ctx}}$), + các hiệu ứng **interaction** | QFR/instance-instability đã bị scoop 7/2026 — xem §2.4 |
| XAI | Span-level causal ablation + control span | Chạy được trên mọi precision, không cần gradient |
| Ngôn ngữ | EN + VI trên **cùng** 500 fact (paired) | Điểm khác biệt chính so với literature |
| Python | **3.12** (KHÔNG dùng 3.14) | Xem §1.4 |

---

## 1. Hạ tầng tính toán

### 1.1 Tài nguyên

| Máy | Vai trò |
|---|---|
| **RTX 5000 Ada, 32 GB VRAM** | **Chạy chính.** Toàn bộ grid, XAI, ablation |
| Colab | Dự phòng / tràn tải. Không dùng làm chính (hay disconnect, khó reproducible) |
| Laptop (AMD iGPU, 16 GB RAM) | Chỉ để viết code, debug trên 0.5B, viết paper |

32 GB VRAM là dư dả cho phạm vi này: 3B FP16 ≈ 6 GB, 7B FP16 ≈ 14 GB, còn thừa
chỗ cho batch lớn. Không có ràng buộc bộ nhớ nào đáng lo.

### 1.2 CUDA mở khóa những gì

So với phương án CPU, có bốn thứ trở nên khả thi và **nên tận dụng ngay ở bài
hội nghị** (trước đây phải để dành cho journal):

1. **Robustness check giữa hai họ quantizer** — xem §1.3. Đây là nâng cấp lớn nhất.
2. **Generation grid đầy đủ** trên mọi precision, không phải rút gọn.
3. **Thêm C3** (multi-document conflict) trở lại thiết kế.
4. **Thang size mở rộng tới 7B** — vượt qua ngưỡng "sub-7B không dùng tốt retrieved
   document" mà literature nêu, nên bản thân việc bắc cầu qua ngưỡng đó là một
   kết quả có ý nghĩa.

### 1.3 Quyết định về stack — vẫn là điểm sống còn

Có CUDA **không** có nghĩa là được trộn stack tùy tiện. Luận điểm cũ vẫn đứng vững:

> Nếu chạy FP16 bằng HF transformers còn 4-bit bằng llama.cpp, bạn đang so sánh
> hai inference stack chứ không phải hai precision. Reviewer sẽ giết bài, và họ đúng.

**Thiết kế hai tầng:**

**Tầng A — stack chính: llama.cpp / GGUF (CUDA build).**
Ladder F16 → Q8_0 → Q4_K_M → Q3_K_M. Cùng runtime, cùng tokenizer, cùng sampler,
cùng cách tính logprob. Toàn bộ kết quả chính báo cáo trên tầng này.

Vì sao vẫn chọn GGUF làm chính dù đã có CUDA:
- Đúng với motivation *local deployment* — người chạy LLM local thật sự dùng
  GGUF (Ollama, LM Studio, llama.cpp).
- Ladder 4 mức trong **một** quantizer duy nhất, sạch về mặt so sánh.
- Reviewer reproduce được không cần GPU.

**Tầng B — robustness check: một họ quantizer thứ hai ở 4-bit.**
Chọn **AWQ** (hoặc GPTQ) trên HF/vLLM, chạy **cùng 500 fact, cùng prompt**, so với
Q4_K_M. Mục đích duy nhất: chứng minh hiệu ứng **không phải artifact của một
phương pháp lượng tử hóa cụ thể**.

Báo cáo tầng B trong một subsection riêng, so sánh *hướng và độ lớn* của $\Delta R$
và QFR, **không** trộn chung bảng với tầng A. Ghi rõ trong Limitations rằng so
sánh chéo stack chỉ mang tính định hướng.

Đây chính là câu mà reviewer khó tính nhất sẽ hỏi — và bây giờ bạn trả lời được
ngay trong bài hội nghị.

**Lưu ý trung thực khi viết:** Q4_K_M / Q3_K_M là *k-quant hỗn hợp*, không phải
uniform INT4/INT3. Mô tả chính xác trong Experimental Setup: bit-width hiệu dụng
trung bình, layer nào giữ precision cao hơn. Đừng gọi tắt "INT4" trong Method.

### 1.4 Python 3.14 là vấn đề

`pyproject.toml` đang đặt `requires-python = ">=3.14"` và venv là CPython 3.14.
`torch`, `llama-cpp-python`, `autoawq` **chưa có wheel cho 3.14**. Bạn sẽ mất
nửa ngày build from source rồi vẫn fail.

**Hành động ngày 1** (cả laptop lẫn server):

```bash
uv venv --python 3.12
uv pip install numpy pandas scipy statsmodels pyyaml tqdm
# tầng A
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-binary llama-cpp-python
# tầng B
uv pip install torch transformers accelerate autoawq
```

Ghim version trong `requirements.lock` và ghi lại driver/CUDA version — thuộc
Reproducibility statement.

### 1.5 Ngân sách compute

Ước lượng trên RTX 5000 Ada (hiệu chỉnh lại ở pilot ngày 6):

| Tác vụ | Throughput ước tính | Ghi chú |
|---|---|---|
| Teacher-forced scoring (batch) | vài nghìn item/giờ | 1 forward pass, batch được |
| Free generation, 32 tok (batch) | ~1–3 nghìn item/giờ cho 3B | Batch + greedy |

Grid đầy đủ tầng A:

```
500 fact × 2 lang × 4 condition × 2 mode = 8,000 prompt / (model, precision)
× 3 model size × 4 precision                = 96,000 lượt
```

Ước tính **15–30 giờ GPU** cho cả scoring lẫn generation. Chia đều trong tuần 2
là thoải mái, còn dư slack cho chạy lại.

**Vẫn giữ thiết kế scoring-first (§5.1)** — nhưng bây giờ lý do là *phương pháp*
chứ không phải tiết kiệm compute: teacher-forced logprob là đại lượng đúng để đo
reliance, không phụ thuộc quirk của decoding. Generation dùng để bắt refusal,
hallucination và các đáp án ngoài hai candidate.

### 1.6 Quy tắc vận hành server

- Chạy trong `tmux` / `screen`, log ra file, **runner resumable** (§8.2).
- Ghim `CUDA_VISIBLE_DEVICES`, cố định seed, `temperature=0`, tắt mọi non-determinism
  có thể tắt. Ghi lại: kernel nào cũng có thể còn non-deterministic — chạy 1 cấu
  hình **2 lần** ở pilot để đo mức dao động, và báo cáo con số đó trong paper.
- `nvidia-smi` log vào run metadata (GPU, driver, CUDA) cho Reproducibility.

---

## 2. Phạm vi: chia đôi ngay từ đầu

### 2.1 Bài AICI (4 tuần) — ĐO LƯỜNG

Đóng góp tuyên bố:

1. Một protocol có kiểm soát tách bạch **factual accuracy** / **context compliance** /
   **parametric retention** — ba thứ mà literature hay gộp chung.
2. Đánh giá paired F16→Q8→Q4→Q3 trên small local LLM dưới context–memory conflict,
   song ngữ EN–VI trên cùng tập fact.
3. Đo **dịch chuyển sự phân xử** giữa context và parametric memory theo precision,
   và các hiệu ứng interaction với instruction mode và ngôn ngữ. QFR báo cáo như
   phần mở rộng của *correctness agreement* (2607.08734) sang bối cảnh xung đột —
   **có trích dẫn, không tuyên bố là mới**.
4. Causal span attribution cho thấy mức ảnh hưởng nhân quả thật sự của evidence
   span thay đổi thế nào theo precision.
5. **Robustness qua hai họ quantizer** (k-quant GGUF và AWQ) — nhờ có GPU nên
   đưa được vào ngay bản hội nghị. Trả lời trước câu hỏi khó nhất của reviewer.

Dừng ở **mô tả hiện tượng**. Không đụng tới cơ chế bên trong, không đề xuất mitigation.

### 2.2 Bài journal Q1 (sau) — CƠ CHẾ + CAN THIỆP

**Cố ý giữ lại, không viết vào bài hội nghị:**

1. **Định vị nguyên nhân.** Activation patching giữa F16 và Q4: thay activation
   của model lượng tử bằng activation F16 tại layer $L$, xem flip biến mất ở đâu.
   → trả lời "sai lệch reliance sinh ra ở tầng nào, attention hay MLP".
   *(Cần HF/PyTorch, không làm được trong llama.cpp — đây là lý do tầng B ở §1.3
   cũng là khoản đầu tư hạ tầng cho bản journal.)*
2. **Mitigation — thứ thực sự nâng hạng.** Nếu giữ một nhóm layer/head cụ thể ở
   precision cao khôi phục được phần lớn context grounding với chi phí bộ nhớ nhỏ,
   bạn có một *phương pháp*, không phải một *quan sát*.
3. **Chuyển sang RAG thật.** Bài hội nghị dùng gold document tự chế. Bài journal
   dùng retrieval thật (BM25 + dense trên Wikipedia EN/VI), top-k thật, có noise
   thật, multi-hop. Chứng minh phát hiện chuyển được sang deployment.
4. Mở rộng: thêm model family (Llama-3.2, Gemma), thêm ngôn ngữ, thang size lên
   7B–14B, quantizer thứ ba.

**Nguyên tắc:** diagnosis + mitigation là cấu trúc kinh điển của bài journal mạnh.
Measurement-only rất khó qua Q1.

### 2.3 Hai thứ phải xác minh trước khi nộp bản hội nghị

- [ ] **Proceedings AICI có index (Scopus/Springer/IEEE) không?**
      Có → prior publication archival → journal sẽ áp luật extension (thường
      ≥30% nội dung mới, bắt buộc khai báo + trích dẫn bản hội nghị).
      Không → tự do tái sử dụng hơn nhiều. Câu trả lời đổi hẳn cách viết bản đầu.
- [ ] **Chính sách extended-paper của 2–3 journal đích.** Đọc Guide for Authors
      NGAY BÂY GIỜ, không phải sau khi có kết quả. Một số journal từ chối thẳng
      bản mở rộng từ hội nghị. Tra quartile trên SJR năm hiện tại.
- [ ] CFP của AICI: deadline thật, page limit, có anonymize không. 8 trang và
      4 trang là hai bài khác nhau.

### 2.4 Định vị so với literature — ĐỌC TRƯỚC KHI CHẠY BẤT CỨ THỨ GÌ

Khảo sát ngày 04/08/2026. **Chưa systematic, cần bạn tự đọc bản đầy đủ của các
paper đánh dấu 🔴.**

| Paper | Có gì trùng | Không có gì |
|---|---|---|
| 🔴 **The Illusion of Equivalency in Quantization** (2607.08734, 7/2026) | **Chính là headline cũ của ta.** Định nghĩa *correctness agreement* = độ trùng dự đoán giữa base và quantized; claim lõi: accuracy/perplexity ổn định trong khi đáp án từng item đổi nhiều. 8→2 bit. Tìm thấy Q/K projection nhạy hơn V/O | Không RAG, không context–memory conflict, không attribution, không đa ngữ |
| 🔴 **Through a Compressed Lens: Quantization on Factual Knowledge Recall** (2505.13963) | Quantization × tri thức factual + neuron-level attribution. Llama3-8B, Qwen2.5-7B/14B; GPTQ/AWQ/BnB 4&8-bit | Chỉ parametric knowledge, **không có context/RAG**, không có xung đột, English-only, không có model <7B |
| **Interpreting the Effects of Quantization on LLMs** (2508.16785) | 5 phương pháp interpretability; có **calibration analysis (ACE)** — xác nhận đúng mối lo ở §5.2 | Không RAG, không conflict, English-only |
| **Impact of Quantization on RAG: Small LLMs** (2406.10251) | Quantization × RAG, FP16 vs INT4, 7–8B | Không conflict, không tách compliance vs parametric, không attribution |
| 🔴 **Does RAG Know When Retrieval Is Wrong?** (2605.14473, 5/2026) | Context compliance dưới knowledge conflict + causal perturbation | **Không có quantization.** Chạy trên Gemini/Claude, không phải small local |
| **Compressed Causal Reasoning** (2512.13725) | Quantization × GraphRAG × counterfactual. Llama-3-8B, BF16/INT8/NF4 | Causal-ladder reasoning, không phải context–memory conflict. Không attribution, English-only |
| **How Does Quantization Affect Multilingual LLMs?** (2407.03211) | Quantization hại non-Latin script nặng hơn | Không RAG, không conflict |
| 🔴 **English K-Quantization Does Not Disproportionately Diminish Multilingual Performance** (2503.03592) | **Phản chứng trực tiếp cho nhánh VI**, và đúng về k-quant — chính stack tầng A của ta | Không RAG, không conflict |

#### Hệ quả bắt buộc

**1. Headline cũ đã bị scoop.** "Instance-level instability" giờ là claim đã công bố
(2607.08734) dưới tên *correctness agreement*. **Không được đặt làm đóng góp chính.**

Đổi headline sang: **quantization có làm dịch chuyển sự phân xử giữa context và
parametric memory không** — tức $\Delta R$ / $P_{\text{ctx}}$ lên làm metric trung
tâm, QFR xuống làm metric bổ trợ và **phải trích dẫn 2607.08734**, đóng khung là
"mở rộng correctness agreement sang bối cảnh context–memory conflict".

**2. Ô trống thật sự còn lại.** Không paper nào chạm vào *sự phân xử* (arbitration):
2505.13963 chỉ đo parametric recall, 2406.10251 chỉ đo accuracy RAG, 2605.14473 đo
arbitration nhưng không có quantization và dùng model frontier. Cộng thêm hai trục
chưa ai làm: **sub-4B** (mọi paper quantization đều ≥7B) và **EN–VI paired**.

**3. Rủi ro null result quay lại** vì headline đã đổi. Đối phó: đặt cược vào
**interaction**, chỗ chưa ai đo và ít khả năng null:
- precision × instruction mode (strict vs truth-seeking)
- precision × language (EN vs VI)
- **tính bất đối xứng của flip** dưới conflict: true→fake so với fake→true.
  Đây là claim *có hướng*, sống sót ngay cả khi $\Delta R$ ròng bằng 0.

**4. Phải engage trực diện 2503.03592.** Nó nói k-quant **không** hại đa ngữ một
cách bất cân xứng — trong khi 2407.03211 nói ngược lại cho non-Latin. Tiếng Việt là
Latin-script nhưng nhiều dấu phụ, nằm đúng giữa hai kết luận đó. Đóng khung nhánh VI
là **kiểm định một tranh cãi đang mở**, không phải "thêm ngôn ngữ cho vui". Mạnh hơn
nhiều và bắt buộc trích cả hai.

**5. Related Work phải có bảng phân biệt** như bảng trên, cột "cái chúng tôi làm khác".
Reviewer sẽ hỏi "khác gì X?" cho ít nhất 3 paper — trả lời sẵn trong bảng.

**6. Việc phải làm tuần 1:** đọc **toàn văn** 2607.08734, 2505.13963, 2605.14473,
2503.03592. Nếu 2607.08734 đã có sẵn phần RAG hoặc conflict mà abstract không nêu,
bạn phải xoay trục lần nữa — và biết điều đó ở ngày 2 rẻ hơn ở ngày 25 rất nhiều.

---

## 3. Dataset

### 3.1 Nguồn — dùng dataset có sẵn, KHÔNG dịch tay 500 fact

**Nguồn chính: `NeelNanda/counterfact-tracing` trên HuggingFace.** 21,919 dòng,
và điểm quyết định: mỗi dòng mang **Wikidata QID** cho cả `target_true_id` lẫn
`target_false_id`, cộng PID cho quan hệ.

Hệ quả: **nhánh tiếng Việt không cần dịch thực thể chút nào.** Lấy label tiếng
Việt chính thức từ Wikidata theo QID → cặp EN–VI khớp *chính xác theo cấu tạo*,
không có nhiễu dịch thuật trong tên thực thể.

Các nguồn khác đã kiểm tra:

| Dataset | Truy cập | Dùng được gì |
|---|---|---|
| `NeelNanda/counterfact-tracing` | HF, 21,919 dòng | **Nguồn chính.** Có QID/PID |
| `azhx/counterfact` | HF | Bản ROME đầy đủ, có paraphrase prompt — nguồn cho `q_filter`/`q_eval` |
| `azhx/counterfact-filtered-gptj6b` | HF | Đã lọc theo tri thức GPT-J. **Không dùng làm known-subset** — lọc theo model khác model của ta |
| ConflictBank | `Warrieryes/CB_claim_evidence`, `Warrieryes/CB_qa` | 7.4M claim-evidence. Quá lớn, English-only. Để trích dẫn và cho bản journal |
| `osunlp/ConflictQA` | HF | Xung đột parametric vs contextual. Đối chiếu, không phải nguồn chính |
| mLAMA | 53 ngôn ngữ, có tiếng Việt, 44 quan hệ | Template VI **dịch máy** và đã có nghiên cứu ghi nhận disfluent. Chỉ dùng làm bản nháp template, phải sửa tay |

**Không để LLM sinh fact.** LLM chỉ dùng để nháp template tiếng Việt, và người
sửa lại toàn bộ.

### 3.2 Đo trước, chọn sau: lọc theo QUAN HỆ

Đã khảo sát thật (`scripts/survey_counterfact_vi.py`, 300 dòng CounterFact):

```
rows where BOTH objects have a Vietnamese label: 252/300 (84.0%)

  P103  17/17  100.0%      P176   6/20   30.0%
  P27   16/16  100.0%      P413   4/14   28.6%
  P17   15/15  100.0%      P740  10/13   76.9%
  P495  15/15  100.0%      P190  13/14   92.9%
  P30   14/14  100.0%
  P20   12/12  100.0%
  P136  11/11  100.0%
  P106  11/11  100.0%
```

Độ phủ **không đồng đều theo quan hệ**, và đó là thông tin dùng được: chọn nhóm
quan hệ phủ ~100% (P103 ngôn ngữ mẹ đẻ, P27 quốc tịch, P17 quốc gia, P495 nước
xuất xứ, P30 châu lục, P20 nơi mất, P136 thể loại, P106 nghề nghiệp) thì có tập
EN–VI paired gần như trọn vẹn, **không dịch một tên thực thể nào**.

Bỏ P176 (nhà sản xuất) và P413 (vị trí thi đấu) — độ phủ VI dưới 30%.

> **Việc này chạy được NGÀY 1, không cần GPU, không cần model.** Rủi ro lớn nhất
> của dự án — nhánh tiếng Việt co lại quá nhỏ — trước đây phải chờ tới ngày 5 mới
> biết. Giờ biết trước khi tiêu một giây GPU nào.

### 3.3 Alias: lấy từ Wikidata, nhưng PHẢI lọc tay

Wikidata trả về alias theo ngôn ngữ, nhưng dùng thô thì nguy hiểm. Kết quả thật:

```
Q84   London  -> 'Luân Đôn'   aliases: []                      <-- rỗng!
Q90   Paris   -> 'Paris'      aliases: ['Ba Lê', 'Kinh đô ánh sáng', 'Pa-ri', ...]
Q30   US      -> 'Hoa Kỳ'     aliases: ['Mỹ', 'Hợp chúng quốc Hoa Kỳ', 'Mĩ', ...]
Q1860 English -> 'tiếng Anh'  aliases: ['tiếng Mỹ', 'tiếng Mĩ', 'Anh ngữ']
```

Hai luật rút ra, cả hai đều bắt buộc:

1. **Luôn thêm label tiếng Anh vào alias set tiếng Việt.** London có alias VI
   **rỗng** — model nhỏ code-switch liên tục và sẽ trả lời "London" trong ngữ
   cảnh tiếng Việt. Không thêm thì câu trả lời đúng bị xếp vào `OTHER`.
2. **Lọc tay các alias suy rộng.** `Kinh đô ánh sáng` là mỹ từ chứ không phải đáp
   án; `tiếng Mỹ` cho English là một bước nhảy ngữ nghĩa sẽ gây false positive.

Đây chính là rủi ro "alias/dấu tiếng Việt làm sai evaluator" trong §10 — Wikidata
giảm được phần lớn công, nhưng không thay được bước rà tay.

### 3.2 Schema

Một file `data/facts.jsonl`, mỗi dòng:

```json
{
  "fact_id": "geo_0042",
  "domain": "geography",
  "relation": "capital_of",
  "subject": {"en": "France", "vi": "Pháp"},
  "object_true": {"en": "Paris", "vi": "Paris"},
  "object_fake": {"en": "London", "vi": "London"},
  "question": {
    "en": "What is the capital of France?",
    "vi": "Thủ đô của Pháp là gì?"
  },
  "evidence_true": {
    "en": "The capital of France is Paris.",
    "vi": "Thủ đô của Pháp là Paris."
  },
  "evidence_fake": {
    "en": "The capital of France is London.",
    "vi": "Thủ đô của Pháp là London."
  },
  "evidence_irrelevant": {
    "en": "France is a member state of the European Union.",
    "vi": "Pháp là một quốc gia thành viên của Liên minh châu Âu."
  },
  "aliases_true":  {"en": ["Paris"], "vi": ["Paris", "Pa-ri"]},
  "aliases_fake":  {"en": ["London"], "vi": ["London", "Luân Đôn"]},
  "source": "counterfact:case_1234",
  "vi_translation_checked": true
}
```

### 3.3 Tiêu chí lọc fact

Giữ lại fact khi:
- Ổn định theo thời gian (không phải "tổng thống hiện tại của X").
- Đáp án ngắn, một thực thể, ≤4 token ở cả hai ngôn ngữ.
- Ít alias gây nhập nhằng.
- `object_fake` **cùng semantic type** với `object_true` (thành phố ↔ thành phố,
  không phải thành phố ↔ số).
- Bản dịch VI đã được người kiểm tra.

**Cạm bẫy tiếng Việt phải xử lý ngay:** tên riêng có hai dạng (London / Luân Đôn,
Vienna / Viên). Alias list phải phủ cả hai, và evaluator phải normalize (bỏ dấu,
lowercase, strip) trước khi match. Đây là chỗ dễ tạo ra số liệu sai nhất.

### 3.4 Quy mô

Thu 800 candidate → lọc còn **500** đưa vào pipeline, chia 5 domain × 100:
geography, science, history, culture, technology.

### 3.5 Lọc parametric-known — CHỖ DỄ SAI THỐNG KÊ NHẤT

Vấn đề: nếu bạn chọn item dựa trên "F16 trả lời đúng khi không có context",
rồi so sánh F16 với Q4 trên chính tập đó, bạn đã **selection on the baseline arm**.
Regression to the mean sẽ tự động làm Q4 trông tệ hơn, kể cả khi quantization
không gây hại gì.

**Giải pháp (bắt buộc):**

Mỗi fact có **2 paraphrase câu hỏi độc lập**: `q_filter` và `q_eval`.

- Lọc bằng `q_filter`.
- Đánh giá bằng `q_eval`.

Và định nghĩa ba tập:

| Tập | Định nghĩa | Dùng để |
|---|---|---|
| `KNOWN_ALL` | Đúng ở **cả** F16, Q8, Q4 (trên `q_filter`, C0) | Phân tích chính — không thiên vị arm nào |
| `KNOWN_FP16` | Chỉ F16 đúng | Báo cáo phụ, có ghi chú về selection bias |
| `UNKNOWN` | Không precision nào đúng | Đối chứng: đo khả năng dùng retrieval thuần túy |

Phân tích chính chạy trên `KNOWN_ALL`. Báo cáo cả hai định nghĩa và chỉ ra kết
luận không đổi — đây là một đoạn nửa trang trong paper cứu bạn khỏi major revision.

---

## 4. Điều kiện thí nghiệm

### 4.1 Context conditions

| Mã | Nội dung context | Đo cái gì |
|---|---|---|
| **C0** | Không có context | Parametric knowledge nền |
| **C1** | `evidence_true` | Sử dụng retrieval bình thường |
| **C2** | `evidence_fake` | **Điều kiện chính** — context–memory conflict |
| **C3** | `evidence_fake` + `evidence_true`, 2 doc | Xung đột giữa các tài liệu; đảo thứ tự để đo position bias |
| **C4** | `evidence_irrelevant` | Control: model có copy bừa từ mọi context không |

C3 đưa trở lại nhờ có GPU. Chạy **hai biến thể thứ tự** (fake-first / true-first)
— chênh lệch giữa hai biến thể chính là position bias, một kết quả phụ rẻ tiền
mà reviewer thích. Nếu tuần 2 trượt lịch, C3 là thứ **đầu tiên** bị cắt.

C4 rẻ và trả lời một câu hỏi reviewer chắc chắn hỏi: "model chỉ đơn giản là
nhại lại context bất kể nội dung à?" — đừng bỏ.

### 4.2 Instruction modes

**Mode A — strict grounding**

```
Answer using ONLY the provided document, even if it contradicts what you
previously knew. Answer with the entity name only.
```

CCR cao = tuân thủ tốt.

**Mode B — truth-seeking**

```
Use the document as a reference, but prioritize giving the factually correct
answer. If the document appears incorrect, say so. Answer with the entity
name only, optionally followed by a brief warning.
```

CCR cao ở C2 = bị misinformation dắt mũi.

> **Quy tắc bất di bất dịch:** không bao giờ báo cáo CCR mà không kèm instruction mode.
> Cùng một con số mang hai ý nghĩa trái ngược.

Chạy cùng một dataset dưới hai instruction là điểm sắc nhất của thiết kế — nó tách
được *khả năng follow instruction* khỏi *mức phụ thuộc context*.

### 4.3 Prompt template

Cố định, versioned trong `configs/prompts/v1.yaml`. Dùng chat template gốc của
Qwen2.5 qua `llama.cpp`. **Không đổi template giữa các precision** — hiển nhiên
nhưng là lỗi người ta vẫn mắc.

---

## 5. Giao thức đo lường

### 5.1 Hai chế độ đo

**(a) Teacher-forced scoring — chạy trên TOÀN BỘ grid.**

Với mỗi prompt $x$ và mỗi candidate answer $a$, tính logprob của $a$ bị ép
(không sinh tự do):

$$s_{\text{raw}}(a \mid x) = \sum_{j=1}^{m} \log P(t_j \mid x, t_{<j})$$

$$s(a \mid x) = \frac{1}{m}\, s_{\text{raw}}(a \mid x) \quad \text{(length-normalized)}$$

Chỉ 1 forward pass, batch được. Đây là đại lượng **đúng** để đo reliance: nó
không phụ thuộc quirk của decoding, và cho tín hiệu liên tục thay vì nhị phân.

**Length normalization là bắt buộc cho nhánh song ngữ.** "London" có thể là 1 token
còn "Luân Đôn" là 3–4 token; không normalize thì bạn đang đo tokenizer chứ không
đo model. Báo cáo cả `s_raw` và `s` trong appendix, xác nhận kết luận không đảo.

**(b) Free generation — chạy trên TOÀN BỘ grid** (đủ GPU, không cần rút gọn nữa).

Cần để báo cáo CCR/PRR từ văn bản thật, và để bắt refusal, hallucination, các đáp
án nằm ngoài hai candidate — những thứ scoring không thấy được.

Greedy decoding (`temperature=0`), seed cố định, `max_tokens=32`, batch.

**Đối chiếu bắt buộc:** tính tỉ lệ mà đáp án generate trùng với argmax của
teacher-forced scoring. Nếu tỉ lệ này thấp (<90%), hai metric đang đo hai thứ khác
nhau và bạn phải giải thích tại sao trước khi diễn giải bất cứ điều gì.

### 5.2 Metrics

Ký hiệu: $\hat{y}_i$ = đáp án model, $o_i^{ctx}$ = đáp án theo context,
$o_i^{true}$ = đáp án đúng.

**Context Compliance Rate** (trên C2)

$$\text{CCR} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = o_i^{ctx}]$$

**Parametric Retention Rate** (trên C2, tập `KNOWN_ALL`)

$$\text{PRR} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = o_i^{true}]$$

**Quantization Flip Rate** — *metric headline*

$$\text{QFR}_b = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[\hat{y}_i^{(b)} \neq \hat{y}_i^{(\text{F16})}\right]$$

Bắt buộc phân loại flip, không báo con số trần:

| Loại flip | Ý nghĩa |
|---|---|
| true → fake | Quantization làm model ngả về context sai |
| fake → true | Quantization làm model kháng cự tốt hơn |
| answer → refusal | Mất năng lực trả lời |
| answer → other | Hallucination mới |

**Knowledge Reliance Score**

$$R(x) = s(a_{ctx} \mid x) - s(a_{par} \mid x)$$

$$\Delta R_b = R_b - R_{\text{F16}}$$

$R>0$: ngả về context. $R<0$: ngả về parametric memory.

**Cảnh báo về $\Delta R$:** quantization làm đổi calibration/độ nhọn của phân bố
logit, nên $\Delta R$ thô có thể bị chi phối bởi một dịch chuyển kiểu temperature
toàn cục chứ không phải thay đổi reliance thật. Vì vậy báo cáo song song dạng
chuẩn hóa 2 lựa chọn, ít nhạy với scale:

$$P_{\text{ctx}}(x) = \frac{\exp\!\big(s(a_{ctx}\mid x)\big)}{\exp\!\big(s(a_{ctx}\mid x)\big) + \exp\!\big(s(a_{par}\mid x)\big)}$$

**Kiểm tra calibration bắt buộc:** đo entropy trung bình của phân bố token đầu
tiên ở mỗi precision. Nếu entropy dịch chuyển mạnh, nói rõ trong Limitations và
lấy $P_{\text{ctx}}$ làm metric chính thay vì $\Delta R$.

### 5.3 Evaluator

`src/eval/normalize.py` — thứ tự xử lý:

1. Strip, lowercase, bỏ dấu câu.
2. Bỏ dấu tiếng Việt (tùy chọn, có flag) để match alias.
3. Match theo alias list (exact set membership sau normalize).
4. Phân loại: `TRUE` / `FAKE` / `REFUSAL` / `OTHER`.

Refusal detection bằng pattern list, versioned.

**Validation bắt buộc:** người kiểm tra tay **200 mẫu ngẫu nhiên**, báo cáo
agreement giữa evaluator tự động và người. Nếu <95%, sửa evaluator rồi làm lại.
Con số này phải vào paper.

---

## 6. Causal XAI

### 6.1 Phương pháp chính: span-level causal ablation

Chia prompt thành các span: `instruction`, `evidence`, `question`.

Cố định $y$ = đáp án model sinh ra dưới prompt đầy đủ. Với mỗi span $s$:

$$A(s) = \log P(y \mid x) - \log P(y \mid x_{\setminus s})$$

$A(\text{evidence})$ lớn ⟹ evidence thật sự có ảnh hưởng **nhân quả** lên đáp án,
chứ không chỉ "được attention nhìn vào".

### 6.2 Control span — đừng bỏ

Xóa token thì logprob giảm dù xóa gì đi nữa. Cần baseline:
xóa một span **độ dài token tương đương** nhưng không mang thông tin (ví dụ
`evidence_irrelevant` đã cắt cho khớp độ dài). Báo cáo attribution đã hiệu chỉnh:

$$A^{*}(\text{evidence}) = A(\text{evidence}) - A(\text{control})$$

Thiếu control span thì attribution của bạn vô nghĩa và reviewer sẽ chỉ ra.

### 6.3 Margin control — phép kiểm định phủ định, BẮT BUỘC

Miễn phí (tái dùng dữ liệu scoring đã có) và là thứ có thể lật đổ toàn bộ bài.

Dùng margin baseline $|R_{\text{F16}}(x)|$ để dự đoán việc item đó có flip hay không:

- Fit logistic regression `flip ~ |R_F16|`, báo cáo AUC.
- Vẽ flip rate theo decile của $|R_{\text{F16}}|$.

**Diễn giải:**

| Kết quả | Nghĩa là |
|---|---|
| Flip **chỉ** tập trung ở margin thấp, AUC cao | Quantization không làm gì đặc biệt — chỉ là nhiễu quanh biên quyết định. **Claim chính của bài sụp**, phải viết lại thành nghiên cứu về độ nhạy biên |
| Flip xảy ra cả ở margin cao, AUC gần 0.5 | Có thay đổi mang tính cấu trúc, không quy về nhiễu được. Claim đứng vững |

Reviewer sẽ hỏi câu này. Chạy trước, ngày 12.

### 6.4 Paraphrase evidence — ngữ nghĩa hay sao chép bề mặt

Với 100 mẫu XAI, tạo `evidence_fake_para`: **giữ nguyên nội dung mệnh đề, đổi hoàn
toàn cách diễn đạt** (đổi cấu trúc câu, từ vựng, trật tự). Chạy như một điều kiện
bổ sung, so CCR và $R$ với bản gốc.

- Reliance sống sót qua paraphrase ⇒ grounding theo **ngữ nghĩa**.
- Reliance sụp ⇒ model đang **sao chép chuỗi bề mặt**.

Chỉ số: $\text{SemanticGap} = R(\text{evidence}) - R(\text{evidence\_para})$, so
sánh giữa các precision.

Cho phép một claim mang hơi hướng cơ chế **mà vẫn chỉ dùng phương pháp input-level**:
*quantization làm context grounding thoái hóa từ ngữ nghĩa xuống sao chép bề mặt*.
Không đụng lãnh địa journal.

Paraphrase phải người viết hoặc người rà — không để LLM sinh rồi dùng thẳng.

### 6.5 Dose–response — tùy chọn, hình mạnh nhất nếu kịp

Biến thiên **áp lực bằng chứng** thay vì bật/tắt:

| Mức | Evidence |
|---|---|
| 0 | không có |
| 1 | nhắc 1 lần, có hedging ("theo một số nguồn…") |
| 2 | nhắc 1 lần, khẳng định |
| 3 | nhắc 2 lần, khẳng định |
| 4 | nhắc 3 lần, khẳng định, ở đầu và cuối document |

Cho ra **đường cong liên tục theo precision** thay vì một con số flip rate. Đại
lượng cần báo cáo: **ngưỡng lật** — mức dose nhỏ nhất khiến $P_{\text{ctx}}$ vượt 0.5.
Nếu ngưỡng này dịch chuyển theo precision, đó là kết quả cỡ headline và dễ diễn
giải hơn $\Delta R$ nhiều.

Chi phí ~4 mức × 100 mẫu × mọi cấu hình.

> **BẮT BUỘC THU, TÙY CHỌN PHÂN TÍCH.**
> Dữ liệu dose–response là nguyên liệu cho **metric contribution của bản Q1**
> (ngưỡng lật / Context Override Threshold — xem §12.1). Phải chạy và lưu **trong
> đợt chạy của bài hội nghị**, kể cả khi bài hội nghị không phân tích tới.
>
> Chạy lại sau 6 tháng không chỉ tốn công: bạn sẽ không tái tạo được chính xác
> môi trường cũ (driver, lib version, kernel), nên số liệu journal và số liệu
> conference **không còn so sánh được với nhau**. Đó là chi phí không thể sửa.

Nếu tuần 3 hết thời gian, cắt phần *phân tích*, không cắt phần *thu dữ liệu*.

### 6.6 Attribution có dự đoán được flip không — định lượng hóa RQ4

Không để RQ4 ở dạng định tính. Dùng $A^{*}(\text{evidence})$ dự đoán flip ở mức item,
báo cáo AUC, và kiểm soát thêm margin (§6.3) để tách hai hiệu ứng:

```
flip ~ A*(evidence) + |R_F16|
```

Nếu $A^{*}$ vẫn có ý nghĩa sau khi kiểm soát margin, attribution đang mang thông
tin thật chứ không chỉ phản ánh độ gần biên. Miễn phí, tái dùng output ablation.

### 6.7 Mẫu

100 mẫu phân tầng, **giữ nguyên cùng 100 mẫu cho mọi model × precision × language**:
- 20 mẫu/domain.
- Trong mỗi domain, cân bằng: stable-correct / flip / failure.

### 6.8 Ranh giới với bài journal — KHÔNG vượt

**Không đưa vào bài hội nghị:** activation patching, neuron/layer attribution,
logit lens, head-level ablation.

Ba lý do: (a) đó là novelty của bản journal (§2.2); (b) llama.cpp không expose
activation, làm sẽ phải chuyển tầng A sang HF/PyTorch và **phá vỡ lập luận một
stack duy nhất**; (c) tuần 3 không đủ chỗ.

Mọi thứ ở §6.1–6.6 đều là **input-level causal** — an toàn, rẻ, và không tiêu tốn
vốn của bài sau.

### 6.9 Attention chỉ để minh họa

Một figure định tính, tối đa. Trong caption ghi rõ nó là illustrative chứ không
phải evidence. Bằng chứng chính là §6.1. Không so sánh heatmap giữa các precision.

**Bỏ hẳn Integrated Gradients** — không đáng công, và không chạy được qua
quantized weights trong llama.cpp.

---

## 7. Thống kê

Mọi cấu hình chạy trên **cùng** tập câu hỏi ⟹ dùng paired test:

| Test | Dùng cho |
|---|---|
| **McNemar** | Thay đổi đúng/sai giữa F16 và Q4 (binary, paired) |
| **Wilcoxon signed-rank** | So sánh $R$ và $P_{\text{ctx}}$ giữa các precision |
| **Paired bootstrap 95% CI** | CCR, PRR, QFR (10,000 resample ở mức item) |
| **Holm–Bonferroni** | Hiệu chỉnh khi test nhiều model × language × precision |

Kèm **effect size**, không chỉ p-value: rank-biserial correlation cho Wilcoxon,
odds ratio cho McNemar.

> Không bao giờ báo cáo một bảng phần trăm trần không có khoảng tin cậy.

---

## 8. Cấu trúc repo

```
QuantRAG/
├── configs/
│   ├── models.yaml            # model × precision, đường dẫn GGUF, sha256
│   ├── experiment.yaml        # grid: condition × mode × language
│   └── prompts/v1.yaml        # template, versioned
├── data/
│   ├── raw/counterfact/
│   ├── facts.jsonl            # 500 fact đã lọc
│   ├── facts_candidates.jsonl # 800 fact trước lọc
│   └── splits/                # known_all.txt, known_fp16.txt, unknown.txt
├── src/
│   ├── data/build_dataset.py
│   ├── data/translate_vi.py
│   ├── runner/llama_backend.py   # wrapper llama.cpp: score() và generate()
│   ├── runner/run_grid.py        # driver, resumable
│   ├── eval/normalize.py
│   ├── eval/metrics.py           # CCR, PRR, QFR, R, P_ctx
│   ├── xai/span_ablation.py
│   └── stats/tests.py
├── runs/                      # output JSONL, một file / cấu hình
├── analysis/                  # notebook + script sinh bảng, hình
├── paper/                     # LaTeX, viết từ tuần 1
└── docs/PLAN.md               # file này
```

### 8.1 Config schema

`configs/models.yaml`:

```yaml
models:
  - id: qwen2.5-0.5b
    family: qwen2.5
    params: 0.5B
    variants:
      # tầng A — stack chính, llama.cpp/GGUF
      - {backend: llamacpp, precision: F16,    path: models/qwen2.5-0.5b-f16.gguf}
      - {backend: llamacpp, precision: Q8_0,   path: models/qwen2.5-0.5b-q8_0.gguf}
      - {backend: llamacpp, precision: Q4_K_M, path: models/qwen2.5-0.5b-q4_k_m.gguf}
      - {backend: llamacpp, precision: Q3_K_M, path: models/qwen2.5-0.5b-q3_k_m.gguf}
      # tầng B — robustness check, báo cáo riêng, KHÔNG trộn bảng
      - {backend: hf_awq,   precision: AWQ4,   path: models/qwen2.5-0.5b-awq}
  # ... 1.5b, 3b, (7b nếu kịp)

runtime:
  n_ctx: 2048
  seed: 1234
  temperature: 0.0
  batch_size: 32
  device: cuda:0
```

`configs/experiment.yaml`:

```yaml
grid:
  languages:  [en, vi]
  conditions: [C0, C1, C2, C3_fake_first, C3_true_first, C4]
  modes:      [strict, truth_seeking]
  scoring:    all_variants
  generation: all_variants
output:
  save_top_k_logprobs: 20      # BẮT BUỘC — xem §8.2
  save_answer_logprobs: true
  resumable: true
  record_env: true             # gpu, driver, cuda, commit sha, lib versions
```

**Ràng buộc kiến trúc:** hai backend phải ẩn sau cùng một interface
(`score(prompt, answer) -> logprobs`, `generate(prompt) -> text`). Nếu code phân
tích phải biết backend nào đang chạy, bạn sẽ vô tình để lệch prompt template giữa
tầng A và tầng B — và mất luôn giá trị của robustness check.

### 8.2 Bốn nguyên tắc kỹ thuật — làm ngay, tiết kiệm hàng tuần sau này

1. **Lưu top-k logprob token-level cho MỌI lần chạy**, không chỉ đáp án cuối.
   Chạy lại toàn bộ grid sau 6 tháng là cực hình, và bản journal sẽ cần logprob
   cho các phân tích chưa nghĩ ra.
2. **Pipeline config-driven.** model / precision / language / condition / mode
   là tham số YAML, không hardcode. Bản journal sẽ nhân đôi số cấu hình.
3. **Resumable runner.** Ghi kết quả theo dòng JSONL, skip item đã có. Chạy CPU
   qua đêm sẽ bị gián đoạn — chấp nhận điều đó từ đầu.
4. **Log design decision + hypothesis kèm ngày tháng** vào `docs/JOURNAL.md`.
   Bản journal thêm phân tích mới sẽ dễ bị nghi HARKing; có log thì phản biện được.

---

## 9. Lịch 4 tuần

### Tuần 1 — Hạ tầng + dataset + pilot

| Ngày | Việc |
|---|---|
| 1 | Venv 3.12 trên **server**. Build llama-cpp-python CUDA. Tải + quantize GGUF 3 model × 4 precision, xác minh sha256. Ghim `requirements.lock`. |
| 2 | Backend interface chung + `llamacpp` impl: `score()` / `generate()`. Test tính đúng logprob trên ví dụ thủ công. |
| 3 | Tải `NeelNanda/counterfact-tracing` + `azhx/counterfact`. **Chạy `survey_counterfact_vi.py` trên toàn bộ** → chốt danh sách quan hệ theo độ phủ VI (§3.2). Lọc 800 candidate trong các quan hệ đó. |
| 4 | Kéo label + alias VI từ Wikidata theo QID. **Lọc tay alias** (§3.3). Viết tay ~8 bộ template VI cho các quan hệ đã chọn — không dịch 500 fact. Sinh `q_filter`/`q_eval`. |
| 5 | Chạy C0 baseline trên `q_filter` cho mọi precision → `KNOWN_ALL` / `KNOWN_FP16` / `UNKNOWN`. **Kiểm tra ngay kích thước tập VI** (§10). Chốt 500 fact. |
| 6 | **Pilot 50 fact trên toàn grid.** Đo throughput thật. **Chạy 1 cấu hình 2 lần** để đo non-determinism. Hiệu chỉnh ngân sách. |
| 7 | `hf_awq` backend + kiểm tra prompt template khớp tuyệt đối với tầng A. Sửa evaluator theo lỗi pilot. **Bắt đầu viết Method + Setup.** |

**Đầu ra:** `facts.jsonl`, backend đã test, splits, bảng pilot, §Method bản nháp.

### Tuần 2 — Chạy thí nghiệm hành vi

| Ngày | Việc |
|---|---|
| 8 | Khởi động scoring grid tầng A đầy đủ (tmux, resumable). Song song viết `metrics.py`. |
| 9–10 | Generation grid tầng A đầy đủ. Đối chiếu generate vs argmax scoring. |
| 11 | **Đệm cho bug pipeline.** Luôn có bug. Nếu không có: chạy tầng B (AWQ). |
| 12 | Tính CCR, PRR, QFR, $R$, $P_{\text{ctx}}$. Kiểm tra calibration/entropy. **Chạy margin control (§6.3) ngay hôm nay** — nếu flip quy hết về margin thấp, bạn phải biết trước khi bước sang tuần 3. |
| 13 | Validate evaluator: 200 mẫu tay, báo cáo agreement. Chạy tầng B nếu chưa xong. |
| 14 | Phân loại lỗi + phân loại flip. Bảng kết quả chính bản nháp. **Điểm quyết định:** còn slack thì thêm 7B, không thì bỏ. |

**Đầu ra:** `runs/` đầy đủ, bảng kết quả chính, biểu đồ quantization shift.

### Tuần 3 — Causal XAI + thống kê

| Ngày | Việc |
|---|---|
| 15 | Chọn 100 mẫu phân tầng. Viết `span_ablation.py` kèm control span. Viết tay paraphrase evidence (§6.4). |
| 16–17 | Chạy ablation + điều kiện paraphrase trên toàn bộ model × precision × language. |
| 18 | Tính $A^{*}$ và SemanticGap. Định lượng RQ4: `flip ~ A* + |R_F16|` (§6.6). |
| 19 | McNemar, Wilcoxon, bootstrap CI, Holm. Effect size. |
| 20 | Vẽ 4 hình chính. 3–4 case study định tính. |
| 21 | **Điểm quyết định:** còn slack thì chạy dose–response (§6.5); không thì đệm + rà bất thường. |

**Đầu ra:** kết quả XAI, bảng thống kê, hình, case study.

### Tuần 4 — Viết

| Ngày | Việc |
|---|---|
| 22–23 | Results + Analysis (Method/Setup đã viết từ tuần 1). |
| 24 | Introduction + Related Work. |
| 25 | Limitations + Ethics + Reproducibility statement. |
| 26 | Rà soát từng claim ngược về số liệu. Chạy lại mọi trường hợp bất thường. |
| 27 | Dọn code, viết README, chuẩn bị release dataset + prompt. |
| 28 | Đọc soát, format theo template hội nghị, nộp. |

**Nguyên tắc:** Method + Experimental Setup viết từ tuần 1 lúc còn nhớ chi tiết —
và viết ra sẽ lộ luôn lỗ hổng thiết kế khi còn kịp sửa.

---

## 10. Risk register

| Rủi ro | Xác suất | Tác động | Giảm thiểu |
|---|---|---|---|
| **Scope creep vì có GPU mạnh** | **Cao** | **Cao** | GPU dư không phải lý do để thêm model/ngôn ngữ/điều kiện. Bottleneck của bạn là *thời gian phân tích và viết*, không phải FLOPs. Xem §11 |
| $\Delta R$ ròng null, không significant | **Cao** | **Cao** (tăng, vì headline đã đổi sang $\Delta R$) | Đặt cược vào **interaction** (precision × mode, precision × language) và **bất đối xứng flip** true→fake vs fake→true — đều là claim có hướng, sống sót qua null ròng. Q3_K_M làm điểm stress |
| Bị scoop tiếp trong lúc làm | Trung bình | Cao | Lĩnh vực đang chạy nhanh (§2.4). Đặt alert arXiv cho `quantization + knowledge conflict`; rà lại literature ngày 20 trước khi chốt claim |
| Tập `KNOWN_ALL` tiếng Việt co quá nhỏ | Trung bình *(hạ từ Cao)* | Cao | Độ phủ label VI **đo được ngày 1**, không cần GPU (§3.2) — chọn quan hệ phủ ~100%. Phần phụ thuộc tri thức model vẫn phải đợi ngày 5 |
| Alias/dấu tiếng Việt làm sai evaluator | Cao | Cao | Seed từ Wikidata + **hai luật bắt buộc** §3.3 (thêm label EN vào alias VI; lọc mỹ từ). Validate 200 mẫu tay, báo cáo agreement |
| Bản dịch VI lệch nghĩa | **Thấp** *(hạ từ Trung bình)* | Cao | Thực thể lấy label chính thức Wikidata, không dịch. Chỉ còn ~8 bộ template phải viết tay |
| Prompt template lệch giữa tầng A và B | Trung bình | **Cao** | Interface chung (§8.1); assert byte-level rằng prompt string đưa vào hai backend là **giống hệt** |
| Non-determinism GPU làm nhiễu QFR | Trung bình | Cao | Đo ở pilot ngày 6 bằng cách chạy đôi. Nếu dao động không nhỏ hơn hẳn QFR quan sát được, kết quả vô nghĩa — phải xử lý trước khi chạy full |
| Model quá nhỏ, C0 accuracy sàn | Trung bình | Trung bình | 0.5B có thể gần như không biết gì → dùng làm điểm dưới của thang, đừng kỳ vọng |
| Q3_K_M làm 0.5B sập hoàn toàn | Cao | Thấp | Đó chính là một kết quả. Báo cáo, đừng giấu |
| Colab disconnect giữa chừng | Cao | Thấp | Chỉ dùng Colab cho việc phụ; runner resumable |

---

## 11. Nguyên tắc chống scope creep

> Failure mode phổ biến nhất của kế hoạch "hội nghị trước, journal sau" không phải
> là bài journal hỏng — mà là **tham vọng journal làm phình scope bài hội nghị**,
> rồi trượt deadline và không có bài nào cả.

Trong 4 tuần tới, mọi ý tưởng "thêm cái này cho bản journal sau" phải trả lời:
**nó có làm chậm bản hội nghị không?** Nếu có → cắt, ghi vào `docs/JOURNAL_IDEAS.md`.

Ngoại lệ duy nhất: 4 nguyên tắc kỹ thuật ở §8.2. Chúng tốn vài giờ, tiết kiệm vài tuần.

**Cạm bẫy riêng của việc có GPU 32 GB:** rất dễ nghĩ "chạy thêm cũng có mất gì đâu".
Nhưng mỗi cấu hình thêm vào đều kéo theo một dòng trong bảng, một đoạn diễn giải,
một lần hiệu chỉnh multiple-comparison, và một khả năng gặp kết quả bất thường phải
điều tra. Bottleneck của bạn là **thời gian phân tích và viết**, không phải FLOPs.

Thứ tự cắt khi trượt lịch: (1) C3 → (2) 7B → (3) tầng B AWQ → (4) Q3_K_M.
Không bao giờ cắt: EN–VI paired, hai instruction mode, control span, validate
evaluator 200 mẫu.

---

## 12. Research questions

- **RQ1.** Quantization ảnh hưởng thế nào đến accuracy và context compliance của
  small local LLM trong RAG?
- **RQ2.** Khi retrieved context mâu thuẫn với parametric knowledge, quantization
  làm mô hình nghiêng về nguồn tri thức nào?
- **RQ3.** Mức thay đổi có phụ thuộc instruction là strict-grounding hay
  truth-seeking không?
- **RQ4.** Thay đổi hành vi có tương ứng với causal attribution của evidence span không?
- **RQ5.** Ảnh hưởng của quantization đến context reliance có khác nhau giữa
  tiếng Anh và tiếng Việt không?

---

## 13. Về cách phát biểu đóng góp

Không tuyên bố "first study" khi chưa systematic search đầy đủ. Dùng:

> To our knowledge, limited work has jointly examined quantization, context–memory
> reliance, and causal input attribution in sub-4B local language models, and none
> in a paired English–Vietnamese setting.

Câu này an toàn, kiểm chứng được, và vẫn nêu đúng điểm khác biệt.

---

## 14. Checklist trước khi nộp

- [ ] Mọi arm tầng A dùng cùng runtime, cùng tokenizer, cùng template, cùng seed
- [ ] Tầng B (AWQ) báo cáo ở subsection riêng, không trộn bảng với tầng A
- [ ] Mức non-determinism đã đo và báo cáo, nhỏ hơn hẳn QFR quan sát được
- [ ] Môi trường (GPU, driver, CUDA, lib version, commit sha) có trong Reproducibility
- [ ] Mô tả chính xác k-quant, không gọi tắt "INT4" trong Method
- [ ] Kết quả báo trên `KNOWN_ALL`, có phụ lục cho `KNOWN_FP16`
- [ ] Mọi tỉ lệ có 95% CI
- [ ] Mọi p-value có effect size và đã hiệu chỉnh Holm
- [ ] CCR luôn đi kèm instruction mode
- [ ] Attribution có control span
- [ ] Attention figure có caption ghi rõ "illustrative only"
- [ ] Agreement evaluator-người (200 mẫu) có trong paper
- [ ] Kiểm tra calibration/entropy có trong Limitations
- [ ] Không có claim "first"
- [ ] Code + prompt + dataset subset đã đóng gói release
- [ ] Đã giữ lại held-out extension (domain/ngôn ngữ thêm) chưa release
