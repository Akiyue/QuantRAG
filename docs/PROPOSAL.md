# Research Proposal — QuantRAG

| | |
|---|---|
| **Tiêu đề** | Does Quantization Change What Small Language Models Trust? Measuring Context–Memory Arbitration in Local RAG |
| **Tiêu đề dài (dự phòng)** | Quantization-Induced Shifts in Context–Memory Arbitration of Small Local Language Models: A Bilingual Causal Analysis |
| **Venue đích** | AICI 2027 (conference, ~8 trang) |
| **Venue tiếp theo** | Journal Q1, bản mở rộng (cơ chế + mitigation) — xem §11 |
| **Thời gian** | 4 tuần |
| **Trạng thái** | Proposal v1 — 04/08/2026 |
| **Tài liệu liên quan** | `docs/PLAN.md` (kế hoạch triển khai chi tiết) |

---

## 1. Draft Abstract *(bản tiếng Anh, có thể bê thẳng vào paper)*

> Quantization is the default enabling technology for running language models
> locally, yet its evaluation still relies almost entirely on aggregate accuracy.
> Recent work shows that aggregate metrics can remain stable while individual
> model responses change substantially, and separate lines of work show that
> quantization degrades parametric factual recall and that small models often
> fail to exploit retrieved evidence. What remains unmeasured is the *arbitration*
> itself: when retrieved context contradicts what a model learned during training,
> does quantization change which source the model follows?
>
> We introduce a controlled protocol that separates three quantities usually
> conflated in RAG evaluation — factual accuracy, context compliance, and
> parametric knowledge retention — and apply it to sub-4B instruction-tuned
> models across a four-level precision ladder (F16, Q8_0, Q4_K_M, Q3_K_M) within
> a single inference stack. Using 500 factual triples paired with type-matched
> counterfactual evidence, evaluated under both strict-grounding and truth-seeking
> instructions and in both English and Vietnamese, we measure a knowledge
> reliance score and its shift under quantization, and we verify behavioural
> findings with span-level causal ablation rather than attention weights.
>
> [RESULTS PLACEHOLDER]
>
> Our results indicate that evaluating quantized models for retrieval-augmented
> deployment requires reporting compliance and retention separately, under an
> explicitly stated instruction regime.

*(Đoạn RESULTS viết sau tuần 3. Không viết trước — xem §8.)*

---

## 2. Vấn đề

Triển khai LLM local hầu như luôn đi kèm lượng tử hóa: Ollama, LM Studio,
llama.cpp đều phục vụ model đã nén. Đồng thời, cách dùng phổ biến nhất của các
model này là RAG — đưa tài liệu truy xuất vào context rồi yêu cầu trả lời dựa
trên đó.

Hai điều kiện đó gặp nhau ở một chỗ chưa được đo: **khi tài liệu truy xuất mâu
thuẫn với tri thức model đã học, model tin ai — và quantization có làm đổi câu
trả lời đó không?**

Đây không phải câu hỏi học thuật thuần túy. Một hệ RAG local dùng cho tra cứu nội
bộ có hai chế độ hỏng ngược nhau:

- Model **quá tin context** → một tài liệu sai hoặc bị nhiễm độc sẽ điều khiển
  đầu ra, kể cả khi model "biết" đáp án đúng.
- Model **quá tin parametric memory** → tài liệu cập nhật bị bỏ qua, hệ thống trả
  lời theo dữ liệu huấn luyện đã cũ.

Người triển khai chọn mức lượng tử hóa dựa trên accuracy và bộ nhớ. Nếu mức lượng
tử hóa **cũng** dịch chuyển cán cân giữa hai chế độ hỏng đó, thì đó là một hệ quả
vận hành hiện đang vô hình trong mọi bảng benchmark.

---

## 3. Khoảng trống nghiên cứu

Ba mảng đã có nghiên cứu riêng, nhưng chưa giao nhau.

| Mảng | Đại diện | Đã làm | Chưa làm |
|---|---|---|---|
| Quantization × hành vi | *The Illusion of Equivalency in Quantization* (2607.08734) | Chỉ ra accuracy ổn định trong khi đáp án từng item đổi nhiều; định nghĩa *correctness agreement* | Không RAG, không xung đột tri thức, không attribution, không đa ngữ |
| Quantization × tri thức | *Through a Compressed Lens* (2505.13963) | Quantization làm suy giảm parametric factual recall; neuron-level attribution | Chỉ tri thức trong tham số, **không có context**; ≥7B; English-only |
| Quantization × RAG | *Impact of Quantization on RAG* (2406.10251) | 7–8B, FP16 vs INT4, RAG vẫn dùng được sau lượng tử hóa | Không có context mâu thuẫn; không tách compliance khỏi retention |
| Xung đột tri thức | *Does RAG Know When Retrieval Is Wrong?* (2605.14473) | Context compliance dưới knowledge conflict, có causal perturbation | **Không có quantization**; chạy trên model frontier, không phải small local |
| Quantization × đa ngữ | 2407.03211 vs 2503.03592 | **Hai kết luận trái nhau**: một bên nói hại non-Latin nặng, bên kia nói k-quant không hại đa ngữ bất cân xứng | Không RAG, không xung đột |

**Khoảng trống:** không công trình nào đo *sự phân xử* giữa context và parametric
memory như một hàm của precision. Cộng thêm hai trục chưa được phủ: **sub-4B**
(mọi nghiên cứu quantization nêu trên đều ≥7B) và **tiếng Việt**.

> **Cách phát biểu an toàn trong paper:**
> To our knowledge, no prior work has jointly examined quantization, context–memory
> arbitration, and causal input attribution in sub-4B language models, or tested
> whether such effects differ across languages on paired factual content.

Không dùng "first study" khi chưa systematic search đầy đủ.

---

## 4. Research questions

**RQ1.** Quantization ảnh hưởng thế nào đến accuracy và context compliance của
small local LLM trong RAG?

**RQ2 *(trung tâm)*.** Khi retrieved context mâu thuẫn với parametric knowledge,
quantization có làm dịch chuyển sự phân xử giữa hai nguồn tri thức không, và theo
hướng nào?

**RQ3.** Mức dịch chuyển có phụ thuộc vào việc instruction là strict-grounding hay
truth-seeking không?

**RQ4.** Thay đổi hành vi có tương ứng với thay đổi trong causal attribution của
evidence span không — và có còn ý nghĩa sau khi kiểm soát margin ở baseline không?

**RQ5.** Ảnh hưởng có khác nhau giữa tiếng Anh và tiếng Việt trên cùng tập fact không?

---

## 5. Giả thuyết *(phát biểu trước khi chạy — chống HARKing)*

Ghi ngày và commit vào repo trước ngày 8. Giả thuyết sai không phải vấn đề; giả
thuyết viết sau khi thấy số liệu mới là vấn đề.

| # | Giả thuyết | Lý lẽ | Bác bỏ khi |
|---|---|---|---|
| **H1** | Accuracy ở C1 (context đúng) giữ trong biên hẹp từ F16→Q4_K_M; chỉ suy giảm rõ ở Q3_K_M | Khớp với 2406.10251; sao chép từ context là thao tác nông | Suy giảm đáng kể ngay ở Q8_0 |
| **H2a** | Dưới xung đột (C2), precision thấp dịch chuyển phân xử **về phía context** ($\Delta R > 0$) | Quantization làm hỏng parametric recall (2505.13963) mạnh hơn làm hỏng khả năng sao chép từ context | $\Delta R \leq 0$ |
| **H2b** *(cạnh tranh)* | Precision thấp dịch chuyển **về phía parametric memory** ($\Delta R < 0$) | Khả năng làm theo instruction xuống cấp trước, model rơi về prior | $\Delta R \geq 0$ |
| **H3** | Khoảng cách giữa hai instruction mode **thu hẹp** khi precision giảm | Instruction-following mong manh hơn nội dung; model nén ít phân biệt được hai yêu cầu | Khoảng cách giữ nguyên hoặc rộng ra |
| **H4** | Dịch chuyển lớn hơn ở tiếng Việt so với tiếng Anh | Theo 2407.03211 | Không khác biệt — ủng hộ 2503.03592 |
| **H5** | $A^{*}(\text{evidence})$ thay đổi cùng hướng với dịch chuyển hành vi | Nếu behavioral shift là thật, nó phải có dấu vết nhân quả | Attribution không đổi trong khi hành vi đổi ⇒ hành vi đổi vì lý do khác |
| **H6** | SemanticGap **tăng** khi precision giảm: model nén phụ thuộc bề mặt của evidence nhiều hơn | Grounding ngữ nghĩa cần biểu diễn tinh hơn sao chép chuỗi | SemanticGap không đổi hoặc giảm |
| **H7** *(phủ định)* | Flip **không** quy hết về margin thấp — AUC của `flip ~ \|R_F16\|` rõ rệt dưới mức bão hòa | Nếu quy hết được, hiệu ứng chỉ là nhiễu quanh biên quyết định | AUC rất cao ⇒ **claim chính của bài sụp**, phải viết lại thành nghiên cứu độ nhạy biên |

H2a và H2b **cạnh tranh có chủ ý**. Nêu cả hai trong proposal là điểm mạnh: bất kỳ
kết quả nào cũng có ý nghĩa, và không ai buộc được bạn tội chọn giả thuyết sau khi
thấy dữ liệu.

---

## 6. Phương pháp

### 6.1 Dữ liệu

500 factual triple $(s, r, o_{\text{true}}, o_{\text{fake}})$ lấy nền từ CounterFact,
5 domain × 100 (geography, science, history, culture, technology). `o_fake` **cùng
semantic type** với `o_true`. Dịch sang tiếng Việt và rà tay 100% → **paired
500 × 2 ngôn ngữ**, nên mọi khác biệt giữa hai ngôn ngữ không bị nhiễu bởi nội dung.

**Lọc parametric-known** (điểm quan trọng về thống kê): mỗi fact có hai paraphrase
độc lập, `q_filter` để lọc và `q_eval` để đánh giá. Tập phân tích chính là
`KNOWN_ALL` = model trả lời đúng ở **mọi** precision khi không có context. Lọc bằng
riêng F16 rồi so F16 với Q4 trên chính tập đó là *selection on the baseline arm* và
sẽ tạo ra hiệu ứng giả.

### 6.2 Điều kiện

| | Context | Đo |
|---|---|---|
| **C0** | không có | Parametric knowledge nền |
| **C1** | evidence đúng | Sử dụng retrieval thông thường |
| **C2** | evidence sai | **Điều kiện chính** — xung đột |
| **C3** | cả hai tài liệu, 2 thứ tự | Xung đột giữa tài liệu + position bias |
| **C4** | evidence không liên quan | Control: model có nhại context bừa không |

× hai **instruction mode**:
- **Strict grounding** — "chỉ dùng tài liệu, kể cả khi trái với điều đã biết"
- **Truth-seeking** — "dùng tài liệu làm tham khảo nhưng ưu tiên đáp án đúng; cảnh báo nếu tài liệu sai"

Hai mode này là điểm sắc nhất của thiết kế: chúng tách *khả năng làm theo instruction*
khỏi *mức phụ thuộc context*. Cùng một CCR mang hai ý nghĩa trái ngược tùy mode —
nên **không bao giờ báo cáo CCR mà thiếu mode**.

### 6.3 Model và precision

Qwen2.5-Instruct **0.5B / 1.5B / 3B**, ladder **F16 → Q8_0 → Q4_K_M → Q3_K_M**,
toàn bộ trong **một stack duy nhất** (llama.cpp/GGUF, CUDA). Một runtime, một
tokenizer, một sampler — khác biệt duy nhất giữa các arm là số bit.

Bổ sung **AWQ 4-bit** (họ quantizer thứ hai) làm robustness check, báo cáo ở
subsection riêng: chứng minh hiệu ứng không phải artifact của một phương pháp
lượng tử hóa cụ thể.

*Lưu ý thuật ngữ:* Q4_K_M / Q3_K_M là k-quant hỗn hợp, không phải uniform INT4/INT3.
Mô tả chính xác trong Experimental Setup, không gọi tắt "INT4" trong Method.

### 6.4 Chỉ số

Đo bằng **teacher-forced scoring** (logprob của candidate answer bị ép,
length-normalized) — đại lượng liên tục, không phụ thuộc quirk của decoding — cộng
free generation để bắt refusal và đáp án ngoài hai candidate.

- **CCR** — Context Compliance Rate, tỉ lệ theo context ở C2
- **PRR** — Parametric Retention Rate, tỉ lệ giữ đáp án đúng ở C2
- **QFR** — Quantization Flip Rate, có phân loại (true→fake / fake→true /
  answer→refusal / answer→other). Báo cáo như **mở rộng** của *correctness
  agreement* (2607.08734) sang bối cảnh xung đột, có trích dẫn.
- **KRS** — $R = s(a_{ctx}) - s(a_{par})$, và $\Delta R_b = R_b - R_{\text{F16}}$.
  **Metric trung tâm.**
- Kèm dạng chuẩn hóa 2 lựa chọn $P_{\text{ctx}} = \sigma(R)$, ít nhạy với thay đổi
  calibration do quantization gây ra (mối lo này được 2508.16785 xác nhận là có thật).

### 6.5 Giải thích nhân quả

**Span-level causal ablation**, không phải attention. Cố định $y$ = đáp án dưới
prompt đầy đủ, xóa từng span (instruction / evidence / question), đo sụt logprob:

$$A(s) = \log P(y \mid x) - \log P(y \mid x_{\setminus s})$$

Trừ đi **control span** cùng độ dài token nhưng vô thông tin:
$A^{*} = A(\text{evidence}) - A(\text{control})$. Thiếu control span thì attribution
vô nghĩa — xóa token nào cũng làm logprob giảm.

Ba phân tích bổ trợ, tất cả đều ở **tầng input-level causal**:

**(a) Margin control — phép kiểm định phủ định.** Dùng margin baseline
$|R_{\text{F16}}|$ để dự đoán flip. Nếu flip quy hết về các item gần biên quyết
định, quantization không làm gì đặc biệt và claim chính của bài sụp. Báo cáo AUC
và flip rate theo decile của margin. Miễn phí, tái dùng dữ liệu đã có.

**(b) Paraphrase evidence — ngữ nghĩa hay sao chép bề mặt.** Giữ nguyên nội dung
mệnh đề, đổi hoàn toàn cách diễn đạt. Reliance sống sót ⇒ grounding ngữ nghĩa;
reliance sụp ⇒ sao chép chuỗi. Cho phép một claim mang hơi hướng cơ chế mà vẫn chỉ
dùng phương pháp input-level: *quantization làm context grounding thoái hóa từ ngữ
nghĩa xuống sao chép bề mặt*.

**(c) Dose–response** *(tùy chọn, chạy nếu tuần 3 còn slack)*. Biến thiên áp lực
bằng chứng (số lần nhắc lại × mức độ hedging) thay vì bật/tắt, cho ra đường cong
liên tục theo precision. Đại lượng báo cáo: **ngưỡng lật** — mức bằng chứng nhỏ
nhất khiến $P_{\text{ctx}}$ vượt 0.5. Ngưỡng này dịch theo precision là kết quả cỡ
headline và dễ diễn giải hơn $\Delta R$.

RQ4 định lượng hóa, không để định tính: `flip ~ A*(evidence) + |R_F16|`, báo cáo
AUC. Nếu $A^{*}$ vẫn có ý nghĩa sau khi kiểm soát margin, attribution mang thông
tin thật chứ không chỉ phản ánh độ gần biên.

Attention chỉ dùng cho **một** hình minh họa định tính, caption ghi rõ là illustrative.
Attention weight lớn không chứng minh quan hệ nhân quả, và phân bố attention không
so sánh trực tiếp được giữa các precision.

100 mẫu phân tầng, giữ nguyên cùng 100 mẫu cho mọi cấu hình.

**Ranh giới có chủ ý:** không dùng activation patching, neuron/layer attribution
hay logit lens ở bài này. Đó là novelty của bản journal (§12), và chúng đòi chuyển
tầng A sang HF/PyTorch, phá vỡ lập luận một-stack-duy-nhất.

### 6.6 Thống kê

Mọi cấu hình chạy trên cùng tập câu hỏi ⇒ paired test: **McNemar** (đúng/sai),
**Wilcoxon signed-rank** ($R$, $P_{\text{ctx}}$), **paired bootstrap 95% CI**
(CCR/PRR/QFR, 10k resample), **Holm** hiệu chỉnh đa so sánh. Kèm effect size, không
chỉ p-value. Không có bảng phần trăm nào thiếu khoảng tin cậy.

---

## 7. Đóng góp dự kiến

1. **Protocol** tách bạch factual accuracy / context compliance / parametric
   retention — ba đại lượng thường bị gộp trong đánh giá RAG — dưới hai chế độ
   instruction đối lập.
2. **Đo lường sự phân xử** context ↔ memory như một hàm của precision trên sub-4B,
   vùng model chưa được nghiên cứu quantization nào phủ.
3. **Kiểm chứng nhân quả** bằng span ablation có control span và margin control,
   thay vì suy diễn từ attention — kèm phép thử paraphrase phân biệt grounding
   ngữ nghĩa với sao chép chuỗi bề mặt.
4. **Bằng chứng song ngữ paired EN–VI** trên cùng nội dung fact, kiểm định trực tiếp
   một tranh cãi đang mở giữa 2407.03211 và 2503.03592.
5. **Dataset + code + prompt** công khai, tái lập được không cần GPU.

---

## 8. Kết quả kỳ vọng và kịch bản null

Viết phần này **trước** khi có số liệu, để không bị cám dỗ diễn giải ngược.

| Kịch bản | Xác suất | Bài vẫn đứng được vì |
|---|---|---|
| $\Delta R$ dịch rõ về context | Trung bình | Kết quả trực tiếp, có hệ quả vận hành: model nén dễ bị tài liệu nhiễm độc điều khiển hơn |
| $\Delta R$ dịch về parametric | Trung bình | Ngược lại, cũng có hệ quả: model nén bỏ qua tài liệu cập nhật |
| **$\Delta R$ ròng ≈ 0** | **Cao** | Rơi về **interaction** và **bất đối xứng flip** — xem dưới |
| Mọi thứ null kể cả interaction | Thấp | Null result có CI chặt trên một câu hỏi chưa ai hỏi, cộng protocol tái sử dụng được. Yếu hơn, nhưng vẫn nộp được ở venue này |

**Chỗ tín hiệu nhiều khả năng nằm nếu hiệu ứng ròng null:**
- precision × instruction mode (H3)
- precision × language (H4)
- **bất đối xứng** true→fake so với fake→true — claim *có hướng*, sống sót ngay cả
  khi tổng số flip không đổi

Nguyên tắc: nếu H2a và H2b đều bị bác, **báo cáo đúng như thế**. Một null result
trung thực với khoảng tin cậy chặt vẫn tốt hơn một hiệu ứng nặn ra từ subgroup.

---

## 9. Phạm vi và giới hạn *(viết sẵn cho mục Limitations)*

- Gold document tự tạo, không phải retrieval thật — cô lập được biến số nhưng chưa
  chứng minh chuyển sang deployment. → bản journal.
- Fact ngắn, một thực thể; không phủ multi-hop hay câu hỏi mở.
- Một model family; hiệu ứng có thể phụ thuộc kiến trúc và corpus huấn luyện.
- So sánh chéo stack (GGUF vs AWQ) chỉ mang tính định hướng, không phải so sánh có
  kiểm soát.
- k-quant là hỗn hợp, bit-width hiệu dụng khác con số danh nghĩa.
- Tiếng Việt: tập `KNOWN_ALL` có thể nhỏ hơn tiếng Anh đáng kể; nếu vậy nhánh VI
  hạ xuống phân tích phụ và nói rõ.
- Quantization làm đổi calibration; đó là lý do báo cáo $P_{\text{ctx}}$ song song
  với $\Delta R$.

---

## 10. Kế hoạch 4 tuần *(chi tiết ở `docs/PLAN.md`)*

| Tuần | Nội dung | Đầu ra |
|---|---|---|
| **1** | Hạ tầng, dataset 500 fact EN+VI, lọc parametric-known, pilot 50 mẫu. **Đọc toàn văn 4 paper 🔴.** Bắt đầu viết Method | `facts.jsonl`, splits, bảng pilot, §Method nháp |
| **2** | Chạy toàn bộ grid (scoring + generation), tính CCR/PRR/QFR/$R$, validate evaluator 200 mẫu tay | Bảng kết quả chính |
| **3** | Span ablation 100 mẫu có control, đối chiếu attribution với flip, thống kê paired, 4 hình chính | Kết quả XAI + thống kê |
| **4** | Results, Intro, Related Work, Limitations, rà claim, release code | Bản nộp |

Method + Experimental Setup viết từ tuần 1 — lúc còn nhớ chi tiết, và viết ra sẽ lộ
lỗ hổng thiết kế khi còn kịp sửa.

---

## 11. Đề cương paper *(giả định 8 trang — chỉnh theo CFP)*

| § | Nội dung | Trang |
|---|---|---|
| 1 | Introduction — vấn đề vận hành, hai chế độ hỏng, đóng góp | 1.0 |
| 2 | Related Work — **có bảng phân biệt** với 5 paper gần nhất | 1.0 |
| 3 | Protocol — điều kiện, instruction mode, chỉ số | 1.5 |
| 4 | Experimental Setup — model, k-quant, stack, dataset, lọc known | 1.0 |
| 5 | Results — RQ1–RQ3, bảng chính, hình dịch chuyển | 1.5 |
| 6 | Causal Attribution — RQ4 | 0.75 |
| 7 | Bilingual Analysis — RQ5 | 0.5 |
| 8 | Robustness (AWQ), Limitations, Conclusion | 0.75 |

Bảng phân biệt ở §2 là bắt buộc. Reviewer sẽ hỏi "khác gì X?" cho ít nhất ba paper.

---

## 12. Lộ trình lên journal Q1

**Cố ý không đưa vào bài hội nghị**, để bản mở rộng có đủ nội dung mới. Cấu trúc
đích: **metric contribution + method contribution + validation**, không phải
"conference paper + thêm thí nghiệm".

### 12.1 Đóng góp về metric — chọn MỘT

Bar cho một metric mới cao hơn người ta tưởng. Nó chỉ được tính là đóng góp khi:
(a) đo được thứ metric hiện có **chứng minh được là không đo nổi**, (b) có quy
trình ước lượng kèm khoảng tin cậy, (c) dẫn tới một quyết định người ta làm khác đi.
Thiếu (c) thì chỉ là thêm một cột trong bảng. **Đề xuất một, không đề xuất ba.**

| Ứng viên | Vì sao mạnh | Rủi ro |
|---|---|---|
| **Context Override Threshold** — lượng áp lực bằng chứng tối thiểu để model bỏ tri thức nội tại, ước lượng bằng logistic fit trên dose–response, báo ED50 + CI | Không quy về CCR/accuracy được (hai cái đó đo tại **một** mức dose cố định). Khung ước lượng mượn từ dose–response dược lý nên vững về thống kê. Có nghĩa vận hành trực tiếp: hệ RAG chịu được bao nhiêu áp lực nhiễu độc trước khi đầu hàng | Cần dữ liệu dose–response — **phải thu từ giai đoạn conference** (§6.5) |
| **Phân rã $\Delta R$** thành thành phần calibration và thành phần preference | Mọc thẳng ra từ limitation mà bài hội nghị ghi nhận. Cung liên kết sạch nhất giữa hai bài: không phải "thêm thí nghiệm" mà là "giải quyết vấn đề bài trước nêu" | Cần chứng minh phân rã là well-defined, không chỉ là hai số cộng lại |

Cả hai đều cần một **mục validation riêng**, không chỉ một định nghĩa: chứng minh
metric mới dự đoán được thứ mà metric cũ không dự đoán được.

### 12.2 Đóng góp về method — một cặp, không phải một cái

**Chính: reliance-aware mixed-precision.** Dùng activation patching F16 ↔ Q4 theo
layer để xác định nhóm layer/head chịu trách nhiệm cho dịch chuyển arbitration,
giữ riêng nhóm đó ở precision cao.

**Baseline bắt buộc phải đánh bại: calibration repair.** Nếu dịch chuyển chủ yếu
là calibration, một hiệu chỉnh temperature/logit fit trên vài chục mẫu held-out có
thể khôi phục hành vi với chi phí gần bằng không.

Điểm hay: **cả hai kết quả đều publishable.**
- Calibration repair *chạy được* → kết quả thực dụng bất ngờ: không cần mixed
  precision, chỉ cần ~20 ví dụ hiệu chỉnh
- Calibration repair *không chạy* → chứng minh dịch chuyển là **cấu trúc**, biện
  minh cho method nặng hơn

Dù sao reviewer cũng sẽ đòi baseline này. Chạy sớm thì nó thành một nhánh kết quả
thay vì một yêu cầu major revision.

> **Không thương lượng:** so sánh phải ở **cùng ngân sách bit trung bình**.
> Mixed-precision thắng uniform 4-bit trong khi tốn 4.6 bit là so sánh gian lận,
> và đó là câu đầu tiên reviewer hỏi.

**Plan B:** nếu cả hai method đều không khôi phục được hành vi, báo cáo negative
result kèm phần định vị nguyên nhân. Yếu hơn, nhưng vẫn là một bài — với điều kiện
phần diagnosis đủ sâu.

### 12.3 Phần còn lại

3. **RAG thật** — BM25 + dense trên Wikipedia EN/VI, top-k thật, noise thật,
   multi-hop. Chứng minh phát hiện chuyển được sang deployment.
4. Mở rộng model family, ngôn ngữ, thang size.

Cần HF/PyTorch cho activation patching (llama.cpp không expose activation) — đây là
lý do backend tầng B ở giai đoạn conference cũng là **đầu tư hạ tầng cho bản journal**,
không chỉ là robustness check.

Diagnosis + metric + mitigation là cấu trúc kinh điển của bài journal mạnh.
Measurement-only rất khó qua Q1.

**Phải xác minh trước khi nộp bản hội nghị:** proceedings AICI có index không (quyết
định luật extension ≥30% nội dung mới), và chính sách extended-paper của 2–3 journal đích.

---

## 13. Rủi ro chính

| Rủi ro | Đối phó |
|---|---|
| **Bị scoop tiếp** — lĩnh vực đang chạy nhanh, riêng 2026 đã ≥4 paper liên quan | Alert arXiv cho `quantization + knowledge conflict`; rà lại literature ngày 20 trước khi chốt claim |
| $\Delta R$ ròng null | Đã có kế hoạch: interaction + bất đối xứng flip (§8) |
| Tập VI co quá nhỏ | Đo ngay ngày 5; nếu <150 item, hạ VI xuống phân tích phụ và nói rõ |
| Alias/dấu tiếng Việt làm sai evaluator | Validate 200 mẫu tay, báo cáo agreement trong paper |
| Non-determinism GPU làm nhiễu QFR | Chạy đôi một cấu hình ở pilot; nếu dao động không nhỏ hơn hẳn QFR quan sát được thì kết quả vô nghĩa |
| Scope creep vì GPU dư | Bottleneck là thời gian phân tích và viết, không phải FLOPs. Thứ tự cắt: C3 → 7B → AWQ → Q3_K_M |

---

## 14. Tài liệu tham khảo chính

⚠️ Khảo sát ngày 04/08/2026, **chưa systematic**. Bốn mục 🔴 phải đọc toàn văn
trong tuần 1 — nếu 2607.08734 đã có sẵn phần RAG hoặc conflict mà abstract không
nêu, phải xoay trục lần nữa.

- 🔴 *The Illusion of Equivalency in Quantization*, arXiv 2607.08734 — **đọc trước tiên**
- 🔴 *Through a Compressed Lens: The Impact of Quantization on Factual Knowledge Recall*, arXiv 2505.13963
- 🔴 *Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict*, arXiv 2605.14473
- 🔴 *English K-Quantization Does Not Disproportionately Diminish Multilingual Performance*, arXiv 2503.03592
- *Interpreting the Effects of Quantization on LLMs*, arXiv 2508.16785
- *The Impact of Quantization on RAG: An Analysis of Small LLMs*, arXiv 2406.10251
- *How Does Quantization Affect Multilingual LLMs?*, arXiv 2407.03211
- *Compressed Causal Reasoning: Quantization and GraphRAG Effects*, arXiv 2512.13725
- *Task Matters: Knowledge Requirements Shape LLM Responses to Context–Memory Conflict*, arXiv 2506.06485
- *Knowledge Conflicts for LLMs: A Survey*, EMNLP 2024
- ConflictBank (NeurIPS 2024), CounterFact, FaithEval, WikiContradict — nguồn dataset
- *Attention is not Explanation* — cơ sở cho lựa chọn không dùng attention làm bằng chứng
