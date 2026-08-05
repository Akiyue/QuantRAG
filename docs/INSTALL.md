# Cài đặt

Có **ba lớp**, cài theo thứ tự. Lớp 1 chạy được mọi thứ trừ inference thật.

| Lớp | Gồm | Bắt buộc? |
|---|---|---|
| 1. Phân tích | numpy, scipy, pandas, statsmodels, matplotlib | Có |
| 2. Tầng A | `llama-cpp-python` build CUDA | Có, để chạy model |
| 3. Tầng B | torch + transformers + autoawq | Không, chỉ để robustness check |

---

## Lớp 1 — môi trường conda

```bash
conda create -n quantrag python=3.12 -y
conda activate quantrag

cd /path/to/QuantRAG
pip install -e ".[dev]"
pytest -q                    # phải xanh 49/49 trước khi đi tiếp
```

> **Python 3.12 hoặc 3.11, không dùng 3.13+.** `torch`, `llama-cpp-python`,
> `autoawq` chưa có wheel. `run.sh` sẽ chặn và báo rõ nếu bạn dùng sai phiên bản.

`run.sh` tự nhận môi trường conda đang active — không cần đặt biến gì. Kiểm tra:

```bash
./run.sh status
```

---

## Lớp 2 — llama.cpp CUDA (tầng A, stack chính)

Kiểm tra CUDA của server trước:

```bash
nvidia-smi | head -4        # xem dòng "CUDA Version: 12.x"
```

### Cách A — wheel dựng sẵn (thử cách này trước)

Nhanh nhất, không cần compiler. Chọn tag khớp CUDA của bạn (`cu121`, `cu122`, `cu124`, `cu125`):

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Xác minh nó thật sự thấy GPU:

```bash
python -c "from llama_cpp import Llama; import llama_cpp; print(llama_cpp.__version__)"
```

### Cách B — build từ nguồn

Nếu không có wheel khớp:

```bash
# toolchain, cài vào chính env conda để không đụng hệ thống
conda install -c conda-forge cmake ninja gcc_linux-64 gxx_linux-64 -y
conda install -c nvidia cuda-toolkit=12.4 -y      # khớp với nvidia-smi

CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
  --no-binary llama-cpp-python --no-cache-dir
```

Build mất 10–20 phút. Nếu lỗi `nvcc not found`, `CUDA_HOME` chưa trỏ đúng:

```bash
export CUDA_HOME="$CONDA_PREFIX"
```

### llama.cpp binary — để lượng tử hóa model

`./run.sh models` cần `llama-quantize`, là chương trình riêng chứ không phải
Python:

```bash
git clone https://github.com/ggerganov/llama.cpp ../llama.cpp
cmake -B ../llama.cpp/build -DGGML_CUDA=ON ../llama.cpp
cmake --build ../llama.cpp/build -j
```

Nếu để chỗ khác thì báo cho `run.sh`:

```bash
export LLAMA_CPP=/duong/dan/toi/llama.cpp
```

Cũng cần `huggingface-cli` để tải model:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli login       # nếu repo yêu cầu
```

---

## Lớp 3 — tầng B, AWQ (tùy chọn)

Chỉ để chứng minh hiệu ứng không phải artifact của riêng k-quant. **Cắt được nếu
trượt lịch** — mọi kết quả chính đều nằm ở tầng A.

```bash
# torch khớp CUDA; đổi cu124 theo server
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[hf]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Kiểm tra toàn bộ

```bash
conda activate quantrag
cd /path/to/QuantRAG

./run.sh status                                   # môi trường + cổng
pytest -q                                         # 49 test
python scripts/survey_counterfact_vi.py 300       # mạng + Wikidata, không cần GPU
python scripts/run_grid.py --pass main --mock \
    --facts data/facts.sample.jsonl --out-dir runs/tmp   # pipeline không cần model
rm -rf runs/tmp
```

Bốn lệnh này xanh là mọi thứ trừ inference thật đã sẵn sàng.

---

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `Python 3.13 at ...` | `conda create -n quantrag python=3.12` rồi activate |
| `no Python found` | Chưa `conda activate quantrag` |
| `llama-cpp-python` build fail, `nvcc not found` | `export CUDA_HOME="$CONDA_PREFIX"`, hoặc `conda install -c nvidia cuda-toolkit` |
| Model chạy nhưng chậm như CPU | Wheel không có CUDA. Cài lại theo Cách A với tag đúng, hoặc build lại với `CMAKE_ARGS` |
| `llama-quantize not built` | Chưa build binary llama.cpp, hoặc `LLAMA_CPP` trỏ sai |
| `missing command: huggingface-cli` | `pip install "huggingface_hub[cli]"` |
| pip đè lên gói conda | Bình thường ở đây; chỉ cài `pip` **sau khi** đã activate env |

> **Đừng trộn `conda install` và `pip install` cho cùng một gói.** Ở dự án này:
> conda chỉ lo Python và toolchain CUDA, pip lo toàn bộ gói Python.
