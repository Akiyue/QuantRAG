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

Kiểm tra driver trước:

```bash
nvidia-smi | head -4
```

> **Con số "CUDA Version" ở đây là trần driver hỗ trợ, không phải bản bạn phải
> cài.** CUDA tương thích ngược: driver 580 (CUDA 13.0) chạy tốt binary build cho
> 12.x. Đừng đi tìm wheel `cu130` — không có, và cũng không cần.

### Cách nhanh nhất: để script tự chọn

```bash
bash scripts/install_llamacpp.sh
```

Nó kiểm tra `nvcc` rồi chọn đường phù hợp, và tự chạy `check_gpu.py` ở cuối.
Phần dưới giải thích nó chọn thế nào và vì sao.

### Cân nhắc: wheel hay build từ nguồn?

| | Wheel dựng sẵn | Build từ nguồn |
|---|---|---|
| Tải về | **1.3–1.8 GB** | ~70 MB |
| Cần `nvcc` | Không | Có |
| Thời gian (mạng nhanh) | vài phút | 15–25 phút |
| Thời gian (mạng 100 kB/s) | **4–5 giờ** | ~35 phút |

Wheel to như vậy vì nó gói kernel cho **mọi** kiến trúc CUDA. Build từ nguồn chỉ
biên dịch cho đúng GPU của bạn.

> **pip không resume được.** Đứt mạng giữa chừng là mất sạch và phải tải lại từ
> đầu — đúng cái xảy ra với wheel 1.8 GB trên đường truyền chậm. Script dùng
> `wget -c` nên ngắt rồi chạy lại là nó đi tiếp.

### Cách A — build từ nguồn (nếu có `nvcc`)

```bash
nvcc --version                    # có kết quả thì dùng cách này
conda install -c conda-forge cmake ninja -y

CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" \
  pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir
```

`89` là compute capability của Ada Lovelace (RTX 5000 Ada, 4090, L40S).
Ampere A100 dùng `80`, A6000/3090 dùng `86`, H100 dùng `90`.

### Cách B — wheel, tải có resume

Index chỉ có `cu121`–`cu124`; `cu125` trả 404. Dùng `cu124`:

```bash
tmux new -s wheel                 # 4-5 tiếng, đừng để đứt SSH
mkdir -p ~/.cache/quantrag && cd ~/.cache/quantrag
wget -c --tries=0 --read-timeout=30 \
  https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-cu124/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl

pip install --no-deps llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl
pip install diskcache jinja2 numpy typing-extensions
```

**Xác minh nó thật sự chạm GPU** — bước này không được bỏ:

```bash
python scripts/check_gpu.py
```

Bản CPU-only cài sạch sẽ và cho số liệu đúng, chỉ chậm hơn khoảng 50 lần. Lỗi này
im lặng, và trên grid 96,000 cell nó là khác biệt giữa một đêm và hai tuần.

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
Python.

**Phần này không cần CUDA.** Lượng tử hóa là thao tác CPU thuần: đọc GGUF F16,
ghi ra GGUF đã nén. Build không CUDA nhanh hơn và tránh hẳn việc phải có `nvcc`:

```bash
git clone https://github.com/ggerganov/llama.cpp ../llama.cpp
cmake -B ../llama.cpp/build ../llama.cpp
cmake --build ../llama.cpp/build -j
```

Inference vẫn đi qua `llama-cpp-python` ở trên, và bản đó *có* CUDA.

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
python scripts/check_gpu.py                       # backend có chạm GPU không
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
| `check_gpu.py` báo `gpu offload: False` | Wheel không có CUDA. Cài lại theo Cách A, hoặc build lại với `CMAKE_ARGS` |
| Đi tìm wheel `cu130` | Không tồn tại và không cần — driver mới chạy được runtime cũ hơn. Dùng `cu124` |
| `llama-quantize not built` | Chưa build binary llama.cpp, hoặc `LLAMA_CPP` trỏ sai |
| `missing command: huggingface-cli` | `pip install "huggingface_hub[cli]"` |
| pip đè lên gói conda | Bình thường ở đây; chỉ cài `pip` **sau khi** đã activate env |
| `IncompleteRead` / `Connection broken` khi pip tải wheel | Wheel 1.8 GB và pip không resume. Dùng `bash scripts/install_llamacpp.sh` |
| Build từ nguồn lỗi, `nvcc` không tìm thấy host compiler | `conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y` |

> **Đừng trộn `conda install` và `pip install` cho cùng một gói.** Ở dự án này:
> conda chỉ lo Python và toolchain CUDA, pip lo toàn bộ gói Python.
