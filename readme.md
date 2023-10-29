## Llama GPT with Langchain and streamLit

LlamaGPT using Langchain and StreamLit with support to upload PDF file.

## Installation

### Pre Requistes

1. Python
2. C++ dev tools
3. pipenv

**clone the Repo**

```bash
  git clone https://github.com/varunkumarnr/LlamaGPT.git
  cd Llamagpt
```

**Create a virtual environemnt and install dependencies**

```bash
  pipenv shell
  pipenv install
  pipenv sync
```

**clone llama-cpp-python Repo**

```bash
  git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git
  set FORCE_CMAKE=1
  set CMAKE_ARGS=-DLLAMA_CUBLAS=OFF
```

**Download Llama Model** and copy inside the project to models folder
[Link](https://huggingface.co/localmodels/Llama-2-7B-Chat-ggml/tree/main) Model Name: llama-2-7b-chat.ggmlv3.q2_K.bin

Model is in GGML format which is not supported by llama-cpp-python anymore so run below command to convert to GGUF (Replace with your path to models)

```bash
python ./convert-llama-ggml-to-gguf.py --eps 1e-5 --input C:/Users/User/Documents/Learning/ML/Langchain/models/llama-2-7b-chat.ggmlv3.q2_K.bin  --output  C:/Users/User/Documents/Learning/ML/Langchain/models/llama-2-7b-chat.ggufv3.q2_K.bin
```

**Run the app**

```bash
streamlit run app.py
```
