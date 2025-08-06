# 🚀 smaLLMs - AI Studio-Level Benchmarking Platform with Marathon Mode

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Local & Cloud](https://img.shields.io/badge/🔄-Ollama%20%2B%20LM%20Studio%20%2B%20Cloud-blue)](https://ollama.ai/)
[![Marathon Mode](https://img.shields.io/badge/🏃‍♂️-Marathon%20Mode-purple)](https://github.com/mmdmcy/smaLLMs)
[![OpenAI-Level](https://img.shields.io/badge/🏆-16%20AI%20Studio%20Benchmarks-red)](https://openai.com/)

> **�‍♂️ Marathon Mode: Run overnight evaluation of ALL your local models with ALL 16 benchmarks**  
> Supporting Ollama, LM Studio, and Cloud APIs with comprehensive AI studio-level evaluation!

![smaLLMs Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Benchmarks](https://img.shields.io/badge/Benchmarks-16-orange) ![Models](https://img.shields.io/badge/Local%20Models-23%2B-green)

## 🎯 What is smaLLMs?

**smaLLMs** is the most comprehensive local and cloud LLM evaluation platform, supporting **ALL 16 benchmarks used by OpenAI, Anthropic, Google DeepMind, and xAI**. Evaluate your Ollama and LM Studio models with the same rigor as top AI labs.

### 🌟 Key Features

- **🏃‍♂️ Marathon Mode**: Overnight evaluation of ALL local models with ALL benchmarks
- **🏆 16 AI Studio Benchmarks**: Complete suite including AIME, GPQA, Codeforces, HealthBench
- **🔄 Local + Cloud**: Seamlessly works with Ollama (15 models), LM Studio (8 models), and cloud APIs
- **🚀 One Command**: `python smaLLMs.py` - everything integrated into a single file
- **💰 Cost-Optimized**: Smart sampling and rate limiting for efficient evaluation
- **📊 Beautiful Interface**: Real-time progress with color-coded results
- **🎯 Production-Ready**: Battle-tested evaluation methodology
- **📁 Organized Results**: Date-based structure with clean exports
- **⚡ Windows Compatible**: Full Unicode support and robust error handling

## 🔥 NEW: Marathon Mode + 16 AI Studio Benchmarks

### 🏃‍♂️ **Marathon Mode**
Run overnight comprehensive evaluation of ALL your models:
- **Auto-Discovery**: Finds all 23+ local models (Ollama + LM Studio)
- **Smart Selection**: Choose specific models or run ALL discovered models
- **Benchmark Suites**: 18 different benchmark combinations to choose from
- **Progress Tracking**: Real-time updates and resume capability
- **Organized Results**: Clean date/time-based result organization
- **Windows Compatible**: Full Unicode support and robust timeout handling

### 🏆 **16 AI Studio Benchmarks**
Complete benchmark suite matching major AI companies:

#### **Competition & Expert Level**
- **AIME 2024/2025**: American Invitational Mathematics Examination (o3/o4 level)
- **GPQA Diamond**: PhD-level science questions (Google-Proof Q&A) 
- **Codeforces**: Competitive programming with ELO ratings
- **HLE**: Humanity's Last Exam - Expert cross-domain evaluation
- **HealthBench**: Medical conversation safety (includes Hard variant)
- **TauBench**: Function calling and tool use evaluation

#### **Core Academic Standards**
- **GSM8K**: Grade school mathematics reasoning
- **MMLU**: Massive multitask language understanding
- **MATH**: Mathematical reasoning and competition problems
- **HumanEval**: Code generation and programming capabilities
- **ARC**: Abstract reasoning challenge
- **HellaSwag**: Commonsense reasoning

#### **Advanced Reasoning**
- **WinoGrande**: Winograd schema challenge
- **BoolQ**: Boolean question answering
- **OpenBookQA**: Multi-step reasoning with facts
- **PIQA**: Physical interaction question answering

### 🎯 **18 Benchmark Suites Available**
- **Individual Benchmarks**: Any single benchmark (16 options)
- **OpenAI Suite**: Complete o3/o4 benchmark set  
- **Competition Suite**: AIME + Codeforces + MATH
- **Expert Suite**: GPQA + HLE + HealthBench
- **Academic Suite**: MMLU + GSM8K + HumanEval
- **Reasoning Suite**: ARC + HellaSwag + WinoGrande
- **Comprehensive Suite**: Best 8-benchmark coverage

## 🔥 Confirmed Working Local Models (23+)

### � **Ollama Models (15 discovered)**
- **llama3.2** - Meta's latest compact models
- **qwen2.5** - Alibaba's optimized instruction models  
- **qwen2.5-coder** - Specialized coding variants
- **granite3.2** - IBM's enterprise-ready models
- **deepseek-r1** - Reasoning-focused models
- **gemma-3** - Google's efficient instruction models
- **liquid** - High-performance compact models
- **And 8+ more automatically discovered**

### 🎯 **LM Studio Models (8 discovered)**  
- **Meta Llama variants** - Multiple sizes and versions
- **Qwen2.5 series** - Instruction and coder variants
- **Google Gemma models** - Various parameter sizes
- **Granite models** - IBM's latest offerings
- **DeepSeek variants** - Reasoning and general models

### ☁️ **Cloud Models (HuggingFace)**
All models from the original cloud configuration still supported for comparison.

## 🛠️ Technology Stack

### **Core Technologies**
- **Python 3.8+**: Modern async/await patterns for concurrent evaluation
- **HuggingFace Hub**: Direct API integration for model inference
- **Datasets Library**: Standardized benchmark data loading
- **AsyncIO**: Non-blocking concurrent model evaluation
- **YAML**: Human-readable configuration management

### **Data & Analytics**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing for metrics calculation
- **SciPy**: Statistical analysis and significance testing
- **Matplotlib/Seaborn**: Data visualization for reports

### **Web & Interface**
- **Gradio**: Optional web interface for interactive evaluation
- **FastAPI**: REST API for programmatic access
- **Beautiful Terminal**: Custom ANSI-colored terminal interface
- **HTML Export**: Static website generation from results

### **Evaluation Framework**
- **Custom Benchmarks**: Modular benchmark system
- **Async Model Manager**: Efficient model loading and inference
- **Result Aggregation**: Statistical analysis and ranking
- **Cost Estimation**: Real-time API cost tracking

## 🚀 Quick Start

### 1. **Installation**
```bash
git clone https://github.com/mmdmcy/smaLLMs.git
cd smaLLMs
pip install -r requirements.txt
```

### 2. **Setup Local Models (Optional)**
```bash
# Install Ollama (if you want local models)
# Windows: Download from https://ollama.ai
# Then pull some models:
ollama pull llama3.2
ollama pull qwen2.5:0.5b
ollama pull granite3.2:2b

# Or use LM Studio: Download from https://lmstudio.ai
```

### 3. **Configuration (Cloud models only)**
```bash
# Only needed if using cloud models
cp config/config.example.yaml config/config.yaml
# Add your HuggingFace token to config/config.yaml
```

### 4. **Run Marathon Mode**
```bash
python smaLLMs.py
```

**Marathon Mode Options:**
- **🔄 Local**: Auto-discover and evaluate all Ollama + LM Studio models
- **☁️ Cloud**: Evaluate HuggingFace models (requires config)
- **📊 Choose Models**: Select specific models from 23+ discovered
- **� Choose Benchmarks**: Pick from 18 benchmark suite options
- **🏃‍♂️ Run ALL**: Overnight evaluation of everything!

### 5. **Export & Analysis**
```bash
python simple_exporter.py
```
Generate beautiful websites, leaderboards, and analysis reports from your Marathon Mode results.

## 📊 Marathon Mode Performance

| Setup | Models | Benchmarks | Samples | Duration | Use Case |
|-------|--------|------------|---------|----------|----------|
| **Quick Test** | 3 local | 2 core | 25 | ~15 min | Testing setup |
| **Standard** | 8 local | 4 suites | 50 | ~2 hours | Daily evaluation |
| **Comprehensive** | 15 local | 8 benchmarks | 100 | ~6 hours | Weekly analysis |
| **Marathon ALL** | 23 models | 16 benchmarks | 200 | ~12 hours | Complete evaluation |

*Local model evaluation is FREE - no API costs!*

## 🎯 Confirmed Working Models

smaLLMs focuses on **reliability** with automatic model discovery:

### **Local Models (FREE)** ✅
**Auto-discovered from Ollama & LM Studio:**
- 📈 **15 Ollama models** - Automatically detected and configured
- 🎯 **8 LM Studio models** - Seamlessly integrated
- 🔄 **Progressive timeouts** - Handles slower local inference
- 💾 **Efficient caching** - Faster repeat evaluations

### **Cloud Models (API Required)** ✅  
**Battle-tested HuggingFace models:**
- `google/gemma-2-2b-it` - Google's efficient instruction model
- `Qwen/Qwen2.5-1.5B-Instruct` - Alibaba's optimized model  
- `meta-llama/Llama-3.2-1B-Instruct` - Meta's compact model
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` - HF's optimized model
- *Plus 6 more proven models*

*Marathon Mode automatically discovers your available models - no manual configuration needed!*

## 📁 File Structure (Streamlined)

```
smaLLMs/
├── 📄 smaLLMs.py              # 🏃‍♂️ Main Marathon Mode launcher (ALL-IN-ONE)
├── 🧠 intelligent_evaluator.py # Smart evaluation engine
├── 📊 simple_exporter.py      # Results export & website generation
├── 🎨 beautiful_terminal.py   # Color terminal interface
├── 🧪 test_everything.py      # Comprehensive test suite (15 tests)
├── 🔍 check_local_services.py # Local model discovery utility
├── ⚙️ config/
│   ├── config.yaml            # Your configuration (cloud only)
│   ├── config.example.yaml    # Example configuration
│   └── models.yaml            # Model definitions
├── 🔧 src/                    # Core evaluation modules
│   ├── models/                # Model management & discovery
│   ├── benchmarks/            # 16 benchmark implementations
│   ├── evaluator.py           # Evaluation orchestration
│   ├── metrics/               # Result analysis & aggregation
│   ├── utils/                 # Storage and utilities
│   └── web/                   # Optional web interface
└── 📁 smaLLMs_results/        # Marathon Mode results
    └── 2025-MM-DD/            # Date-based organization
        └── run_HHMMSS/        # Time-stamped runs
            ├── individual_results/ # Raw benchmark data
            ├── reports/           # Human-readable summaries
            └── exports/           # Website/analysis exports
```

**Everything you need in 17 essential files - no bloat!**

## 🛠️ How It Works

### **1. Marathon Mode Discovery**
```python
# Automatic model discovery across platforms
models = discover_local_models()  # Finds Ollama + LM Studio
benchmarks = load_benchmark_suite()  # All 16 AI studio benchmarks
```

### **2. Intelligent Orchestration**
- **Progressive Timeouts**: Adapts to local model inference speed
- **Smart Sampling**: Optimizes evaluation depth based on model performance  
- **Error Recovery**: Robust handling of model failures and timeouts
- **Progress Tracking**: Real-time updates with beautiful terminal interface

### **3. Local Model Integration**
- **Ollama API**: Direct integration with local Ollama models
- **LM Studio API**: Seamless connection to LM Studio inference server
- **Unified Interface**: Same benchmarks work across all model types
- **No API Costs**: Free evaluation of local models

### **4. Organized Data Management**
- **Auto Directory Creation**: Date/timestamp-based result organization
- **Multiple Formats**: JSON for machines, human-readable summaries
- **Export Ready**: One-click website and analysis generation
- **Resume Capability**: Continue interrupted Marathon Mode runs

## 💰 Cost Optimization & FREE Local Evaluation

### **FREE Local Models** 🎉
Marathon Mode with local models is **completely FREE**:
- **No API costs** for Ollama and LM Studio models
- **Unlimited evaluations** - run as many benchmarks as you want
- **23+ models available** - comprehensive local model comparison
- **Perfect for research** and experimentation

### **Cost-Efficient Cloud Models** 💰
When using cloud APIs, smaLLMs is optimized for efficiency:
- **Smart Sampling**: Don't waste tokens on failing models
- **Progressive Evaluation**: Start small, scale for promising models  
- **Rate Limiting**: Respect free tier limits
- **Early Stopping**: Skip models that consistently fail

**Typical cloud costs:**
- Quick test (3 models, 2 benchmarks): ~$0.05
- Standard evaluation (8 models, 4 benchmarks): ~$0.30
- Comprehensive (15 models, 8 benchmarks): ~$1.20

## 🌐 Export & Integration

### **Marathon Mode Results Export**
```bash
python simple_exporter.py
```
**Generates:**
- **📊 Beautiful Websites**: Interactive leaderboards and analysis
- **📈 Comparison Charts**: Visual model performance comparisons
- **📋 CSV/JSON Data**: Excel and analysis-ready formats
- **📝 Markdown Reports**: AI assistant and documentation ready
- **🏆 Leaderboards**: Rank your local models against benchmarks

### **Integration Ready**
- **REST API**: Optional web interface (via FastAPI)
- **JSON Data**: Machine-readable results for custom analysis
- **Modular Architecture**: Easy to extend with custom benchmarks
- **Plugin System**: Add new model providers and benchmarks

## 🔧 Configuration

### **Local Models (No config needed!)**
Marathon Mode automatically discovers your models:
```bash
# Just run it - no configuration required!
python smaLLMs.py
```

### **Cloud Models (Optional)**
```yaml
# config/config.yaml (only if using cloud models)
huggingface:
  token: "your_hf_token_here"

evaluation:
  default_samples: 100
  max_concurrent_requests: 3
  
marathon_mode:
  local_models_enabled: true
  auto_discovery: true
  result_organization: "date_time"
```

### **Advanced Customization**
- **Custom Benchmark Configurations**: Modify sample sizes and parameters
- **Model-Specific Settings**: Timeout and generation configs per model
- **Result Organization**: Customize directory structure and naming
- **Export Formats**: Configure output formats and destinations

## 🚀 Get Started with Marathon Mode

```bash
git clone https://github.com/mmdmcy/smaLLMs.git
cd smaLLMs
pip install -r requirements.txt
python smaLLMs.py
```

**Choose your adventure:**
1. **🔄 Local Models**: Auto-discover and evaluate all Ollama + LM Studio models (FREE!)
2. **☁️ Cloud Models**: Add HuggingFace token and evaluate cloud models  
3. **🏃‍♂️ Marathon Mode**: Run ALL models with ALL 16 benchmarks overnight
4. **📊 Custom**: Pick specific models and benchmark suites

**Join the local model revolution!** 🔥

---

## 🤝 Contributing

Help make smaLLMs even better:

- **🧪 New Benchmarks**: Add domain-specific evaluation tasks
- **🤖 Model Providers**: Integrate new local and cloud model platforms  
- **📊 Visualization**: Enhance Marathon Mode result analysis
- **� Performance**: Optimize local model inference and evaluation speed

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Created By

**mmdmcy** - [GitHub](https://github.com/mmdmcy)

*Building comprehensive local model evaluation with Marathon Mode - because your local models deserve AI studio-level benchmarking.*

---

*smaLLMs Marathon Mode - Run overnight evaluation of ALL your models with ALL benchmarks. Local is the new cloud.* 🏃‍♂️✨
