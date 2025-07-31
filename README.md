# 🚀 smaLLMs - Small Language Model Benchmarking Platform

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Powered by HuggingFace](https://img.shields.io/badge/🤗-Powered%20by%20HuggingFace-yellow)](https://huggingface.co/)

> **Enterprise-grade benchmarking platform for small language models (1B-7B parameters)**  
> One command to rule them all - streamlined evaluation with beautiful terminal interface and organized results.

![smaLLMs Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

## 🎯 What is smaLLMs?

**smaLLMs** is a production-ready, cost-optimized benchmarking platform designed specifically for evaluating small language models (1B-7B parameters). Built for researchers, developers, and organizations who need reliable, efficient, and comprehensive LLM evaluation without the enterprise-level costs.

### 🌟 Why smaLLMs?

- **🚀 One Command Setup**: `python smaLLMs.py` - that's it!
- **💰 Cost-Optimized**: Intelligent sampling and rate limiting to minimize API costs
- **📊 Beautiful Interface**: Terminal UI that matches modern development workflows
- **🎯 Production-Ready**: Reliable evaluation of confirmed working models only
- **📁 Organized Results**: Clean file structure with date-based organization
- **🌐 Web Export**: Generate beautiful websites from your evaluation results
- **⚡ Lightning Fast**: Quick demos in ~2 minutes, comprehensive evals in ~60 minutes

## 🔥 Key Features

### 🧠 **Intelligent Evaluation Engine**
- **Smart Sampling**: Progressive sample sizing based on model performance
- **Rate Limiting**: Respects API limits with intelligent request throttling
- **Early Stopping**: Automatically stops evaluating failing models
- **Cost Tracking**: Real-time cost monitoring and optimization

### 📊 **Comprehensive Benchmarking**
- **GSM8K**: Grade school math problems
- **MMLU**: Massive multitask language understanding
- **MATH**: Mathematical reasoning
- **HumanEval**: Code generation capabilities

### 🎮 **Beautiful Terminal Interface**
- **Real-time Progress**: Live model evaluation with progress bars
- **Color-coded Results**: Instant visual feedback on model performance
- **Cost Display**: Live cost tracking during evaluation
- **Clean Output**: Professional terminal interface

### 📁 **Organized File Management**
- **Date-based Structure**: `smaLLMs_results/YYYY-MM-DD/run_TIMESTAMP/`
- **Separated Results**: Individual results, reports, and exports in organized folders
- **JSON + Human-readable**: Machine-readable data with human-friendly summaries
- **Export Ready**: One-click website generation

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

### 2. **Configuration**
```bash
# Copy and edit config file
cp config/config.example.yaml config/config.yaml
# Add your HuggingFace token to config/config.yaml
```

### 3. **Run Evaluation**
```bash
python smaLLMs.py
```

Choose from 4 preset configurations:
- **⚡ Lightning** (3 models, 10 samples, ~2 min) - Perfect for testing
- **🔥 Quick** (5 models, 25 samples, ~8 min) - Rapid benchmarking
- **📊 Standard** (8 models, 50 samples, ~25 min) - Comprehensive evaluation
- **🏆 Comprehensive** (12 models, 100 samples, ~60 min) - Full enterprise evaluation

### 4. **Export Results**
```bash
python simple_exporter.py
```

Generate beautiful websites, CSV files, and JSON exports from your evaluation results.

## 📊 Evaluation Presets

| Preset | Models | Benchmarks | Samples | Duration | Cost | Use Case |
|--------|--------|------------|---------|----------|------|----------|
| **Lightning** | 3 | 2 | 10 | ~2 min | $0.03 | Quick testing |
| **Quick** | 5 | 2 | 25 | ~8 min | $0.08 | Rapid comparison |
| **Standard** | 8 | 3 | 50 | ~25 min | $0.25 | Production evaluation |
| **Comprehensive** | 12 | 4 | 100 | ~60 min | $0.60 | Enterprise benchmarking |

## 🎯 Confirmed Working Models

smaLLMs focuses on **reliability** - we only include models that are confirmed to work:

### **Tier 1: Proven Models** ✅
- `google/gemma-2-2b-it` - Google's efficient 2B instruction model
- `Qwen/Qwen2.5-1.5B-Instruct` - Alibaba's optimized 1.5B model
- `meta-llama/Llama-3.2-1B-Instruct` - Meta's latest compact model

### **Tier 2: Reliable Models** ✅
- `Qwen/Qwen2.5-3B-Instruct` - Larger Qwen variant
- `meta-llama/Llama-3.2-3B-Instruct` - Meta's 3B instruction model
- `google/gemma-2-9b-it` - Google's larger Gemma model

### **Tier 3: Extended Models** ✅
- `Qwen/Qwen2.5-7B-Instruct` - Full-size Qwen model
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` - HuggingFace's optimized model
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Compact conversation model
- `mistralai/Mistral-7B-Instruct-v0.3` - Mistral's instruction-tuned model

*No more broken Microsoft models or unreliable inference endpoints!*

## 📁 File Structure

```
smaLLMs/
├── 📄 smaLLMs.py              # Main unified launcher
├── 🧠 intelligent_evaluator.py # Smart evaluation engine
├── 📊 simple_exporter.py      # Website/data export system
├── 🎨 beautiful_terminal.py   # Terminal interface
├── ⚙️ config/
│   ├── config.yaml            # Your configuration
│   └── config.example.yaml    # Example configuration
├── 🔧 src/                    # Core evaluation modules
│   ├── models/                # Model management
│   ├── benchmarks/            # Benchmark implementations
│   ├── metrics/               # Result analysis
│   ├── utils/                 # Storage and utilities
│   └── web/                   # Web interface (optional)
└── 📁 smaLLMs_results/        # Organized results
    └── 2025-07-31/            # Date-based organization
        └── run_20250731_164729/   # Timestamped runs
            ├── individual_results/ # Raw evaluation data
            ├── reports/           # Human-readable summaries
            └── exports/           # Website/data exports
```

## 🏗️ How It Works

### **1. Intelligent Orchestration**
```python
# Smart evaluation with cost optimization
orchestrator = IntelligentEvaluationOrchestrator(config)
results = await orchestrator.run_intelligent_evaluation()
```

### **2. Progressive Sampling**
- Starts with small sample sizes for unknown models
- Increases samples for high-performing, reliable models
- Reduces samples for struggling models to save costs

### **3. Rate Limiting & Cost Control**
- Respects HuggingFace API rate limits (20-30 requests/minute)
- Real-time cost tracking and estimation
- Intelligent delays between requests

### **4. Organized Data Management**
- Automatic directory creation with date/timestamp organization
- Separate storage for individual results, reports, and exports
- JSON for machines, human-readable summaries for people

### **5. Beautiful Terminal Experience**
- Real-time progress bars and status updates
- Color-coded performance indicators
- Live cost and duration tracking
- Clean, professional output

## 💰 Cost Optimization

smaLLMs is designed to be **cost-conscious**:

- **Smart Sampling**: Don't waste tokens on failing models
- **Rate Limiting**: Respect free tier limits
- **Progressive Evaluation**: Start small, scale up for promising models
- **Early Stopping**: Stop evaluating models that consistently fail
- **Cost Tracking**: Know exactly how much you're spending

**Typical costs:**
- Lightning Demo: $0.03 (perfect for testing)
- Standard Evaluation: $0.25 (production ready)
- Comprehensive Benchmark: $0.60 (enterprise grade)

## 🌐 Export & Integration

### **Website Export**
```bash
python simple_exporter.py
```
Generates:
- **HTML**: Beautiful standalone websites
- **JSON**: Machine-readable data for integration
- **CSV**: Excel/Google Sheets compatible
- **Markdown**: AI assistant ready summaries

### **Integration Ready**
- REST API endpoints (via FastAPI)
- JSON data format for easy parsing
- Modular architecture for custom benchmarks
- Plugin system for additional models

## 🔧 Configuration

### **Basic Setup**
```yaml
# config/config.yaml
huggingface:
  token: "your_hf_token_here"
  use_pro_features: true

evaluation:
  default_samples: 50
  max_concurrent_requests: 5
  
storage:
  results_format: "json"
  local_cache_mb: 500
```

### **Advanced Options**
- Custom benchmark configurations
- Model-specific settings
- Cost limits and budgets
- Custom export formats

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- 🧪 **New Benchmarks**: Add domain-specific evaluation tasks
- 🤖 **Model Support**: Integrate new model providers
- 📊 **Visualization**: Enhance result visualization
- 🔧 **Optimization**: Improve performance and cost efficiency

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Created By

**mmdmcy** - [GitHub](https://github.com/mmdmcy)

*Building the future of efficient LLM evaluation, one small model at a time.*

---

## 🚀 Get Started Now

```bash
git clone https://github.com/mmdmcy/smaLLMs.git
cd smaLLMs
pip install -r requirements.txt
python smaLLMs.py
```

**Join the small model revolution!** 🔥

---

*smaLLMs - Because sometimes smaller is smarter, faster, and more cost-effective.*
