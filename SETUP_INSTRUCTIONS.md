# FloatChat Setup Instructions

## Quick Setup for Smart India Hackathon 2025

### 1. Clone Repository
```bash
git clone https://github.com/NITHISHKUMAR0283/marinex2.git
cd marinex2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` with these API keys:
```
# For full functionality, add your API keys:
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Provided Groq API key for demo:
GROQ_API_KEY=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
```

### 4. Initialize Database
```bash
python enhanced_indian_ocean_setup.py
```

### 5. Run FloatChat
```bash
python main.py
```

Visit: http://localhost:8501

## Demo API Key Information

For Smart India Hackathon 2025 demonstration, a Groq API key is provided:
- **API Key**: `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`
- **Provider**: Groq (Llama3-70B model)
- **Usage**: Fast inference for oceanographic queries
- **Quota**: Sufficient for hackathon demonstration

## System Requirements
- Python 3.9+
- 8GB RAM (recommended)
- 2GB disk space for database
- Internet connection for LLM APIs

## Features Available
✅ Multi-modal embeddings  
✅ FAISS vector search  
✅ RAG pipeline  
✅ Natural language to SQL  
✅ Interactive interface  
✅ 960K+ oceanographic measurements  
✅ Real-time visualization  

Built for Smart India Hackathon 2025