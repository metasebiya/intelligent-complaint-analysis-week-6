
# 🧠 Intelligent Complaint Analysis with RAG

This project builds a **Retrieval-Augmented Generation (RAG)** system to help internal teams at **CrediTrust Financial** quickly understand customer complaints across five core financial products:

- Credit Cards
- Personal Loans
- Buy Now, Pay Later (BNPL)
- Savings Accounts
- Money Transfers

The tool enables internal users (like product managers or compliance officers) to ask **natural language questions** and receive **contextual, evidence-backed answers** sourced directly from complaint narratives.

---

## 🚀 Project Architecture

### 1. 🧹 **Data Preprocessing**
- Clean and normalize complaint data from the Consumer Financial Protection Bureau (CFPB)
- Filter by relevant product categories and drop noisy entries
- Extract and clean complaint narratives

### 2. ✂️ **Text Chunking and Embedding**
- Split long complaint texts using `RecursiveCharacterTextSplitter`
- Embed chunks using `sentence-transformers/all-MiniLM-L6-v2`
- Store embeddings in a local **ChromaDB vector store** with metadata (e.g., product, complaint ID)

### 3. 🤖 **RAG Pipeline (Retriever + Generator)**
- **Retriever**: Top-k similarity search via ChromaDB
- **Prompt Template**: Guides the LLM to answer using only retrieved complaint excerpts
- **Generator**: Uses an open-source LLM (e.g., Mistral or LLaMA2) to synthesize an answer

### 4. 💬 **Gradio UI**
- Ask questions via a user-friendly chat interface
- View answer + source chunks to validate and build trust
- One-click "Clear" functionality

---

## 📁 Project Structure

```
intelligent-complaint-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── chroma_db/
├── src/
│   ├── data_loader.py
│   ├── data_processor.py
│   ├── embed_index.py
│   ├── rag_pipeline.py
│   └── app.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Repository
```bash
git clone https://github.com/your-username/intelligent-complaint-analysis.git
cd intelligent-complaint-analysis
```

### ✅ 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### ✅ 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### ✅ 4. Run Preprocessing
```bash
python src/data_processor.py
```

### ✅ 5. Build Vector Index
```bash
python src/embed_index.py
```

### ✅ 6. Launch Streamlit UI
```bash
streamlit run src/app.py
```

---

## 🧪 Sample Questions to Try

- *Why are users unhappy with BNPL?*
- *What are the common issues reported with credit cards?*
- *How often are savings accounts mentioned in complaints?*
- *What type of companies are involved in personal loan disputes?*

---

## ✅ Key Features

- 💬 Natural language interface
- 🔍 Context-based retrieval with ChromaDB
- 🤖 LLM-generated answers with citation of sources
- 🚀 Open-source, lightweight, and fully local

---

## 📈 Performance & Evaluation

| Question | Answer Summary | Source Chunks (Preview) | Score (1-5) | Notes |
|----------|----------------|--------------------------|-------------|-------|
| "Why are users unhappy with BNPL?" | High late fees, unclear terms | [BNPL complaint 1…], [BNPL complaint 2…] | 5 | Matches human intuition |

---

## 📌 Model Notes

- **Embedding Model:** `all-MiniLM-L6-v2`
- **Vector Store:** `ChromaDB`
- **LLM:** `flan-t5-small`

---

## 🧠 Future Enhancements

- Add filters by product, date range, or company in the UI
- Add support for multilingual queries and translations
- Quantitative metrics (BLEU, ROUGE) for evaluating response quality

---

## 📜 License

MIT License

---

## 👤 Author

**Metasebiya Bizuneh**  
Data & AI Engineer
[LinkedIn](https://www.linkedin.com/in/metasebiya-bizuneh) • [GitHub](https://github.com/your-username)
