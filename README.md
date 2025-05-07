# 🛠️ Valve Data Integrity Checker

An intelligent Python + Streamlit application that validates valve product specifications from e-commerce websites against manufacturer PDF spec sheets using NLP, OCR, and computer vision.

---

## 📌 Features

* 🧠 **AI-Powered Validation**
  Detects mismatches, range/format inconsistencies, unit differences, and semantic equivalence using Sentence-BERT.

* 📄 **PDF and Website Comparison**
  Extracts product data from both website HTML and downloadable spec sheets.

* 📟 **OCR + Fuzzy Matching Support**
  Automatically falls back to OCR for scanned PDFs, with fuzzy matching enabled for resilience.

* 🖼️ **Image Similarity Check**
  Compares product images on the website vs. PDF using ResNet + cosine similarity.

* 📊 **Export Reports**
  Results downloadable as CSV, PDF, and ZIP report bundles.

---

## 🚀 How to Run

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Launch the App**

```bash
python run.py
```

The app will auto-launch at: `http://localhost:8501`

---

## 📁 Project Structure

```bash
.
├── run.py               # Startup script for EXE / browser launch
├── app.py               # Main Streamlit application logic
├── requirements.txt     # All Python dependencies
├── README.md            # Project documentation
└── _config.yml          # GitHub Pages config (for documentation hosting)
```

---

## 🧠 Intelligent Reasoning Example

Instead of hardcoded messages, the app produces dynamic actions like:

> **"Moderate semantic similarity; May refer to a range expression; Possible unit or format variation. Closest phrase: 'Pressure limits between 80 and 116 psi under standard use...' (Similarity: 72.4%)"**

---

## 🛠️ Technologies Used

* **Streamlit** – UI
* **BeautifulSoup** – HTML scraping
* **PyMuPDF (fitz)** – PDF parsing
* **pdf2image + pytesseract** – OCR fallback
* **fuzzywuzzy** – Fuzzy text match
* **Torch + ResNet** – Image comparison
* **SentenceTransformer** – Semantic analysis (MiniLM-L6-v2)

---

## 📬 Contact

* GitHub: [github.com/aeronabrahan](https://github.com/aeronabrahan)
* Email: [aerongabrahan@gmail.com](mailto:aerongabrahan@gmail.com)
