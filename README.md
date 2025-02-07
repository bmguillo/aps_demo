# APS Demo - AI-Powered Medical Document Analysis

An intelligent application for analyzing Attending Physician Statements (APS) using IBM WatsonX AI. This tool helps life insurance underwriters quickly assess medical documents, classify risk, and generate comprehensive insights.

## Overview

APS Demo is a full-stack application that leverages IBM WatsonX AI (Mixtral-8x7b-instruct model) to automate the analysis of medical documents for life insurance underwriting. The application provides:

- **Automated Document Analysis**: Generate comprehensive summaries of medical records
- **Risk Classification**: Classify applicants as Preferred, Standard, or Substandard risk
- **Hospitalization Detection**: Identify hospitalization history from medical records
- **Key Insights Extraction**: Extract the top 5 most significant medical events
- **Underwriting Recommendations**: Generate actionable recommendations for underwriters
- **Interactive Chat**: RAG-based chat interface to query documents using natural language

## Architecture

### Backend (FastAPI + Python)
- **Framework**: FastAPI for high-performance REST API
- **AI Integration**: IBM WatsonX AI with Mixtral-8x7b-instruct model
- **PDF Processing**: PyPDF for document text extraction
- **Configuration**: Environment-based settings with python-dotenv

### Frontend (React)
- **Framework**: React 18.2.0
- **UI Components**: Custom PDF viewer and analyzer components
- **Styling**: CSS with responsive design
- **API Communication**: Axios for HTTP requests

## Prerequisites

- Python 3.8+
- Node.js 14+
- IBM WatsonX AI account with API credentials
- npm or yarn package manager

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd aps_demo
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your IBM WatsonX credentials
```

**Required Environment Variables** (`.env`):
```env
WATSONX_API_KEY=your_api_key_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com/
WATSONX_PROJECT_ID=your_project_id_here
CORS_ORIGINS=http://localhost:3000
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## Usage

### Starting the Application

1. **Start Backend Server**:
```bash
cd backend
uvicorn main:app --reload --port 8000
```

2. **Start Frontend Development Server**:
```bash
cd frontend
npm start
```

3. **Access the Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Using the Application

1. **Select an Applicant**: Choose a PDF document from the dropdown menu
2. **Run Analysis**: Click "Run Analysis" to generate all insights
3. **Review Results**: View the generated analysis in five sections:
   - Summary
   - Hospitalization Classifier
   - Risk Classifier
   - Recommendations
   - Key Insights
4. **Chat with Document**: Use the RAG chat interface to ask specific questions about the document

## Project Structure

```
aps_demo/
├── backend/
│   ├── main.py                 # FastAPI application and endpoints
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example           # Environment variables template
│   ├── config/
│   │   └── settings.py        # Configuration management
│   └── pdfs/                  # Sample APS documents
│       ├── APS_Emily_Wilson.pdf
│       └── APS_Rogers_Pamela.pdf
├── frontend/
│   ├── package.json           # Node.js dependencies
│   ├── public/                # Static assets
│   └── src/
│       ├── App.js            # Main application component
│       ├── App.css           # Application styles
│       └── components/
│           ├── PDFAnalyzer.js    # Analysis interface component
│           ├── PDFAnalyzer.css   # Analyzer styles
│           ├── PDFViewer.js      # PDF display component
│           └── PDFViewer.css     # Viewer styles
└── pdfs/                      # Additional PDF storage
```

## API Endpoints

### GET `/pdfs`
List all available PDF documents.

**Response**:
```json
{
  "pdf_files": ["APS_Emily_Wilson.pdf", "APS_Rogers_Pamela.pdf"]
}
```

### POST `/analyze-section`
Analyze a specific section of a PDF document.

**Request Body**:
```json
{
  "pdf": "APS_Emily_Wilson.pdf",
  "section": "summary",
  "model": "mistralai/mixtral-8x7b-instruct-v01",
  "custom_prompt": "",
  "decoding_method": "greedy",
  "min_tokens": 5,
  "max_tokens": 300
}
```

**Supported Sections**:
- `summary`: Comprehensive document summary
- `hospitalization`: Hospitalization status detection
- `riskClassifier`: Risk classification (Preferred/Standard/Substandard)
- `keyInsights`: Top 5 key medical events
- `recommendations`: Underwriting recommendations

### POST `/chat-with-document`
Interactive chat with document context using RAG.

**Request Body**:
```json
{
  "user_input": "What medications is the patient taking?",
  "document_context": "APS_Emily_Wilson.pdf"
}
```

## AI Model Configuration

The application uses different parameter configurations for each analysis type:

- **Summary**: Sample decoding, 100-500 tokens
- **Key Insights**: Greedy decoding, 50-300 tokens
- **Recommendations**: Sample decoding, 5-300 tokens
- **Hospitalization**: Greedy decoding, 1-20 tokens (binary classification)
- **Risk Classifier**: Greedy decoding, 1-300 tokens

## Security Notes

- Never commit `.env` files with real credentials
- Use `.env.example` as a template
- Keep API keys secure and rotate regularly
- Implement proper authentication for production use
- Sanitize user inputs before processing

## Technologies Used

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **IBM WatsonX AI**: Enterprise AI platform for foundation models
- **PyPDF/PyMuPDF**: PDF text extraction
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for FastAPI

### Frontend
- **React**: JavaScript library for building user interfaces
- **Axios**: Promise-based HTTP client
- **React-PDF**: PDF rendering in React
- **CSS3**: Modern styling and responsive design

## Development

### Adding New Analysis Types

1. Define parameters in `main.py`:
```python
NEW_ANALYSIS_PARAMS = {
    "model": "mistralai/mixtral-8x7b-instruct-v01",
    "decoding_method": "greedy",
    "max_new_tokens": 300,
    "min_new_tokens": 5,
}
```

2. Add prompt template to `DEFAULT_PROMPTS`:
```python
"newAnalysis": """Your prompt template here..."""
```

3. Update frontend component to display new section

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## Troubleshooting

### Common Issues

1. **PDF Not Loading**: Ensure the PDF path is correct and the file exists in the `pdfs/` directory
2. **API Connection Error**: Verify backend is running on port 8000
3. **WatsonX Authentication Error**: Check your API key and project ID in `.env`
4. **CORS Issues**: Ensure CORS_ORIGINS includes your frontend URL

## License

This project is proprietary software. All rights reserved.

## Contributors

- Development Team: Life Insurance Underwriting AI Solutions

## Support

For issues, questions, or contributions, please contact the development team.

## Version History

- **v0.1.0** (Current): Initial release with core analysis features
  - Document summarization
  - Risk classification
  - Hospitalization detection
  - Key insights extraction
  - Underwriting recommendations
  - RAG-based chat interface

## Future Enhancements

- [ ] Multi-document comparison
- [ ] Historical analysis tracking
- [ ] Export reports to PDF/Word
- [ ] Advanced filtering and search
- [ ] User authentication and authorization
- [ ] Batch processing capabilities
- [ ] Integration with underwriting systems
- [ ] Custom model fine-tuning options