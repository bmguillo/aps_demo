from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ibm_watsonx_ai.foundation_models import Model
from dotenv import load_dotenv
import os
from pypdf import PdfReader
import logging
from fastapi.responses import FileResponse
import requests
import ssl
import certifi
import warnings

warnings.filterwarnings('ignore', category=Warning)
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/pdfs", StaticFiles(directory="/Users/brandyguillory/aps_demo/backend/pdfs"), name="pdfs")

class AnalysisRequest(BaseModel):
    pdf: str
    section: str
    custom_prompt: str = ""
    model: str = "mistralai/mixtral-8x7b-instruct-v01"
    decoding_method: str = "greedy"
    min_tokens: int = 5
    max_tokens: int = 300

SUMMARY_PARAMS = {
    "model": "mistralai/mixtral-8x7b-instruct-v01",
    "decoding_method": "sample",
    "max_new_tokens": 500,
    "min_new_tokens": 100,
    # "temperature": 0.7,
    # "top_p": 0.8,
    # "top_k": 50,
    # "repetition_penalty": 1.1,
    
}

KEY_INSIGHTS_PARAMS = {
    "model": "mistralai/mixtral-8x7b-instruct-v01",
    "decoding_method": "greedy",
    "max_new_tokens": 300,
    "min_new_tokens": 50,
    # "temperature": 0.5,
    # "top_p": 0.7,
    # "top_k": 40,
    # "repetition_penalty": 1.2,
 
}

RECOMMENDATIONS_PARAMS = {
    "model": "mistralai/mixtral-8x7b-instruct-v01",
    "decoding_method": "sample",
    "max_new_tokens": 300,
    "min_new_tokens": 5,
    # "temperature": 0.6,
    # "top_p": 0.85,
    # "top_k": 45,
    # "repetition_penalty": 1.15,
 
}

HOSPITALIZATION_PARAMS = {
    "model": "mistralai/mixtral-8x7b-instruct-v01",
    "decoding_method": "greedy",
    "max_new_tokens": 20,
    "min_new_tokens": 1,
    # "temperature": 0.3,
    # "top_p": 0.6,
    # "top_k": 30,
    # "repetition_penalty": 1.3,

}

RISK_CLASSIFIER_PARAMS = {
    #"model": "mistralai/mixtral-8x7b-instruct-v01",
    "decoding_method": "greedy",
    "max_new_tokens": 300,
    "min_new_tokens": 5,
    # "temperature": 0.4,
    # "top_p": 0.7,
    # "top_k": 35,
    # "repetition_penalty": 1.25,

}

DEFAULT_PROMPTS = {
    "summary": """System: You are an expert Life Underwriter Officer specializing in analyzing Attending Physicians Statements (APS). You are detail-oriented, thorough, and focused on accurate risk assessment.

Task: Analyze the provided APS document and create a comprehensive two-paragraph summary highlighting:
- Key medical aspects
- Patient's medical journey
- Physician's explanations and observations

Instructions:
1. Write no less than two paragraphs
2. Use proper paragraph formatting
3. Focus on facts without prefacing statements with qualifiers
4. Include all significant medical conditions, treatments, and outcomes
5. Maintain chronological order where relevant
6. Start directly with the summary content (no headers or prefixes)
7. Do not include "Summary:" or any similar prefix in your response

{document_text}
""",

    "hospitalization": """System: You are an expert Medical Records Analyst specializing in hospitalization assessment for insurance underwriting.

Task: Analyze the provided medical document and determine the patient's hospitalization status. Provide only one word as output.

Instructions:
1. Review the entire medical record for any mentions of hospital stays
2. Consider both inpatient and overnight observations
3. Exclude emergency room visits without admission
4. Provide ONLY ONE of these two outputs: "Hospitalized" or "Not Hospitalized"
5. Do not provide any explanations or additional information

Example 1:
"Patient was treated in the ER for chest pain and released same day."
"Not Hospitalized"

Example 2:
"Patient was admitted for 3 days following complications from pneumonia."
"Hospitalized"

Now analyze the following medical record and provide only "Hospitalized" or "Not Hospitalized" as your complete response:


{document_text}
""",

    "riskClassifier": """System: You are an expert Life Insurance Underwriter specializing in risk classification based on medical records.

Task: Analyze the provided medical document and assign ONE of three risk ratings: preferred, standard, or substandard.

Instructions:
1. Consider all medical conditions, treatments, and lifestyle factors
2. Evaluate against standard life insurance industry criteria
3. First line: Provide ONLY ONE of these three words: preferred, standard, or substandard
4. Second line: Start your explanation with "While" (no dash or prefix)
5. add a period at the end of the sentences


Evaluation Criteria:
- Preferred: Excellent health, no chronic conditions, favorable lifestyle
- Standard: Average health, well-managed conditions, typical lifestyle
- Substandard: Multiple health issues, poorly managed conditions, or high-risk factors

Example Output:
Preferred

Reasoning:
While the applicant shows excellent health markers with no chronic conditions, maintains regular exercise, and has no adverse family history.

Example Output:
Substandard

Reasoning:
While the patient has multiple cardiovascular risk factors, poorly controlled diabetes, and continues to smoke despite medical advice.

Now analyze the following medical record and provide your classification following the format above:

{document_text}
""",

    "keyInsights": """System: You are a Medical Claims Analyst specializing in identifying key patient actions and medical events from healthcare documents.

Task: Extract the top 5 most significant actions or events from the provided medical document.

Instructions:
1. Identify only direct patient actions and medical events
2. List exactly 5 key points
3. Number each point (1-5)
4. Place each point on its own line
5. End each point with a period
6. Start directly with numbered points (no header or prefix)
7. Add a line break between each point

Example Output:
1. Underwent heart surgery in March 2024.

2. Completed two weeks of post-surgical hospitalization.

3. Participated in thrice-weekly physical therapy sessions.

4. Maintained prescribed medication regimen including blood thinners.

5. Completed cardiac rehabilitation program in June 2024.

Now analyze the following document and provide the top 5 key insights following this format:

{document_text}
""",

    "recommendations": """System: You are a Senior Life Insurance Underwriter at New York Life with expertise in medical risk assessment and policy recommendations.

Task: Review the provided medical document and generate specific underwriting recommendations based on NYL guidelines.

Instructions:
1. Analyze all medical conditions and risk factors
2. Consider current health status and treatment compliance
3. Apply standard NYL underwriting criteria
4. Start directly with numbered points (no header or prefix)
5. Provide as a numbered list and end with a period

Key Considerations:
- Medical history and current conditions
- Treatment compliance and outcomes
- Lifestyle factors and risk mitigation
- Family history relevance
- Laboratory results and trends

Example Output:

1. Request blood pressure readings from the last 6 months to confirm consistent control.

2. Obtain medication compliance report from primary physician.

3. Review family history details for age of onset of heart disease.

4. Consider standard rates if BP remains stable and no other risk factors present.

5. Schedule follow-up review in 12 months to assess blood pressure management.

Now analyze the following medical record and provide your numbered recommendations:

{document_text}
""",
}

def get_watson_model(model_id, analysis_type="summary"):
    project_id = os.getenv('WATSONX_PROJECT_ID')
    apikey = os.getenv('WATSONX_API_KEY')

    if not all([project_id, apikey]):
        raise ValueError("Missing required environment variables")

    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": apikey
    }

    params_map = {
        "summary": SUMMARY_PARAMS,
        "keyInsights": KEY_INSIGHTS_PARAMS,
        "recommendations": RECOMMENDATIONS_PARAMS,
        "hospitalization": HOSPITALIZATION_PARAMS,
        "riskClassifier": RISK_CLASSIFIER_PARAMS
    }
    
    params = params_map.get(analysis_type, SUMMARY_PARAMS)

    return Model(
        model_id=model_id,
        params=params,
        credentials=credentials,
        project_id=project_id
    )

def extract_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

@app.get("/pdfs")
async def list_pdfs():
    try:
        pdf_directory = "pdfs"
        if not os.path.exists(pdf_directory):
            raise HTTPException(status_code=404, detail="PDF directory not found")
            
        pdfs = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        return {"pdf_files": pdfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing PDFs: {str(e)}")

@app.post("/analyze-section")
async def analyze_section(request: AnalysisRequest):
    model = get_watson_model(request.model, analysis_type=request.section)
    
    pdf_path = os.path.join("pdfs", request.pdf)
    document_text = extract_pdf_text(pdf_path)

    prompt = request.custom_prompt or DEFAULT_PROMPTS.get(request.section, "Analyze this document.")
    full_prompt = prompt.format(document_text=document_text)

    response = model.generate(prompt=full_prompt)
    
    generated_text = response['results'][0]['generated_text'] if isinstance(response, dict) else str(response)
    
    return {
        "section": request.section,
        "content": generated_text.strip(),
        "model": request.model
    }

@app.post("/chat-with-document")
async def chat_with_document(data: dict):
    model = get_watson_model("mistralai/mixtral-8x7b-instruct-v01", analysis_type="summary")
    
    pdf_path = os.path.join("pdfs", data.get('document_context', ''))
    document_context = extract_pdf_text(pdf_path) if os.path.exists(pdf_path) else data.get('document_context', '')
    user_query = data.get('user_input', '')

    prompt = f"""System: You are a expert New York Life underwriter, you will be given a APS as context and question will be asked. You need to use APS document as context and answer the question.

Context Document:
{document_context}

User Query:
{user_query}

Instructions:
1. Start your response by repeating the user's question
2. Provide clear, factual responses based on the document content
3. Stay within the scope of the provided document
4. Use medical terminology appropriately
5. Acknowledge any limitations in the document's information

Question: [User's question]
Answer: [Your detailed response]"""

    response = model.generate(prompt=prompt)
    generated_text = response['results'][0]['generated_text'] if isinstance(response, dict) else str(response)

    if not generated_text.strip().startswith("Question:"):
        formatted_response = f"Question: {user_query}\nAnswer: {generated_text.strip()}"
    else:
        formatted_response = generated_text.strip()

    return {
        "response": formatted_response
    }