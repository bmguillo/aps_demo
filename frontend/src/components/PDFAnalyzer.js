import React, { useState, useEffect } from "react";
import "./PDFAnalyzer.css";

// function PDFAnalyzer() {

function PDFAnalyzer({ onPdfSelect }) {
  const [selectedPDF, setSelectedPDF] = useState("");
  const [pdfs, setPdfs] = useState([]);
  const [sections, setSections] = useState({
    summary: "",
    hospitalization: "",
    riskClassifier: "",
    recommendations: "",
    keyInsights: "",
  });
  const [chatContent, setChatContent] = useState("");
  const [chatResponse, setChatResponse] = useState("");

  useEffect(() => {
    fetchPDFs();
  }, []);

  // const fetchPDFs = async () => {
  //   try {
  //     const response = await fetch("http://localhost:8000/pdfs");
  //     const data = await response.json();
  //     console.log("Fetched PDFs:", data); // Debug log
  //     setPdfs(data.pdf_files || []);
  //   } catch (error) {
  //     console.error("Error fetching PDFs:", error);
  //   }
  // };

  const fetchPDFs = async () => {
    try {
      const response = await fetch("http://localhost:8000/pdfs");
      const data = await response.json();
      console.log("Fetched PDFs:", data); // Debug log
      if (Array.isArray(data.pdf_files)) {
        setPdfs(data.pdf_files);
      } else {
        console.error("Unexpected data structure:", data);
        setPdfs([]);
      }
    } catch (error) {
      console.error("Error fetching PDFs:", error);
      setPdfs([]);
    }
  };

  const handleRunAnalysis = async () => {
    if (!selectedPDF) return;

    const sectionNames = [
      "summary",
      "hospitalization",
      "riskClassifier",
      "recommendations",
      "keyInsights",
    ];
    const updatedSections = { ...sections };

    for (const section of sectionNames) {
      try {
        const response = await fetch("http://localhost:8000/analyze-section", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            pdf: selectedPDF,
            section: section,
          }),
        });
        const data = await response.json();
        updatedSections[section] = data.content || "No data available";
      } catch (error) {
        console.error(`Error fetching analysis for ${section}:`, error);
        updatedSections[section] = "Error fetching data";
      }
    }

    setSections(updatedSections);
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatContent.trim()) return;

    try {
      const response = await fetch("http://localhost:8000/chat-with-document", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_input: chatContent,
          document_context: selectedPDF,
        }),
      });
      const data = await response.json();
      setChatResponse(data.response || "No response received");
      setChatContent("");
    } catch (error) {
      console.error("Error sending chat:", error);
      setChatResponse("Error sending chat message");
    }
  };

  return (
    <div className="analyzer-container">
      <div className="header">
        <h1>IBM WatsonX APS Application</h1>
      </div>

      <div className="controls">
        <select
          className="applicant-select"
          value={selectedPDF}
          // onChange={(e) => setSelectedPDF(e.target.value)
          onChange={(e) => {
            setSelectedPDF(e.target.value);
            onPdfSelect && onPdfSelect(e.target.value);
          }}
        >
          <option value="">Select an Applicant</option>
          {pdfs.map((pdf) => (
            <option key={pdf} value={pdf}>
              {pdf}
            </option>
          ))}
        </select>

        <button className="run-button" onClick={handleRunAnalysis}>
          Run Analysis
        </button>
      </div>

      <div className="sections-grid">
        <div className="section-box summary-box">
          <h2>Summary</h2>
          <div className="section-content">{sections.summary}</div>
        </div>
        <div className="section-box">
          <h2>Hospitalization Classifier</h2>
          <div className="section-content">{sections.hospitalization}</div>
        </div>
        <div className="section-box">
          <h2>Risk Classifier</h2>
          <div className="section-content">{sections.riskClassifier}</div>
        </div>
        <div className="section-box">
          <h2>Recommendations</h2>
          <div className="section-content">{sections.recommendations}</div>
        </div>
        <div className="section-box">
          <h2>Key Insights</h2>
          <div className="section-content">{sections.keyInsights}</div>
        </div>
      </div>

      <div className="chat-section">
        <h2>RAG Chat with Documents</h2>
        <form onSubmit={handleChatSubmit} className="chat-form">
          <textarea
            className="chat-input"
            value={chatContent}
            onChange={(e) => setChatContent(e.target.value)}
            placeholder="Type your message here..."
          />
          <button type="submit" className="send-button">
            Send
          </button>
        </form>
        {chatResponse && (
          <div className="chat-response">
            <h3>Response:</h3>
            <p>{chatResponse}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default PDFAnalyzer;
