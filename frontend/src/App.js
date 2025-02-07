import React, { useState, useEffect } from "react";
import PDFAnalyzer from "./components/PDFAnalyzer";
import PDFViewer from "./components/PDFViewer";
import "./App.css";

function App() {
  const [selectedPdf, setSelectedPdf] = useState(null);

  useEffect(() => {
    console.log("App - selectedPdf:", selectedPdf);
  }, [selectedPdf]);

  return (
    <div className="app-container">
      <div className="left-panel">
        <PDFAnalyzer onPdfSelect={setSelectedPdf} />
      </div>
      <div className="right-panel">
        {selectedPdf ? (
          <PDFViewer selectedPdf={selectedPdf} />
        ) : (
          <div>No PDF selected</div>
        )}
      </div>
    </div>
  );
}

export default App;