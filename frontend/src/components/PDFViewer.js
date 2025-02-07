import React from "react";

function PDFViewer({ selectedPdf }) {
  console.log("PDFViewer - selectedPdf:", selectedPdf);

  if (!selectedPdf) {
    return <div>No PDF selected</div>;
  }

  return (
    <div className="right-panel">
      <object
        data={`http://localhost:8000/pdfs/${selectedPdf}`}
        type="application/pdf"
        width="100%"
        height="100%"
        style={{ border: "none", minHeight: "800px" }}
      >
        <p>Unable to display PDF. Selected PDF: {selectedPdf}</p>
      </object>
    </div>
  );
}

export default PDFViewer;
