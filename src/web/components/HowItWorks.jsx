const STEPS = [
  {
    number: "01",
    title: "You upload an MRI scan",
    body:
      "A magnetic resonance imaging (MRI) scan produces detailed images of the brain using magnetic fields. " +
      "The model accepts standard grayscale MRI images in JPEG or PNG format.",
  },
  {
    number: "02",
    title: "A neural network analyzes it",
    body:
      "The image is fed into EfficientNetB0 — a deep convolutional neural network originally trained on " +
      "1.2 million photos and then fine-tuned on 5,700 labeled brain MRI scans across 4 classes.",
  },
  {
    number: "03",
    title: "The model votes across 4 classes",
    body:
      "The network outputs a confidence score for each class: Glioma, Meningioma, Pituitary, and No Tumor. " +
      "The class with the highest score wins. The confidence bar chart shows all four scores.",
  },
  {
    number: "04",
    title: "Grad-CAM explains the decision",
    body:
      "Gradient-weighted Class Activation Mapping (Grad-CAM) traces which pixels most strongly influenced " +
      "the prediction. Red areas contributed the most — giving you a visual explanation rather than just a label.",
  },
];

export default function HowItWorks() {
  return (
    <section className="how-section">
      <h2 className="how-title">How does it work?</h2>
      <p className="how-subtitle">
        No machine learning background needed — here is the full pipeline in plain language.
      </p>
      <div className="steps-grid">
        {STEPS.map((step) => (
          <div key={step.number} className="step-card">
            <div className="step-number">{step.number}</div>
            <h3 className="step-title">{step.title}</h3>
            <p className="step-body">{step.body}</p>
          </div>
        ))}
      </div>

      <div className="model-stats">
        <div className="stat">
          <span className="stat-value">95%</span>
          <span className="stat-label">Test Accuracy</span>
        </div>
        <div className="stat-divider" />
        <div className="stat">
          <span className="stat-value">5,712</span>
          <span className="stat-label">Training Images</span>
        </div>
        <div className="stat-divider" />
        <div className="stat">
          <span className="stat-value">4</span>
          <span className="stat-label">Tumor Classes</span>
        </div>
        <div className="stat-divider" />
        <div className="stat">
          <span className="stat-value">20</span>
          <span className="stat-label">Training Epochs</span>
        </div>
      </div>
    </section>
  );
}
