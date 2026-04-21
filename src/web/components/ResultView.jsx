import ConfidenceBars from "./ConfidenceBars";
import ModelMetrics from "./ModelMetrics";

const CLASS_INFO = {
  glioma: {
    label: "Glioma",
    color: "#ef4444",
    badge: "bg-red",
    description:
      "A glioma is a tumor that grows from glial cells. " +
      "The heatmap highlights the area the model found most characteristic of this tumor type. " +
      "Gliomas vary widely in aggressiveness and require specialist evaluation.",
  },
  meningioma: {
    label: "Meningioma",
    color: "#f97316",
    badge: "bg-orange",
    description:
      "A meningioma grows from the meninges. These are the membranes that wrap around the brain and spinal cord. " +
      "Most meningiomas are benign (non-cancerous) and grow slowly. " +
      "The model focused on the tumor's boundary and density patterns to make this call.",
  },
  notumor: {
    label: "No Tumor Detected",
    color: "#22c55e",
    badge: "bg-green",
    description:
      "The model found no patterns associated with known tumor types in this scan. " +
      "The highlighted areas show the brain structures the model used as reference points " +
      "to confirm the absence of abnormal tissue.",
  },
  pituitary: {
    label: "Pituitary Tumor",
    color: "#6366f1",
    badge: "bg-indigo",
    description:
      "A pituitary tumor forms on the pituitary gland. It is a pea-sized gland at the brain's base " +
      "that regulates hormones throughout the body. Most are non-cancerous and highly treatable. " +
      "The model detected characteristic density changes in that region.",
  },
};

const TRUE_CLASS_LABELS = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  notumor: "No Tumor",
  pituitary: "Pituitary",
};

export default function ResultView({ result, originalSrc, trueClass, onReset }) {
  const { predicted_class, confidence, scores, heatmap_base64 } = result;
  const info = CLASS_INFO[predicted_class] || {
    label: predicted_class,
    color: "#94a3b8",
    description: "",
  };

  const heatmapSrc = `data:image/png;base64,${heatmap_base64}`;
  const correct = trueClass ? trueClass === predicted_class : null;

  return (
    <div className="result-wrapper">
      {/* Prediction badge */}
      <div className="prediction-header">
        <div className="prediction-badge" style={{ borderColor: info.color }}>
          <span className="prediction-label" style={{ color: info.color }}>
            {info.label}
          </span>
          <span className="prediction-conf" style={{ color: info.color }}>
            {(confidence * 100).toFixed(1)}% confidence
          </span>
        </div>
      </div>

      {/* Ground-truth match banner (only for random samples) */}
      {trueClass !== null && (
        <div className={`ground-truth-banner ${correct ? "gt-correct" : "gt-wrong"}`}>
          <span className="gt-icon">{correct ? "✓" : "✗"}</span>
          <span>
            <strong>Ground Truth:</strong> {TRUE_CLASS_LABELS[trueClass] || trueClass}
            {" — "}
            {correct
              ? "Model prediction is correct."
              : `Model predicted ${info.label} instead.`}
          </span>
        </div>
      )}

      {/* Images */}
      <div className="images-grid">
        <div className="image-card">
          <p className="image-caption">Original MRI</p>
          <img src={originalSrc} alt="Original MRI" className="scan-image" />
        </div>
        <div className="image-card">
          <p className="image-caption">
            ML Attention Heatmap
            <span className="caption-hint"> — red = where the model looked</span>
          </p>
          <img src={heatmapSrc} alt="Grad-CAM heatmap" className="scan-image" />
        </div>
      </div>

      {/* Plain-language explanation */}
      <div className="explanation-card" style={{ borderLeftColor: info.color }}>
        <p className="explanation-text">{info.description}</p>
        <p className="explanation-disclaimer">
          This is a machine learning prediction for educational purposes only. It is not a medical diagnosis.
        </p>
      </div>

      {/* Confidence bars */}
      <ConfidenceBars scores={scores} predictedClass={predicted_class} />

      {/* Model performance metrics */}
      <ModelMetrics />

      {/* Reset */}
      <div className="result-actions">
        <button className="btn-primary" onClick={onReset}>
          Try Another Scan
        </button>
      </div>
    </div>
  );
}
