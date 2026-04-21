const CLASS_COLORS = {
  glioma:     "#ef4444",
  meningioma: "#f97316",
  notumor:    "#22c55e",
  pituitary:  "#6366f1",
};

// Metrics from the held-out test set (1,600 scans, 400 per class)
const PER_CLASS = [
  { key: "glioma",     label: "Glioma",     precision: 97, recall: 83, f1: 89 },
  { key: "meningioma", label: "Meningioma", precision: 90, recall: 96, f1: 93 },
  { key: "notumor",    label: "No Tumor",   precision: 99, recall: 99, f1: 99 },
  { key: "pituitary",  label: "Pituitary",  precision: 97, recall: 99, f1: 98 },
];

export default function ModelMetrics() {
  return (
    <div className="metrics-panel">
      <h3 className="metrics-title">Model Performance — Held-Out Test Set</h3>

      {/* Top-level stats */}
      <div className="metrics-summary">
        <div className="metric-stat">
          <span className="metric-value">95%</span>
          <span className="metric-label">Overall Accuracy</span>
        </div>
        <div className="metric-divider" />
        <div className="metric-stat">
          <span className="metric-value">1,600</span>
          <span className="metric-label">Test Scans</span>
        </div>
        <div className="metric-divider" />
        <div className="metric-stat">
          <span className="metric-value">20</span>
          <span className="metric-label">Epochs</span>
        </div>
        <div className="metric-divider" />
        <div className="metric-stat">
          <span className="metric-value" style={{ fontSize: "1rem" }}>EfficientNetB0</span>
          <span className="metric-label">Architecture</span>
        </div>
      </div>

      {/* Per-class breakdown */}
      <div className="metrics-table">
        <div className="metrics-table-head">
          <span>Class</span>
          <span>Precision</span>
          <span>Recall</span>
          <span>F1</span>
        </div>
        {PER_CLASS.map(({ key, label, precision, recall, f1 }) => (
          <div key={key} className="metrics-table-row">
            <span className="metrics-class-label" style={{ color: CLASS_COLORS[key] }}>
              {label}
            </span>
            <span>{precision}%</span>
            <span>{recall}%</span>
            <span>{f1}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
