import { useEffect, useState } from "react";

const CLASS_META = {
  glioma:     { label: "Glioma",      color: "#ef4444" },
  meningioma: { label: "Meningioma",  color: "#f97316" },
  notumor:    { label: "No Tumor",    color: "#22c55e" },
  pituitary:  { label: "Pituitary",   color: "#6366f1" },
};

export default function ConfidenceBars({ scores, predictedClass }) {
  const [animated, setAnimated] = useState(false);

  // Trigger animation on mount
  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 50);
    return () => clearTimeout(t);
  }, []);

  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);

  return (
    <div className="confidence-bars">
      <h3 className="bars-title">Confidence Scores</h3>
      {sorted.map(([cls, prob]) => {
        const meta = CLASS_META[cls] || { label: cls, color: "#94a3b8" };
        const pct = (prob * 100).toFixed(1);
        const isTop = cls === predictedClass;

        return (
          <div key={cls} className={`bar-row ${isTop ? "bar-row-top" : ""}`}>
            <div className="bar-label">
              <span className="bar-class">{meta.label}</span>
              <span className="bar-pct" style={{ color: isTop ? meta.color : "#94a3b8" }}>
                {pct}%
              </span>
            </div>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: animated ? `${pct}%` : "0%",
                  backgroundColor: meta.color,
                  opacity: isTop ? 1 : 0.4,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
