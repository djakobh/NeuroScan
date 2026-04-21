import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function Uploader({ onFile }) {
  const [sampling, setSampling] = useState(false);

  async function handleRandomSample() {
    setSampling(true);
    try {
      const res = await fetch(`${API_URL}/random-sample?t=${Date.now()}`);
      if (!res.ok) throw new Error("Could not fetch sample");
      const trueClass = res.headers.get("X-Sample-Class");
      const blob = await res.blob();
      const file = new File([blob], `sample.${blob.type === "image/png" ? "png" : "jpg"}`, {
        type: blob.type,
      });
      onFile(file, trueClass);
    } catch (err) {
      alert("Failed to load a random sample: " + err.message);
    } finally {
      setSampling(false);
    }
  }

  return (
    <div className="uploader-wrapper">
      <div className="dropzone">
        <div className="dropzone-icon">&#129504;</div>
        <p className="dropzone-title">Try a random brain MRI from the dataset</p>
        <p className="dropzone-sub">Picks a random scan — the model classifies it live</p>
        <button
          className="btn-primary"
          type="button"
          onClick={handleRandomSample}
          disabled={sampling}
        >
          {sampling ? "Loading…" : "Analyze a Random Scan"}
        </button>
      </div>
    </div>
  );
}
