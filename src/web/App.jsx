import { useState } from "react";
import DotGrid    from "./components/DotGrid";
import ResultView from "./components/ResultView";
import HowItWorks from "./components/HowItWorks";
import ModelMetrics from "./components/ModelMetrics";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [status,   setStatus]   = useState("idle");
  const [result,   setResult]   = useState(null);
  const [preview,  setPreview]  = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [trueClass,setTrueClass]= useState(null);
  const [loading,  setLoading]  = useState(false);

  async function analyzeRandom() {
    setLoading(true);
    try {
      const sampleRes = await fetch(`${API_URL}/random-sample?t=${Date.now()}`);
      if (!sampleRes.ok) throw new Error("Could not fetch sample");
      const sampleClass = sampleRes.headers.get("X-Sample-Class");
      const blob  = await sampleRes.blob();
      const file  = new File([blob], `sample.${blob.type === "image/png" ? "png" : "jpg"}`, { type: blob.type });

      setPreview(URL.createObjectURL(file));
      setTrueClass(sampleClass);
      setStatus("loading");
      setResult(null);

      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_URL}/explain`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }
      setResult(await res.json());
      setStatus("result");
    } catch (e) {
      setErrorMsg(e.message);
      setStatus("error");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    window.scrollTo({ top: 0, behavior: "instant" });
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setErrorMsg("");
    setTrueClass(null);
    setLoading(false);
  }

  /* ── Landing ─────────────────────────────────────────────── */
  if (status === "idle") {
    return (
      <div className="landing">
        <DotGrid />

        {/* Hero — full viewport */}
        <section className="landing-hero">
          <div className="landing-content">
            <span className="landing-eyebrow">NeuroScan ML</span>

            <h1 className="landing-title">
              Can We Use <br />Machine Learning <br />to Read<br />a Brain Scan?
            </h1>

            <p className="landing-sub">
              Watch the model classify a real MRI in real time.<br />
              The heatmap shows exactly where it looked.
            </p>

            <button
              className={`landing-cta ${loading ? "landing-cta--busy" : ""}`}
              onClick={analyzeRandom}
              disabled={loading}
            >
              {loading ? (
                <span className="cta-spinner" />
              ) : (
                <>
                  Analyze a Random Brain Scan
                  <svg className="cta-arrow" viewBox="0 0 16 16" fill="none">
                    <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </>
              )}
            </button>
          </div>

          <div className="landing-scroll-hint">
            <span>scroll to learn more</span>
            <svg viewBox="0 0 16 16" fill="none" width="12" height="12">
              <path d="M8 3v10M4 9l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        </section>

        {/* How it works — below the fold */}
        <section className="landing-how">
          <HowItWorks />
        </section>
      </div>
    );
  }

  /* ── Active states ───────────────────────────────────────── */
  return (
    <div className="app">
      <DotGrid />

      <main className="main">
        <div className="result-eyebrow-wrap">
          <span className="landing-eyebrow">NeuroScan ML</span>
        </div>

        {status === "loading" && (
          <div className="loading-card">
            <div className="spinner" />
            <p className="loading-text">Analyzing scan&hellip;</p>
            <p className="loading-sub">Running inference + generating heatmap</p>
          </div>
        )}

        {status === "error" && (
          <div className="error-card">
            <p className="error-title">Something went wrong</p>
            <p className="error-msg">{errorMsg}</p>
            <button className="btn-primary" onClick={reset}>Try Again</button>
          </div>
        )}

        {status === "result" && result && (
          <ResultView
            result={result}
            originalSrc={preview}
            trueClass={trueClass}
            onReset={reset}
          />
        )}

        <section className="landing-how">
          <HowItWorks />
        </section>
      </main>

      <footer className="footer">
        <p>
          EfficientNetB0 &middot; 95% test accuracy &middot; 1,600 held-out MRI scans &middot;{" "}
          <span className="footer-note">For educational purposes only — not a medical device</span>
        </p>
      </footer>
    </div>
  );
}
