import React, { useState } from "react";
import axios from "axios";
import "./index.css";

// In dev, use "" so Vite proxies to backend; otherwise use env or fallback
const API_BASE: string =
  import.meta.env.VITE_API_BASE ??
  (import.meta.env.DEV ? "" : "http://127.0.0.1:5000");

type CredibilityAudit = {
  sensationalism: number;
  political_bias: {
    direction: string;
    score: number;
    confidence: number;
  };
  source_credibility: {
    score: number | null;
    domain: string | null;
    tier: string;
  };
  factuality_index: number;
  flagged_terms: Array<{ term: string; weight: number; reason: string }>;
};

type PredictionResponse = {
  text: string;
  prediction: string;
  label: number;
  confidence: number;
  model: string;
  latency_ms: number;
  within_latency_constraint: boolean;
  probabilities: Record<string, number>;
  credibility_audit?: CredibilityAudit;
  explanation?: {
    top_features: Array<[string, number]>;
  };
};

type InputMode = "text" | "url";

const App: React.FC = () => {
  const [mode, setMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [header, setHeader] = useState("");
  const [model, setModel] = useState<string>("tfidf_logistic");
  const [includeExplanation, setIncludeExplanation] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const canSubmit =
    !loading &&
    ((mode === "text" && text.trim().length > 0) ||
      (mode === "url" && url.trim().length > 0));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: any = { model, include_explanation: includeExplanation };
      if (mode === "text") {
        payload.text = text;
      } else {
        payload.url = url;
        if (header.trim()) {
          payload.header = header.trim();
        }
      }

      const response = await axios.post<PredictionResponse>(
        `${API_BASE}/predict`,
        payload
      );
      setResult(response.data);
    } catch (err: any) {
      const msg =
        err?.response?.data?.error ||
        err?.message ||
        "Request failed. Is the API running?";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-root">
      <header className="hero">
        <p className="hero-label">Credibility & misinformation detection</p>
        <h1 className="hero-title">
          <span className="line">Assess what you read.</span>
          <span className="line">Decide what you trust.</span>
        </h1>
        <a href="#analyse" className="hero-scroll">
          Scroll down
        </a>
      </header>

      <section id="analyse" className="section" aria-labelledby="analyse-heading">
        <p className="section-label">Our approach</p>
        <h2 id="analyse-heading" className="section-title">Analyse</h2>
        <p className="section-intro">
          Paste text, a headline, or a URL. We classify content and return a credibility audit with sensationalism, bias, factuality, and flagged terms.
        </p>
        <main className="app-main">
          <div className="card">
            <h2>Input</h2>
          <form onSubmit={handleSubmit} className="form">
            <div className="form-row">
              <label className="label-inline">
                Input mode
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value as InputMode)}
                >
                  <option value="text">Free text / headline</option>
                  <option value="url">URL + optional header</option>
                </select>
              </label>
            </div>

            {mode === "text" ? (
              <label className="label">
                Text to analyse
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste or type a news headline / post to classify..."
                  rows={6}
                />
              </label>
            ) : (
              <>
                <label className="label">
                  URL to analyse
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://example.com/news/article"
                  />
                </label>
                <label className="label">
                  Optional headline / context
                  <input
                    type="text"
                    value={header}
                    onChange={(e) => setHeader(e.target.value)}
                    placeholder="Article headline or short summary (optional)"
                  />
                </label>
              </>
            )}

            <div className="form-row">
              <label className="label-inline">
                Model
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  <option value="tfidf_logistic">TF‑IDF + Logistic</option>
                  <option value="naive_bayes">Naive Bayes</option>
                  <option value="bert">BERT (if trained)</option>
                </select>
              </label>

              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={includeExplanation}
                  onChange={(e) => setIncludeExplanation(e.target.checked)}
                  disabled={model === "bert"}
                />
                Include explanation (LIME)
              </label>
            </div>

            <button
              type="submit"
              className="primary-button"
              disabled={!canSubmit}
            >
              {loading ? "Predicting..." : "Run prediction"}
            </button>

            <p className="hint">
              API:{" "}
              <code>{API_BASE || "same origin (proxied to :5000)"}</code>
            </p>
          </form>
          </div>

          <div className="card">
            <h2>Result</h2>

          {!result && !error && (
            <p className="placeholder">
              Submit text or a URL to see the prediction.
            </p>
          )}

          {error && (
            <div className="alert alert-error">
              <strong>Error:</strong> {error}
            </div>
          )}

          {result && (
            <div className="result">
              <div className="result-header">
                <span className="pill">
                  Model: <strong>{result.model}</strong>
                </span>
                <span className="pill">
                  Latency: <strong>{result.latency_ms.toFixed(2)} ms</strong>
                </span>
                {!result.within_latency_constraint && (
                  <span className="pill pill-warn">Slow</span>
                )}
              </div>

              <div className="result-main">
                <div className="prediction-badge">
                  <span className="prediction-label">
                    {result.prediction}
                  </span>
                  <span className="prediction-confidence">
                    {(result.confidence * 100).toFixed(1)}% confidence
                  </span>
                </div>

                <div className="text-preview">
                  <h3>Analysed text</h3>
                  <p>{result.text}</p>
                </div>

                <div className="probabilities">
                  <h3>Class probabilities</h3>
                  <div className="prob-bars">
                    {Object.entries(result.probabilities).map(
                      ([label, prob]) => (
                        <div key={label} className="prob-bar-row">
                          <span className="prob-label">{label}</span>
                          <div className="prob-bar-track">
                            <div
                              className="prob-bar-fill"
                              style={{ width: `${prob * 100}%` }}
                            />
                          </div>
                          <span className="prob-value">
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      )
                    )}
                  </div>
                </div>

                {result.credibility_audit && (
                  <div className="credibility-audit">
                    <h3>Credibility audit</h3>
                    <div className="audit-metrics">
                      <div className="audit-metric">
                        <span className="audit-label">Sensationalism</span>
                        <div className="audit-bar-track">
                          <div
                            className="audit-bar-fill sensational"
                            style={{
                              width: `${result.credibility_audit.sensationalism * 100}%`,
                            }}
                          />
                        </div>
                        <span className="audit-value">
                          {(result.credibility_audit.sensationalism * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="audit-metric">
                        <span className="audit-label">Political bias</span>
                        <span className="audit-value">
                          {result.credibility_audit.political_bias.direction}{" "}
                          ({(result.credibility_audit.political_bias.score * 100).toFixed(0)}%)
                        </span>
                      </div>
                      <div className="audit-metric">
                        <span className="audit-label">Source credibility</span>
                        <span className="audit-value">
                          {result.credibility_audit.source_credibility.score != null ? (
                            <>
                              {result.credibility_audit.source_credibility.tier}{" "}
                              ({(result.credibility_audit.source_credibility.score * 100).toFixed(0)}%)
                              {result.credibility_audit.source_credibility.domain && (
                                <> — <code>{result.credibility_audit.source_credibility.domain}</code></>
                              )}
                            </>
                          ) : (
                            "N/A (no URL)"
                          )}
                        </span>
                      </div>
                      <div className="audit-metric">
                        <span className="audit-label">Factuality index</span>
                        <div className="audit-bar-track">
                          <div
                            className="audit-bar-fill factual"
                            style={{
                              width: `${result.credibility_audit.factuality_index * 100}%`,
                            }}
                          />
                        </div>
                        <span className="audit-value">
                          {(result.credibility_audit.factuality_index * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    {result.credibility_audit.flagged_terms.length > 0 && (
                      <div className="flagged-terms">
                        <h4>Flagged terms</h4>
                        <p className="hint">
                          Words that may reduce credibility:
                        </p>
                        <div className="flagged-list">
                          {result.credibility_audit.flagged_terms.map(
                            (ft) => (
                              <span
                                key={`${ft.term}-${ft.reason}`}
                                className="flagged-term"
                                title={`${ft.reason}, weight: ${ft.weight.toFixed(2)}`}
                              >
                                {ft.term}
                              </span>
                            )
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {result.explanation && (
                  <div className="explanation">
                    <h3>Top contributing features</h3>
                    <ul>
                      {result.explanation.top_features.map(
                        ([feature, weight]) => (
                          <li key={feature}>
                            <span className="feature">{feature}</span>
                            <span className="weight">
                              {weight.toFixed(3)}
                            </span>
                          </li>
                        )
                      )}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
          </div>
        </main>
      </section>

      <footer className="app-footer">
        <div className="app-footer-inner">
          <span>
            Backend: <code>Flask</code> &nbsp;·&nbsp; Frontend:{" "}
            <code>React + Vite</code>
          </span>
          <a href="#analyse" className="footer-link">
            Analyse
          </a>
        </div>
      </footer>
    </div>
  );
};

export default App;


