import React, { useEffect, useState } from "react";
import "./index.css";

const API_BASE: string = window.location.origin;

type PredictionResponse = {
  verdict: string;
  ensemble_probability: number;
  confidence_percent: number;
  model_breakdown: Record<string, { confidence: number }>;
  source_credibility?: { domain: string; score: number; label: string };
  llm_judge?: { verdict?: string; reasoning?: string; explanation?: string };
  text?: string;
  scraped_url?: string;
  scraped_preview?: string;
  scraped_word_count?: number;
  scraped_char_count?: number;
  explanation?: {
    summary?: {
      misinfo_indicators?: string[];
      credible_indicators?: string[];
    };
  };
};

const App: React.FC = () => {
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [showWordAnalysis, setShowWordAnalysis] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [sessionCount, setSessionCount] = useState(0);
  const [scrapeInfo, setScrapeInfo] = useState<string>("");
  const [llmAvailable, setLlmAvailable] = useState(false);

  const canPredictText = !loading && text.trim().length > 0;
  const canPredictUrl = !loading && url.trim().length > 0;

  const loadLlmStatus = async () => {
    try {
      const r = await fetch(`${API_BASE}/llm-status`);
      const data = await r.json();
      setLlmAvailable(Boolean(data.llm_available));
    } catch {
      setLlmAvailable(false);
    }
  };

  useEffect(() => {
    loadLlmStatus();
  }, []);

  const runPredictText = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canPredictText) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setScrapeInfo("");

    try {
      const body: any = {
        text: text.trim(),
        explain: showWordAnalysis,
      };
      if (url.trim()) body.url = url.trim();
      const r = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "API error");
      setResult(data);
      setSessionCount((v) => v + 1);
    } catch (err: any) {
      setError(err?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const runScrapePredict = async () => {
    if (!canPredictUrl) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await fetch(`${API_BASE}/scrape-and-predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: url.trim(),
          explain: showWordAnalysis,
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "API error");
      setResult(data);
      setSessionCount((v) => v + 1);
      setScrapeInfo(
        `Analysed ${data.scraped_word_count ?? 0} words from ${data.scraped_url ?? url.trim()}`
      );
    } catch (err: any) {
      setError(err?.message || "Request failed");
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
          Paste text or submit only a URL. The app supports source credibility, explainability,
          model breakdown, URL scraping, and local LLM judge status.
        </p>
        <main className="app-main">
          <div className="card">
            <h2>Input</h2>
          <form onSubmit={runPredictText} className="form">
            <label className="label">
              Text to analyse
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste or type text to classify..."
                rows={6}
              />
            </label>
            <label className="label">
              Source URL
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/news/article"
              />
            </label>
            <div className="form-row">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={showWordAnalysis}
                  onChange={(e) => setShowWordAnalysis(e.target.checked)}
                />
                Show word analysis (highlight words driving prediction)
              </label>
              <span className={`pill ${llmAvailable ? "" : "pill-warn"}`}>
                LLM Judge: <strong>{llmAvailable ? "Active" : "Offline"}</strong>
              </span>
            </div>

            <button type="submit" className="primary-button" disabled={!canPredictText}>
              {loading ? "Predicting..." : "Run prediction"}
            </button>
            <button
              type="button"
              className="primary-button"
              disabled={!canPredictUrl}
              onClick={runScrapePredict}
            >
              {loading ? "Scraping..." : "Analyse URL only"}
            </button>

            <p className="hint">
              API: <code>{API_BASE}</code>
            </p>
            <p className="hint">Texts analysed this session: <strong>{sessionCount}</strong></p>
            {scrapeInfo && <p className="hint">{scrapeInfo}</p>}
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
                  Model: <strong>BERT + TF-IDF + NB Ensemble</strong>
                </span>
              </div>

              <div className="result-main">
                <div className="prediction-badge">
                  <span className="prediction-label">
                    {result.verdict}
                  </span>
                  <span className="prediction-confidence">
                    {(result.ensemble_probability * 100).toFixed(1)}% misinformation
                  </span>
                </div>

                {result.source_credibility && (
                  <p className="hint">
                    Source credibility: {result.source_credibility.domain} (
                    {result.source_credibility.label} - {result.source_credibility.score.toFixed(2)})
                  </p>
                )}

                <div className="probabilities">
                  <h3>Model breakdown</h3>
                  <div className="prob-bars">
                    {["bert", "tfidf", "naive_bayes"].map((k) => (
                      <div key={k} className="prob-bar-row">
                        <span className="prob-label">{k.toUpperCase()}</span>
                        <div className="prob-bar-track">
                          <div
                            className="prob-bar-fill"
                            style={{
                              width: `${((result.model_breakdown?.[k]?.confidence ?? 0) * 100).toFixed(1)}%`,
                            }}
                          />
                        </div>
                        <span className="prob-value">
                          {(result.model_breakdown?.[k]?.confidence ?? 0).toFixed(3)}
                        </span>
                      </div>
                    ))}
                    <div className="prob-bar-row">
                      <span className="prob-label">ENSEMBLE</span>
                      <div className="prob-bar-track">
                        <div
                          className="prob-bar-fill"
                          style={{ width: `${(result.ensemble_probability * 100).toFixed(1)}%` }}
                        />
                      </div>
                      <span className="prob-value">{result.ensemble_probability.toFixed(3)}</span>
                    </div>
                  </div>
                </div>

                {!!result.scraped_preview && (
                  <div className="text-preview">
                    <h3>Scraped preview</h3>
                    <p>{result.scraped_preview}</p>
                  </div>
                )}

                {!!result.llm_judge && llmAvailable && (
                  <div className="explanation">
                    <h3>LLM Judge</h3>
                    <p>
                      {(result.llm_judge.reasoning ||
                        result.llm_judge.explanation ||
                        result.llm_judge.verdict ||
                        "Reasoning available")}
                    </p>
                  </div>
                )}

                {result.explanation?.summary && (
                  <div className="explanation">
                    <h3>Word analysis</h3>
                    <div className="flagged-list">
                      {(result.explanation.summary.misinfo_indicators || []).map((w) => (
                        <span key={`mis-${w}`} className="flagged-term">
                          {w}
                        </span>
                      ))}
                    </div>
                    <div className="flagged-list" style={{ marginTop: "0.75rem" }}>
                      {(result.explanation.summary.credible_indicators || []).map((w) => (
                        <span
                          key={`cred-${w}`}
                          className="flagged-term"
                          style={{
                            background: "rgba(93, 232, 154, 0.12)",
                            borderColor: "rgba(93, 232, 154, 0.3)",
                            color: "#9af2bf",
                          }}
                        >
                          {w}
                        </span>
                      ))}
                    </div>
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
            Backend: <code>FastAPI</code> &nbsp;·&nbsp; Frontend:{" "}
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


