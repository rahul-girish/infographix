import { useEffect, useState, type ChangeEvent, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowDown,
  ArrowUp,
  Cpu,
  Database,
  Download,
  FileImage,
  FileText,
  FlaskConical,
  Loader2,
  Plus,
  Sparkles,
  Table2,
  Trash2,
} from "lucide-react";
import { assetsService } from "../services/assets";
import { postersService } from "../services/posters";
import { useEditorStore } from "../stores/editorStore";
import type {
  ContentBlockInput,
  EngineType,
  GenerationRunSummary,
  Orientation,
} from "../types";

const STARTER_BLOCKS: ContentBlockInput[] = [
  {
    id: "intro",
    type: "section",
    heading: "1. Introduction",
    text: "State the research motivation, problem gap, and core contribution concisely for conference attendees.",
  },
  {
    id: "methods",
    type: "section",
    heading: "2. Methodology",
    text: "Describe your model architecture, dataset pipeline, and optimization strategy in clear steps.",
  },
  {
    id: "fig_main",
    type: "figure",
    heading: "Architecture / Primary Result",
    aspectRatio: 1.45,
    caption: "Figure 1: System pipeline and layout quality comparison.",
  },
  {
    id: "results",
    type: "section",
    heading: "3. Key Findings & Conclusion",
    text: "Summarize quantitative improvements, failure modes, and main takeaways.",
  },
];

const ACCENT_COLORS = ["#3458e8", "#7048db", "#087f9b", "#14866d", "#334155"];

export default function CreatePosterPage() {
  const navigate = useNavigate();
  const setRun = useEditorStore((state) => state.setRun);

  const [title, setTitle] = useState(
    "Constraint-Aware Layout Synthesis for Scientific Posters"
  );
  const [authors, setAuthors] = useState("Elena Rostova, Marcus Vance");
  const [affiliations, setAffiliations] = useState("Infographix Research Lab");
  const [orientation, setOrientation] = useState<Orientation>("landscape");
  const [preferredColumns, setPreferredColumns] = useState<number | null>(null);
  const [accentColor, setAccentColor] = useState("#3458e8");
  const [engine, setEngine] = useState<EngineType>("adaptation");
  const [blocks, setBlocks] = useState<ContentBlockInput[]>(STARTER_BLOCKS);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [recentRuns, setRecentRuns] = useState<GenerationRunSummary[]>([]);

  useEffect(() => {
    void postersService
      .listRuns(8)
      .then(setRecentRuns)
      .catch(() => {
        /* ignore if backend isn't reachable yet */
      });
  }, []);

  function updateBlock(id: string, patch: Partial<ContentBlockInput>) {
    setBlocks((prev) => prev.map((b) => (b.id === id ? { ...b, ...patch } : b)));
  }

  function addBlock(type: ContentBlockInput["type"]) {
    const id = `${type}_${crypto.randomUUID().slice(0, 6)}`;
    if (type === "figure") {
      setBlocks((prev) => [
        ...prev,
        {
          id,
          type: "figure",
          heading: "Figure",
          aspectRatio: 1.33,
          caption: "Figure caption",
        },
      ]);
    } else if (type === "table") {
      setBlocks((prev) => [
        ...prev,
        {
          id,
          type: "table",
          heading: "Quantitative Results Table",
          tableRows: 4,
          tableCols: 3,
          caption: "Table 1: Benchmark metrics.",
        },
      ]);
    } else {
      setBlocks((prev) => [
        ...prev,
        {
          id,
          type,
          heading: type === "list" ? "Key Highlights" : "New Section",
          text: "",
        },
      ]);
    }
  }

  function moveBlock(index: number, delta: -1 | 1) {
    const target = index + delta;
    if (target < 0 || target >= blocks.length) return;
    setBlocks((prev) => {
      const next = [...prev];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }

  function removeBlock(id: string) {
    setBlocks((prev) => prev.filter((b) => b.id !== id));
  }

  async function handleFigureImage(blockId: string, event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const { dataUrl, aspectRatio } = await assetsService.readImageWithMetadata(file);
      updateBlock(blockId, {
        imageDataUrl: dataUrl,
        aspectRatio,
        heading: file.name.replace(/\.[^.]+$/, ""),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load image.");
    } finally {
      event.target.value = "";
    }
  }

  async function handleGenerate(event: FormEvent) {
    event.preventDefault();
    if (!title.trim()) {
      setError("Please enter a poster title.");
      return;
    }
    setError("");
    setSubmitting(true);
    try {
      const run = await postersService.generate({
        title: title.trim(),
        authors: authors.trim(),
        affiliations: affiliations.trim(),
        orientation,
        preferredColumns,
        accentColor,
        background: "#ffffff",
        engine,
        blocks,
      });
      setRun(run);
      navigate(`/editor/${run.id}`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Layout generation failed. Check that the backend server is running."
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="workspace-page">
      <header className="workspace-header">
        <div className="brand-lockup">
          <span className="brand-mark">
            <Sparkles size={18} />
          </span>
          <div>
            <strong>Infographix</strong>
            <small>AI Poster Layout Pipeline &amp; Editor</small>
          </div>
        </div>
        <div className="workspace-header-actions">
          <a
            className="secondary-button"
            href={postersService.getTrainingExportUrl("jsonl")}
            download="infographix_training_runs.jsonl"
          >
            <Download size={14} /> Export Training Data (.jsonl)
          </a>
        </div>
      </header>

      <main className="create-main">
        <form className="create-form-column" onSubmit={(e) => void handleGenerate(e)}>
          <div className="flow-heading compact-heading">
            <span className="section-kicker">MODEL INFERENCE &amp; DATA COLLECTION</span>
            <h1>Enter poster content</h1>
            <p>
              Provide your poster text, figures, and metadata. The AI pipeline retrieves a
              compatible layout, optimizes placement, logs inputs/outputs to Supabase, and opens
              the editor.
            </p>
          </div>

          <section className="config-block">
            <div className="config-block-heading">
              <span className="config-icon">
                <FileText size={18} />
              </span>
              <div>
                <h2>Poster details</h2>
                <p>Core header metadata fed into the layout model.</p>
              </div>
            </div>

            <div className="create-field-stack">
              <label className="create-label">
                <span>Poster title *</span>
                <input
                  required
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter your scientific poster title"
                />
              </label>

              <div className="create-two-col">
                <label className="create-label">
                  <span>Authors</span>
                  <input
                    value={authors}
                    onChange={(e) => setAuthors(e.target.value)}
                    placeholder="Jane Doe, John Smith"
                  />
                </label>
                <label className="create-label">
                  <span>Affiliations</span>
                  <input
                    value={affiliations}
                    onChange={(e) => setAffiliations(e.target.value)}
                    placeholder="University / Research Lab"
                  />
                </label>
              </div>
            </div>
          </section>

          <section className="config-block">
            <div className="config-block-heading">
              <span className="config-icon">
                <Cpu size={18} />
              </span>
              <div>
                <h2>Layout &amp; model configuration</h2>
                <p>Choose canvas geometry and which layout engine to test.</p>
              </div>
            </div>

            <div className="create-two-col">
              <div className="create-label">
                <span>Orientation</span>
                <div className="pill-toggle-row">
                  {(["landscape", "portrait"] as const).map((item) => (
                    <button
                      key={item}
                      type="button"
                      className={orientation === item ? "pill-toggle active" : "pill-toggle"}
                      onClick={() => setOrientation(item)}
                    >
                      {item === "landscape" ? "Landscape (1000×700)" : "Portrait (700×1000)"}
                    </button>
                  ))}
                </div>
              </div>

              <div className="create-label">
                <span>Columns</span>
                <div className="pill-toggle-row">
                  {[null, 2, 3, 4].map((col) => (
                    <button
                      key={String(col)}
                      type="button"
                      className={preferredColumns === col ? "pill-toggle active" : "pill-toggle"}
                      onClick={() => setPreferredColumns(col)}
                    >
                      {col === null ? "Auto" : `${col} Cols`}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="create-two-col" style={{ marginTop: 14 }}>
              <div className="create-label">
                <span>Model engine</span>
                <div className="pill-toggle-row">
                  <button
                    type="button"
                    className={engine === "adaptation" ? "pill-toggle active" : "pill-toggle"}
                    onClick={() => setEngine("adaptation")}
                  >
                    <FlaskConical size={13} /> Retrieval + HiGHS LP
                  </button>
                  <button
                    type="button"
                    className={engine === "neural" ? "pill-toggle active" : "pill-toggle"}
                    onClick={() => setEngine("neural")}
                  >
                    <Cpu size={13} /> Neural Layout Model
                  </button>
                </div>
              </div>

              <div className="create-label">
                <span>Accent color</span>
                <div className="color-options" style={{ marginTop: 4 }}>
                  {ACCENT_COLORS.map((color) => (
                    <button
                      key={color}
                      type="button"
                      aria-label={`Accent ${color}`}
                      className={`color-swatch ${accentColor === color ? "active" : ""}`}
                      style={{ backgroundColor: color }}
                      onClick={() => setAccentColor(color)}
                    />
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section className="config-block">
            <div className="config-block-heading">
              <span className="config-icon">
                <Sparkles size={18} />
              </span>
              <div>
                <h2>Content blocks ({blocks.length})</h2>
                <p>Add text sections, figures, or tables for the model to arrange.</p>
              </div>
            </div>

            <div className="review-sections">
              {blocks.map((block, index) => (
                <div key={block.id} className="review-section">
                  <div className="section-order">
                    <span>{index + 1}</span>
                    <button
                      type="button"
                      title="Move up"
                      disabled={index === 0}
                      onClick={() => moveBlock(index, -1)}
                    >
                      <ArrowUp size={13} />
                    </button>
                    <button
                      type="button"
                      title="Move down"
                      disabled={index === blocks.length - 1}
                      onClick={() => moveBlock(index, 1)}
                    >
                      <ArrowDown size={13} />
                    </button>
                  </div>

                  <div className="section-fields">
                    <div className="block-type-row">
                      <span className="block-type-pill">{block.type.toUpperCase()}</span>
                      <input
                        className="section-title-input"
                        value={block.heading || ""}
                        onChange={(e) => updateBlock(block.id, { heading: e.target.value })}
                        placeholder={
                          block.type === "figure"
                            ? "Figure title / label"
                            : block.type === "table"
                            ? "Table title"
                            : "Section heading"
                        }
                      />
                    </div>

                    {(block.type === "section" ||
                      block.type === "text" ||
                      block.type === "list") && (
                      <textarea
                        rows={3}
                        value={block.text || ""}
                        onChange={(e) => updateBlock(block.id, { text: e.target.value })}
                        placeholder="Enter section text (word count directly drives layout sizing)..."
                      />
                    )}

                    {block.type === "figure" && (
                      <div className="figure-input-grid">
                        <label className="secondary-button">
                          <FileImage size={14} />{" "}
                          {block.imageDataUrl ? "Change image" : "Attach image (optional)"}
                          <input
                            className="visually-hidden"
                            type="file"
                            accept="image/*"
                            onChange={(e) => void handleFigureImage(block.id, e)}
                          />
                        </label>
                        <label className="inline-mini-field">
                          <span>Aspect ratio</span>
                          <input
                            type="number"
                            step="0.05"
                            min="0.3"
                            max="4"
                            value={block.aspectRatio ?? 1.33}
                            onChange={(e) =>
                              updateBlock(block.id, {
                                aspectRatio: Number(e.target.value) || 1.33,
                              })
                            }
                          />
                        </label>
                        <input
                          className="caption-input"
                          value={block.caption || ""}
                          onChange={(e) => updateBlock(block.id, { caption: e.target.value })}
                          placeholder="Figure caption (optional)"
                        />
                      </div>
                    )}

                    {block.type === "table" && (
                      <div className="figure-input-grid">
                        <label className="inline-mini-field">
                          <span>Rows</span>
                          <input
                            type="number"
                            min="1"
                            max="20"
                            value={block.tableRows ?? 4}
                            onChange={(e) =>
                              updateBlock(block.id, { tableRows: Number(e.target.value) || 4 })
                            }
                          />
                        </label>
                        <label className="inline-mini-field">
                          <span>Cols</span>
                          <input
                            type="number"
                            min="1"
                            max="12"
                            value={block.tableCols ?? 3}
                            onChange={(e) =>
                              updateBlock(block.id, { tableCols: Number(e.target.value) || 3 })
                            }
                          />
                        </label>
                        <input
                          className="caption-input"
                          value={block.caption || ""}
                          onChange={(e) => updateBlock(block.id, { caption: e.target.value })}
                          placeholder="Table caption (optional)"
                        />
                      </div>
                    )}
                  </div>

                  <div className="section-actions">
                    <button
                      type="button"
                      className="remove-section"
                      title="Remove block"
                      onClick={() => removeBlock(block.id)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="add-block-bar">
              <button
                type="button"
                className="add-section-button"
                onClick={() => addBlock("section")}
              >
                <Plus size={14} /> Add text section
              </button>
              <button
                type="button"
                className="add-section-button"
                onClick={() => addBlock("figure")}
              >
                <FileImage size={14} /> Add figure
              </button>
              <button
                type="button"
                className="add-section-button"
                onClick={() => addBlock("table")}
              >
                <Table2 size={14} /> Add table
              </button>
            </div>
          </section>

          {error && (
            <div className="form-alert error-alert" role="alert">
              {error}
            </div>
          )}

          <div className="flow-actions">
            <button
              type="button"
              className="text-button"
              onClick={() => setBlocks(STARTER_BLOCKS)}
            >
              Reset starter blocks
            </button>
            <button type="submit" className="primary-button" disabled={submitting}>
              {submitting ? (
                <>
                  <Loader2 size={16} className="spin" /> Running layout model…
                </>
              ) : (
                <>
                  <Sparkles size={16} /> Generate &amp; Open Editor
                </>
              )}
            </button>
          </div>
        </form>

        <aside className="runs-sidebar">
          <div className="runs-card">
            <div className="runs-card-header">
              <Database size={16} />
              <div>
                <strong>Recent Supabase Runs</strong>
                <small>Logged inputs + generated/edited layouts</small>
              </div>
            </div>
            {recentRuns.length === 0 ? (
              <p className="runs-empty">
                No runs recorded yet. Generate a poster to log your first MLOps training sample.
              </p>
            ) : (
              <div className="runs-list">
                {recentRuns.map((run) => (
                  <button
                    key={run.id}
                    type="button"
                    className="run-row"
                    onClick={() => navigate(`/editor/${run.id}`)}
                  >
                    <strong>{run.title}</strong>
                    <div className="run-meta">
                      <span>{run.engine}</span>
                      {run.retrievedTemplateId && <span>tmpl #{run.retrievedTemplateId}</span>}
                      {run.wasEdited && <span className="edited-badge">edited</span>}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
