import { useRef, type ChangeEvent } from "react";
import {
  AlignCenter,
  AlignLeft,
  AlignRight,
  ArrowDown,
  ArrowUp,
  Bold,
  Circle,
  Eye,
  EyeOff,
  FolderOpen,
  FolderPlus,
  ImagePlus,
  Italic,
  Lock,
  LockKeyholeOpen,
  Minus,
  Plus,
  RectangleHorizontal,
  Trash2,
  Type,
  Underline,
} from "lucide-react";
import { assetsService } from "../../services/assets";
import { useEditorStore } from "../../stores/editorStore";
import type { PosterElement } from "../../types";

function newText(
  name: string,
  text: string,
  fontSize: number,
  weight: "normal" | "bold" = "normal"
): PosterElement {
  return {
    id: crypto.randomUUID(),
    type: "text",
    name,
    text,
    x: 90,
    y: 180,
    width: 320,
    height: fontSize * 2.5,
    rotation: 0,
    layer: 0,
    visible: true,
    locked: false,
    fill: "#26334b",
    fontFamily: "Arial",
    fontSize,
    fontWeight: weight,
    align: "left",
    lineHeight: 1.3,
  };
}

const ACCENT_PRESETS = [
  ["Indigo", "#3458e8"],
  ["Violet", "#7048db"],
  ["Teal", "#087f9b"],
  ["Emerald", "#14866d"],
  ["Slate", "#334155"],
];

export function EditorSidebar({ onError }: { onError: (message: string) => void }) {
  const imageInput = useRef<HTMLInputElement>(null);
  const addElement = useEditorStore((state) => state.addElement);
  const updatePoster = useEditorStore((state) => state.updatePoster);
  const document = useEditorStore((state) => state.document);
  const selectedId = useEditorStore((state) => state.selectedId);

  async function addImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const { dataUrl, width, height } = await assetsService.readImageWithMetadata(file);
      const scale = Math.min(300 / width, 240 / height, 1);
      addElement({
        id: crypto.randomUUID(),
        type: "image",
        name: file.name,
        src: dataUrl,
        x: 100,
        y: 180,
        width: width * scale,
        height: height * scale,
        rotation: 0,
        layer: 0,
        visible: true,
        locked: false,
        opacity: 1,
        imageFit: "contain",
      });
    } catch (error) {
      onError(error instanceof Error ? error.message : "Could not add this image.");
    } finally {
      event.target.value = "";
    }
  }

  return (
    <aside className="editor-sidebar">
      <div className="editor-panel-title">
        <strong>Canvas Tools</strong>
        <small>Adjust layout or add blocks</small>
      </div>
      <div className="sidebar-scroll">
        <div className="editor-sidebar-section">
          <span className="sidebar-label">HEADER ACCENT</span>
          <div className="sidebar-template-list">
            {ACCENT_PRESETS.map(([name, color]) => (
              <button
                key={color}
                className={
                  document?.accentColor === color
                    ? "sidebar-template active"
                    : "sidebar-template"
                }
                onClick={() => {
                  updatePoster({ accentColor: color });
                  const header = document?.elements.find(
                    (element) => element.name === "Header background"
                  );
                  if (header) {
                    useEditorStore.getState().updateElement(header.id, { fill: color });
                  }
                }}
              >
                <span style={{ background: color }} />
                <span>{name}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="editor-sidebar-section">
          <span className="sidebar-label">ADD ELEMENT</span>
          <div className="editor-tool-grid">
            <button onClick={() => addElement(newText("Text", "Add your text", 14))}>
              <Type size={17} /> Text
            </button>
            <button
              onClick={() => addElement(newText("Heading", "SECTION HEADING", 18, "bold"))}
            >
              <Bold size={16} /> Heading
            </button>
            <button onClick={() => imageInput.current?.click()}>
              <ImagePlus size={17} /> Image
            </button>
            <button
              onClick={() =>
                addElement({
                  id: crypto.randomUUID(),
                  type: "rect",
                  name: "Card",
                  x: 120,
                  y: 160,
                  width: 180,
                  height: 110,
                  rotation: 0,
                  layer: 0,
                  visible: true,
                  locked: false,
                  fill: "#e9edff",
                  stroke: "#3458e8",
                  strokeWidth: 2,
                  opacity: 1,
                })
              }
            >
              <RectangleHorizontal size={17} /> Box
            </button>
            <button
              onClick={() =>
                addElement({
                  id: crypto.randomUUID(),
                  type: "circle",
                  name: "Circle",
                  x: 150,
                  y: 180,
                  width: 110,
                  height: 110,
                  rotation: 0,
                  layer: 0,
                  visible: true,
                  locked: false,
                  fill: "#ddd6fe",
                  stroke: "#7048db",
                  strokeWidth: 2,
                  opacity: 1,
                })
              }
            >
              <Circle size={17} /> Circle
            </button>
            <button
              onClick={() =>
                addElement({
                  id: crypto.randomUUID(),
                  type: "line",
                  name: "Line",
                  x: 100,
                  y: 200,
                  width: 180,
                  height: 0,
                  rotation: 0,
                  layer: 0,
                  visible: true,
                  locked: false,
                  fill: "#3458e8",
                  stroke: "#3458e8",
                  strokeWidth: 4,
                  points: [0, 0, 180, 0],
                })
              }
            >
              <Minus size={17} /> Line
            </button>
            <button
              onClick={() =>
                addElement({
                  id: crypto.randomUUID(),
                  type: "arrow",
                  name: "Arrow",
                  x: 100,
                  y: 230,
                  width: 180,
                  height: 0,
                  rotation: 0,
                  layer: 0,
                  visible: true,
                  locked: false,
                  fill: "#3458e8",
                  stroke: "#3458e8",
                  strokeWidth: 4,
                  points: [0, 0, 180, 0],
                })
              }
            >
              <ArrowUp size={17} /> Arrow
            </button>
          </div>
          <input
            ref={imageInput}
            className="visually-hidden"
            type="file"
            accept="image/*"
            onChange={(event) => void addImage(event)}
          />
        </div>

        <div className="editor-sidebar-section">
          <span className="sidebar-label">
            LAYERS <small>{document?.elements.length || 0}</small>
          </span>
          <div className="layers-list">
            {[...(document?.elements || [])]
              .sort((a, b) => b.layer - a.layer)
              .map((element) => (
                <div
                  key={element.id}
                  className={`layer-row ${selectedId === element.id ? "selected" : ""}`}
                >
                  <button
                    className="layer-name"
                    onClick={(event) =>
                      useEditorStore.getState().select(element.id, event.shiftKey)
                    }
                  >
                    <span className="layer-type-dot" />
                    {element.name}
                  </button>
                  <button
                    title={element.visible ? "Hide layer" : "Show layer"}
                    onClick={() =>
                      useEditorStore
                        .getState()
                        .setElementVisibility(element.id, !element.visible)
                    }
                  >
                    {element.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                  </button>
                  <button
                    title={element.locked ? "Unlock layer" : "Lock layer"}
                    onClick={() =>
                      useEditorStore
                        .getState()
                        .setElementLock(element.id, !element.locked)
                    }
                  >
                    {element.locked ? <Lock size={14} /> : <LockKeyholeOpen size={14} />}
                  </button>
                </div>
              ))}
          </div>
        </div>
      </div>
    </aside>
  );
}

export function EditorProperties({ onError }: { onError: (message: string) => void }) {
  const document = useEditorStore((state) => state.document);
  const selectedId = useEditorStore((state) => state.selectedId);
  const selectedIds = useEditorStore((state) => state.selectedIds);
  const updateElement = useEditorStore((state) => state.updateElement);
  const removeElement = useEditorStore((state) => state.removeElement);
  const duplicateElement = useEditorStore((state) => state.duplicateElement);
  const reorderElement = useEditorStore((state) => state.reorderElement);
  const groupSelected = useEditorStore((state) => state.groupSelected);
  const ungroupSelected = useEditorStore((state) => state.ungroupSelected);
  const setElementLock = useEditorStore((state) => state.setElementLock);
  const element = document?.elements.find((item) => item.id === selectedId);
  const change = (update: Partial<PosterElement>) => {
    if (element) updateElement(element.id, update);
  };
  const numberField = (
    label: string,
    key: "x" | "y" | "width" | "height" | "rotation",
    min = 0
  ) => (
    <label className="property-field">
      <span>{label}</span>
      <input
        type="number"
        value={Math.round(element?.[key] ?? 0)}
        min={min}
        onChange={(event) => change({ [key]: Number(event.target.value) })}
      />
    </label>
  );
  const imageInput = useRef<HTMLInputElement>(null);

  async function replaceImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !element) return;
    try {
      const dataUrl = await assetsService.readImage(file);
      change({ src: dataUrl, name: file.name, assetId: crypto.randomUUID() });
    } catch (error) {
      onError(error instanceof Error ? error.message : "Could not replace the image.");
    } finally {
      event.target.value = "";
    }
  }

  if (selectedIds.length > 1) {
    const grouped = Boolean(
      document?.elements.find((item) => item.id === selectedIds[0])?.groupId
    );
    return (
      <aside className="editor-properties">
        <div className="editor-panel-title">
          <strong>Group selection</strong>
          <small>{selectedIds.length} elements selected</small>
        </div>
        <div className="group-selection-panel">
          <span className="group-selection-icon">
            <FolderPlus size={19} />
          </span>
          <strong>{grouped ? "These elements are grouped" : "Multiple elements"}</strong>
          <p>Shift-click canvas elements to add or remove them from this selection.</p>
          {grouped ? (
            <button className="secondary-button full-width" onClick={ungroupSelected}>
              <FolderOpen size={15} /> Ungroup elements
            </button>
          ) : (
            <button className="primary-button full-width" onClick={groupSelected}>
              <FolderPlus size={15} /> Group elements
            </button>
          )}
        </div>
      </aside>
    );
  }

  if (element?.locked) {
    return (
      <aside className="editor-properties">
        <div className="editor-panel-title">
          <strong>Locked element</strong>
          <small>{element.name}</small>
        </div>
        <div className="group-selection-panel">
          <span className="group-selection-icon">
            <Lock size={19} />
          </span>
          <strong>This layer is locked</strong>
          <p>Unlock it to move or edit its properties.</p>
          <button
            className="secondary-button full-width"
            onClick={() => setElementLock(element.id, false)}
          >
            <LockKeyholeOpen size={15} /> Unlock layer
          </button>
        </div>
      </aside>
    );
  }

  if (!element) {
    return (
      <aside className="editor-properties">
        <div className="properties-empty">
          <span>
            <Type size={20} />
          </span>
          <strong>Nothing selected</strong>
          <p>
            Click any element on the canvas to adjust its position, text, or styling. All edits
            are logged to Supabase for training data.
          </p>
        </div>
        <div className="canvas-settings">
          <span className="sidebar-label">CANVAS</span>
          <label className="property-field">
            <span>Background</span>
            <input
              type="color"
              value={document?.background || "#ffffff"}
              onChange={(event) =>
                useEditorStore.getState().updatePoster({ background: event.target.value })
              }
            />
          </label>
          <label className="property-field" style={{ marginTop: 10 }}>
            <span>Poster title</span>
            <input
              value={document?.title || ""}
              onChange={(event) =>
                useEditorStore.getState().updatePoster({ title: event.target.value })
              }
            />
          </label>
        </div>
      </aside>
    );
  }

  return (
    <aside className="editor-properties">
      <div className="editor-panel-title property-title">
        <div>
          <strong>Properties</strong>
          <small>{element.name}</small>
        </div>
        <button title="Delete element" onClick={() => removeElement(element.id)}>
          <Trash2 size={16} />
        </button>
      </div>
      <div className="properties-scroll">
        {element.type === "text" && (
          <div className="property-group">
            <label className="property-field stacked-field">
              <span>Text content</span>
              <textarea
                rows={4}
                value={element.text || ""}
                onChange={(event) => change({ text: event.target.value })}
              />
            </label>
            <label className="property-field stacked-field">
              <span>Font family</span>
              <select
                value={element.fontFamily || "Arial"}
                onChange={(event) => change({ fontFamily: event.target.value })}
              >
                <option>Arial</option>
                <option>Inter</option>
                <option>Georgia</option>
                <option>Verdana</option>
                <option>Times New Roman</option>
              </select>
            </label>
            <div className="property-inline">
              <label className="property-field">
                <span>Size</span>
                <input
                  type="number"
                  min={8}
                  max={96}
                  value={Math.round(element.fontSize || 16)}
                  onChange={(event) => change({ fontSize: Number(event.target.value) })}
                />
              </label>
              <label className="property-field">
                <span>Color</span>
                <input
                  type="color"
                  value={element.fill || "#172033"}
                  onChange={(event) => change({ fill: event.target.value })}
                />
              </label>
            </div>
            <div className="property-format-row">
              <button
                className={element.fontWeight === "bold" ? "format-active" : ""}
                title="Bold"
                onClick={() =>
                  change({ fontWeight: element.fontWeight === "bold" ? "normal" : "bold" })
                }
              >
                <Bold size={16} />
              </button>
              <button
                className={element.fontStyle === "italic" ? "format-active" : ""}
                title="Italic"
                onClick={() =>
                  change({
                    fontStyle: element.fontStyle === "italic" ? "normal" : "italic",
                  })
                }
              >
                <Italic size={16} />
              </button>
              <button
                className={element.underline ? "format-active" : ""}
                title="Underline"
                onClick={() => change({ underline: !element.underline })}
              >
                <Underline size={16} />
              </button>
              <span className="format-divider" />
              <button
                className={element.align === "left" || !element.align ? "format-active" : ""}
                title="Align left"
                onClick={() => change({ align: "left" })}
              >
                <AlignLeft size={16} />
              </button>
              <button
                className={element.align === "center" ? "format-active" : ""}
                title="Align center"
                onClick={() => change({ align: "center" })}
              >
                <AlignCenter size={16} />
              </button>
              <button
                className={element.align === "right" ? "format-active" : ""}
                title="Align right"
                onClick={() => change({ align: "right" })}
              >
                <AlignRight size={16} />
              </button>
            </div>
            <label className="property-field stacked-field">
              <span>
                Line spacing <b>{element.lineHeight || 1.2}</b>
              </span>
              <input
                type="range"
                min="0.8"
                max="2"
                step="0.1"
                value={element.lineHeight || 1.2}
                onChange={(event) => change({ lineHeight: Number(event.target.value) })}
              />
            </label>
          </div>
        )}

        {element.type === "image" && (
          <div className="property-group">
            <button
              className="secondary-button full-width"
              onClick={() => imageInput.current?.click()}
            >
              <ImagePlus size={16} /> Replace image
            </button>
            <input
              ref={imageInput}
              className="visually-hidden"
              type="file"
              accept="image/*"
              onChange={(event) => void replaceImage(event)}
            />
            <label className="property-field stacked-field">
              <span>Image fitting</span>
              <select
                value={element.imageFit || "contain"}
                onChange={(event) =>
                  change({ imageFit: event.target.value as "contain" | "cover" })
                }
              >
                <option value="contain">Fit whole image</option>
                <option value="cover">Crop to fill</option>
              </select>
            </label>
            <label className="property-field stacked-field">
              <span>
                Opacity <b>{Math.round((element.opacity ?? 1) * 100)}%</b>
              </span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={element.opacity ?? 1}
                onChange={(event) => change({ opacity: Number(event.target.value) })}
              />
            </label>
          </div>
        )}

        {(element.type === "rect" || element.type === "circle") && (
          <div className="property-group">
            <label className="property-field">
              <span>Fill</span>
              <input
                type="color"
                value={element.fill || "#3458e8"}
                onChange={(event) => change({ fill: event.target.value })}
              />
            </label>
            <label className="property-field">
              <span>Border</span>
              <input
                type="color"
                value={element.stroke || "#3458e8"}
                onChange={(event) => change({ stroke: event.target.value })}
              />
            </label>
            <label className="property-field stacked-field">
              <span>
                Border width <b>{element.strokeWidth || 0}px</b>
              </span>
              <input
                type="range"
                min="0"
                max="16"
                value={element.strokeWidth || 0}
                onChange={(event) => change({ strokeWidth: Number(event.target.value) })}
              />
            </label>
            <label className="property-field stacked-field">
              <span>
                Opacity <b>{Math.round((element.opacity ?? 1) * 100)}%</b>
              </span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={element.opacity ?? 1}
                onChange={(event) => change({ opacity: Number(event.target.value) })}
              />
            </label>
          </div>
        )}

        {(element.type === "line" || element.type === "arrow") && (
          <div className="property-group">
            <label className="property-field">
              <span>Color</span>
              <input
                type="color"
                value={element.stroke || element.fill || "#3458e8"}
                onChange={(event) =>
                  change({ stroke: event.target.value, fill: event.target.value })
                }
              />
            </label>
            <label className="property-field">
              <span>Width</span>
              <input
                type="number"
                min={1}
                max={40}
                value={element.strokeWidth || 4}
                onChange={(event) => change({ strokeWidth: Number(event.target.value) })}
              />
            </label>
          </div>
        )}

        <div className="property-group">
          <span className="sidebar-label">POSITION &amp; SIZE</span>
          <div className="property-grid">
            {numberField("X", "x")}
            {numberField("Y", "y")}
            {numberField("Width", "width", 12)}
            {numberField("Height", "height", 12)}
            {numberField("Rotation", "rotation", -360)}
          </div>
        </div>

        <div className="property-group">
          <span className="sidebar-label">LAYER ORDER</span>
          <div className="layer-order-controls">
            <button onClick={() => reorderElement(element.id, "up")}>
              <ArrowUp size={15} /> Forward
            </button>
            <button onClick={() => reorderElement(element.id, "down")}>
              <ArrowDown size={15} /> Backward
            </button>
          </div>
          <button
            className="duplicate-property"
            onClick={() => duplicateElement(element.id)}
          >
            <Plus size={15} /> Duplicate element
          </button>
        </div>
      </div>
    </aside>
  );
}
