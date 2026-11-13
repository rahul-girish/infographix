import { create } from "zustand";
import type { GenerationRun, PosterDocument, PosterElement } from "../types";

interface EditorState {
  run: Omit<GenerationRun, "poster"> | null;
  document: PosterDocument | null;
  selectedId: string | null;
  selectedIds: string[];
  zoom: number;
  showGrid: boolean;
  snapToGrid: boolean;
  undoStack: PosterDocument[];
  redoStack: PosterDocument[];
  stageExporter: (() => string | null) | null;
  setRun: (run: GenerationRun) => void;
  setRunMeta: (meta: Omit<GenerationRun, "poster">) => void;
  setDocument: (document: PosterDocument) => void;
  setStageExporter: (exporter: (() => string | null) | null) => void;
  select: (id: string | null, additive?: boolean) => void;
  updateElement: (id: string, update: Partial<PosterElement>) => void;
  addElement: (element: PosterElement) => void;
  removeElement: (id: string) => void;
  duplicateElement: (id: string) => void;
  reorderElement: (id: string, direction: "up" | "down") => void;
  setElementLock: (id: string, locked: boolean) => void;
  setElementVisibility: (id: string, visible: boolean) => void;
  groupSelected: () => void;
  ungroupSelected: () => void;
  moveGroup: (id: string, deltaX: number, deltaY: number) => void;
  updatePoster: (update: Partial<PosterDocument>) => void;
  setZoom: (zoom: number) => void;
  toggleGrid: () => void;
  toggleSnap: () => void;
  undo: () => void;
  redo: () => void;
}

const snapshot = (document: PosterDocument) => structuredClone(document);

export const useEditorStore = create<EditorState>((set, get) => ({
  run: null,
  document: null,
  selectedId: null,
  selectedIds: [],
  zoom: 0.78,
  showGrid: false,
  snapToGrid: true,
  undoStack: [],
  redoStack: [],
  stageExporter: null,
  setRun: (run) => {
    const { poster, ...meta } = run;
    set({
      run: meta,
      document: snapshot(poster),
      selectedId: null,
      selectedIds: [],
      undoStack: [],
      redoStack: [],
    });
  },
  setRunMeta: (meta) => set({ run: meta }),
  setDocument: (document) =>
    set({
      document: snapshot(document),
      selectedId: null,
      selectedIds: [],
      undoStack: [],
      redoStack: [],
    }),
  setStageExporter: (stageExporter) => set({ stageExporter }),
  select: (id, additive = false) => {
    const document = get().document;
    if (!id) {
      set({ selectedId: null, selectedIds: [] });
      return;
    }
    const element = document?.elements.find((item) => item.id === id);
    const groupIds = element?.groupId
      ? document?.elements
          .filter((item) => item.groupId === element.groupId)
          .map((item) => item.id) || [id]
      : [id];
    if (!additive) {
      set({ selectedId: id, selectedIds: groupIds });
      return;
    }
    const current = get().selectedIds;
    const allSelected = groupIds.every((groupId) => current.includes(groupId));
    const selectedIds = allSelected
      ? current.filter((selected) => !groupIds.includes(selected))
      : [...new Set([...current, ...groupIds])];
    set({
      selectedId: selectedIds.includes(id)
        ? id
        : selectedIds[selectedIds.length - 1] || null,
      selectedIds,
    });
  },
  updateElement: (id, update) => {
    const document = get().document;
    if (!document) return;
    set((state) => ({
      document: {
        ...document,
        elements: document.elements.map((element) =>
          element.id === id && !element.locked ? { ...element, ...update } : element
        ),
      },
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  addElement: (element) => {
    const document = get().document;
    if (!document) return;
    set((state) => ({
      document: {
        ...document,
        elements: [...document.elements, { ...element, layer: document.elements.length }],
      },
      selectedId: element.id,
      selectedIds: [element.id],
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  removeElement: (id) => {
    const document = get().document;
    if (!document) return;
    set((state) => ({
      document: {
        ...document,
        elements: document.elements.filter((element) => element.id !== id),
      },
      selectedId: state.selectedId === id ? null : state.selectedId,
      selectedIds: state.selectedIds.filter((selected) => selected !== id),
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  duplicateElement: (id) => {
    const document = get().document;
    const original = document?.elements.find((element) => element.id === id);
    if (!document || !original) return;
    const copy: PosterElement = {
      ...original,
      id: crypto.randomUUID(),
      groupId: undefined,
      name: `${original.name} copy`,
      x: original.x + 18,
      y: original.y + 18,
      layer: document.elements.length,
    };
    set((state) => ({
      document: { ...document, elements: [...document.elements, copy] },
      selectedId: copy.id,
      selectedIds: [copy.id],
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  reorderElement: (id, direction) => {
    const document = get().document;
    if (!document) return;
    const sorted = [...document.elements].sort((a, b) => a.layer - b.layer);
    const index = sorted.findIndex((element) => element.id === id);
    const target = direction === "up" ? index + 1 : index - 1;
    if (index < 0 || target < 0 || target >= sorted.length) return;
    [sorted[index], sorted[target]] = [sorted[target], sorted[index]];
    const elements = sorted.map((element, layer) => ({ ...element, layer }));
    set((state) => ({
      document: { ...document, elements },
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  setElementLock: (id, locked) => {
    const document = get().document;
    if (!document) return;
    set((state) => ({
      document: {
        ...document,
        elements: document.elements.map((element) =>
          element.id === id ? { ...element, locked } : element
        ),
      },
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  setElementVisibility: (id, visible) => {
    const document = get().document;
    if (!document) return;
    set((state) => ({
      document: {
        ...document,
        elements: document.elements.map((element) =>
          element.id === id ? { ...element, visible } : element
        ),
      },
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  groupSelected: () => {
    const { document, selectedIds } = get();
    if (!document || selectedIds.length < 2) return;
    const groupId = crypto.randomUUID();
    set((state) => ({
      document: {
        ...document,
        elements: document.elements.map((element) =>
          selectedIds.includes(element.id) ? { ...element, groupId } : element
        ),
      },
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  ungroupSelected: () => {
    const { document, selectedIds } = get();
    if (!document) return;
    const groups = new Set(
      document.elements
        .filter((element) => selectedIds.includes(element.id) && element.groupId)
        .map((element) => element.groupId)
    );
    if (!groups.size) return;
    set((state) => ({
      document: {
        ...document,
        elements: document.elements.map((element) =>
          groups.has(element.groupId) ? { ...element, groupId: undefined } : element
        ),
      },
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  moveGroup: (id, deltaX, deltaY) => {
    const document = get().document;
    const target = document?.elements.find((element) => element.id === id);
    if (!document || !target) return;
    set((state) => ({
      document: {
        ...document,
        elements: document.elements.map((element) =>
          !element.locked &&
          (target.groupId ? element.groupId === target.groupId : element.id === id)
            ? { ...element, x: element.x + deltaX, y: element.y + deltaY }
            : element
        ),
      },
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  updatePoster: (update) => {
    const document = get().document;
    if (!document) return;
    set((state) => ({
      document: { ...document, ...update },
      undoStack: [...state.undoStack.slice(-49), snapshot(document)],
      redoStack: [],
    }));
  },
  setZoom: (zoom) => set({ zoom: Math.min(1.2, Math.max(0.05, zoom)) }),
  toggleGrid: () => set((state) => ({ showGrid: !state.showGrid })),
  toggleSnap: () => set((state) => ({ snapToGrid: !state.snapToGrid })),
  undo: () => {
    const { document, undoStack, redoStack } = get();
    if (!document || !undoStack.length) return;
    set({
      document: undoStack[undoStack.length - 1],
      undoStack: undoStack.slice(0, -1),
      redoStack: [...redoStack, snapshot(document)],
      selectedId: null,
      selectedIds: [],
    });
  },
  redo: () => {
    const { document, undoStack, redoStack } = get();
    if (!document || !redoStack.length) return;
    set({
      document: redoStack[redoStack.length - 1],
      undoStack: [...undoStack, snapshot(document)],
      redoStack: redoStack.slice(0, -1),
      selectedId: null,
      selectedIds: [],
    });
  },
}));
