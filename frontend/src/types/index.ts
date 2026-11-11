export type Orientation = "portrait" | "landscape";
export type EngineType = "adaptation" | "neural";
export type PosterElementType = "text" | "image" | "rect" | "circle" | "line" | "arrow";
export type TextAlignment = "left" | "center" | "right";
export type ContentBlockType = "section" | "text" | "figure" | "table" | "list";

export interface ContentBlockInput {
  id: string;
  type: ContentBlockType;
  heading?: string;
  text?: string;
  imageDataUrl?: string;
  aspectRatio?: number;
  tableRows?: number;
  tableCols?: number;
  caption?: string;
  priority?: number;
}

export interface PosterGenerateRequest {
  title: string;
  authors: string;
  affiliations?: string;
  orientation: Orientation;
  preferredColumns?: number | null;
  accentColor: string;
  background: string;
  engine: EngineType;
  blocks: ContentBlockInput[];
}

export interface PosterElement {
  id: string;
  type: PosterElementType;
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  layer: number;
  visible: boolean;
  locked: boolean;
  groupId?: string;
  text?: string;
  fontFamily?: string;
  fontSize?: number;
  fontWeight?: "normal" | "bold";
  fontStyle?: "normal" | "italic";
  underline?: boolean;
  align?: TextAlignment;
  lineHeight?: number;
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  opacity?: number;
  imageFit?: "contain" | "cover";
  assetId?: string;
  src?: string;
  points?: number[];
  sourceBlockId?: string;
  semanticRole?: string;
  bboxNorm?: number[];
}

export interface PosterDocument {
  id: string;
  title: string;
  width: number;
  height: number;
  orientation: Orientation;
  background: string;
  accentColor: string;
  columns: number;
  elements: PosterElement[];
  updatedAt: string;
}

export interface CandidateTemplateMeta {
  rank: number;
  poster_id: string;
  orientation: string;
  estimated_column_count: number;
  relevance_score: number;
}

export interface GenerationRun {
  id: string;
  createdAt: string;
  updatedAt: string;
  title: string;
  authors: string;
  orientation: Orientation;
  engine: EngineType;
  modelName: string;
  modelVersion: string;
  retrievedTemplateId?: string | null;
  candidateTemplates: CandidateTemplateMeta[];
  generationMetrics: Record<string, unknown>;
  poster: PosterDocument;
  wasEdited: boolean;
  exportedFormats: string[];
  feedbackRating?: number | null;
  feedbackNotes?: string | null;
}

export interface GenerationRunSummary {
  id: string;
  createdAt: string;
  title: string;
  orientation: Orientation;
  engine: EngineType;
  modelName: string;
  modelVersion: string;
  retrievedTemplateId?: string | null;
  wasEdited: boolean;
  exportedFormats: string[];
  feedbackRating?: number | null;
  generationMetrics: Record<string, unknown>;
}
