export type ProjectStatus = "draft" | "submitted";
export type Orientation = "portrait" | "landscape";
export type PosterElementType = "text" | "image" | "rect" | "circle" | "line" | "arrow";
export type TextAlignment = "left" | "center" | "right";

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
}

export interface PosterDocument {
  id: string;
  projectId: string;
  title: string;
  width: number;
  height: number;
  orientation: Orientation;
  background: string;
  templateId: string;
  columns: number;
  elements: PosterElement[];
  updatedAt: string;
}

export interface ResearchSection {
  id: string;
  title: string;
  content: string;
  included: boolean;
}

export interface Project {
  id: string;
  title: string;
  status: ProjectStatus;
  updatedAt: string;
  templateId: string;
  sections: ResearchSection[];
  documentName?: string;
  poster?: PosterDocument;
}

export interface ProjectUser {
  name: string;
  email: string;
}

export interface PosterTemplate {
  id: string;
  name: string;
  category: string;
  description: string;
  accent: string;
  background: string;
  columns: number;
}
