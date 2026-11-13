import { useEffect, useRef, useState } from "react";
import {
  Arrow,
  Circle,
  Image as KonvaImage,
  Layer,
  Line,
  Rect,
  Stage,
  Text,
  Transformer,
} from "react-konva";
import type Konva from "konva";
import type { PosterElement } from "../../types";
import { useEditorStore } from "../../stores/editorStore";

function CanvasImage({
  element,
  onSelect,
  onChange,
  onDragMove,
  onDragStart,
  onDragEnd,
  readOnly,
}: {
  element: PosterElement;
  onSelect: (additive: boolean) => void;
  onChange: (update: Partial<PosterElement>) => void;
  onDragMove: (event: Konva.KonvaEventObject<DragEvent>, element: PosterElement) => void;
  onDragStart: (event: Konva.KonvaEventObject<DragEvent>) => void;
  onDragEnd: (event: Konva.KonvaEventObject<DragEvent>, element: PosterElement) => void;
  readOnly: boolean;
}) {
  const [image, setImage] = useState<HTMLImageElement>();
  const validSrc =
    element.src && element.src !== "[inline-image-omitted]" ? element.src : undefined;
  useEffect(() => {
    if (!validSrc) return;
    const loadedImage = new window.Image();
    loadedImage.onload = () => setImage(loadedImage);
    loadedImage.onerror = () => setImage(undefined);
    loadedImage.src = validSrc;
  }, [validSrc]);
  const activeImage = validSrc ? image : undefined;

  const imageRatio = activeImage ? activeImage.naturalWidth / activeImage.naturalHeight : 1;
  const frameRatio = element.width / element.height;
  const crop =
    activeImage && element.imageFit === "cover"
      ? imageRatio > frameRatio
        ? {
            x: (activeImage.naturalWidth - activeImage.naturalHeight * frameRatio) / 2,
            y: 0,
            width: activeImage.naturalHeight * frameRatio,
            height: activeImage.naturalHeight,
          }
        : {
            x: 0,
            y: (activeImage.naturalHeight - activeImage.naturalWidth / frameRatio) / 2,
            width: activeImage.naturalWidth,
            height: activeImage.naturalWidth / frameRatio,
          }
      : undefined;

  return (
    <KonvaImage
      id={element.id}
      image={activeImage}
      crop={crop}
      x={element.x}
      y={element.y}
      width={element.width}
      height={element.height}
      rotation={element.rotation}
      opacity={element.opacity ?? 1}
      draggable={!element.locked && !readOnly}
      onClick={(event) => onSelect(event.evt.shiftKey)}
      onTap={(event) => onSelect(event.evt.shiftKey)}
      onDragStart={onDragStart}
      onDragMove={(event) => onDragMove(event, element)}
      onDragEnd={(event) => onDragEnd(event, element)}
      onTransformEnd={(event) => {
        const node = event.target;
        onChange({
          x: node.x(),
          y: node.y(),
          width: Math.max(20, node.width() * node.scaleX()),
          height: Math.max(20, node.height() * node.scaleY()),
          rotation: node.rotation(),
        });
        node.scaleX(1);
        node.scaleY(1);
      }}
    />
  );
}

export default function PosterCanvas({ readOnly = false }: { readOnly?: boolean }) {
  const document = useEditorStore((state) => state.document);
  const selectedId = useEditorStore((state) => state.selectedId);
  const selectedIds = useEditorStore((state) => state.selectedIds);
  const zoom = useEditorStore((state) => state.zoom);
  const showGrid = useEditorStore((state) => state.showGrid);
  const snapToGrid = useEditorStore((state) => state.snapToGrid);
  const select = useEditorStore((state) => state.select);
  const updateElement = useEditorStore((state) => state.updateElement);
  const setStageExporter = useEditorStore((state) => state.setStageExporter);

  const stageRef = useRef<Konva.Stage>(null);
  const overlayLayerRef = useRef<Konva.Layer>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const lastDragPosition = useRef<{ x: number; y: number } | null>(null);
  const [guides, setGuides] = useState<{ x?: number; y?: number }>({});

  useEffect(() => {
    setStageExporter(() => {
      const stage = stageRef.current;
      if (!stage) return null;
      const overlay = overlayLayerRef.current;
      overlay?.hide();
      stage.batchDraw();
      const pixelRatio = Math.max(2, 2 / Math.max(0.1, zoom));
      const dataUrl = stage.toDataURL({ mimeType: "image/png", pixelRatio });
      overlay?.show();
      stage.batchDraw();
      return dataUrl;
    });
    return () => setStageExporter(null);
  }, [setStageExporter, zoom]);

  useEffect(() => {
    const transformer = transformerRef.current;
    const stage = stageRef.current;
    if (!transformer || !stage) return;
    const selectedNodes = !readOnly
      ? selectedIds
          .map((id) => ({ id, element: document?.elements.find((element) => element.id === id) }))
          .filter((entry) => entry.element && !entry.element.locked && entry.element.visible)
          .map((entry) => stage.findOne(`#${entry.id}`))
          .filter((node): node is Konva.Node => Boolean(node))
      : [];
    transformer.nodes(selectedNodes);
    transformer.getLayer()?.batchDraw();
  }, [selectedId, selectedIds, document, readOnly]);

  if (!document) return null;

  const width = document.width * zoom;
  const height = document.height * zoom;
  const visibleElements = [...document.elements]
    .filter((element) => element.visible)
    .sort((a, b) => a.layer - b.layer);

  const snapPosition = (value: number) => (snapToGrid ? Math.round(value / 10) * 10 : value);

  const handleDragMove = (event: Konva.KonvaEventObject<DragEvent>, element: PosterElement) => {
    if (snapToGrid) {
      event.target.position({
        x: snapPosition(event.target.x()),
        y: snapPosition(event.target.y()),
      });
    }
    const previous = lastDragPosition.current;
    if (previous && element.groupId) {
      const deltaX = event.target.x() - previous.x;
      const deltaY = event.target.y() - previous.y;
      for (const peer of document.elements) {
        if (peer.groupId !== element.groupId || peer.id === element.id || peer.locked) continue;
        const node = stageRef.current?.findOne(`#${peer.id}`);
        if (node) node.position({ x: node.x() + deltaX, y: node.y() + deltaY });
      }
    }
    lastDragPosition.current = { x: event.target.x(), y: event.target.y() };
    const centerX = event.target.x() + element.width / 2;
    const centerY = event.target.y() + element.height / 2;
    const peers = visibleElements.filter((peer) => peer.id !== element.id);
    const xCandidates = [
      document.width / 2,
      ...peers.flatMap((peer) => [peer.x + peer.width / 2, peer.x, peer.x + peer.width]),
    ];
    const yCandidates = [
      document.height / 2,
      ...peers.flatMap((peer) => [peer.y + peer.height / 2, peer.y, peer.y + peer.height]),
    ];
    const x = xCandidates.find((candidate) => Math.abs(candidate - centerX) < 5);
    const y = yCandidates.find((candidate) => Math.abs(candidate - centerY) < 5);
    setGuides({ ...(x === undefined ? {} : { x }), ...(y === undefined ? {} : { y }) });
  };

  const handleDragStart = (event: Konva.KonvaEventObject<DragEvent>) => {
    lastDragPosition.current = { x: event.target.x(), y: event.target.y() };
  };

  const handleDragEnd = (event: Konva.KonvaEventObject<DragEvent>, element: PosterElement) => {
    if (element.groupId) {
      useEditorStore
        .getState()
        .moveGroup(element.id, event.target.x() - element.x, event.target.y() - element.y);
    } else {
      updateElement(element.id, { x: event.target.x(), y: event.target.y() });
    }
    lastDragPosition.current = null;
    setGuides({});
  };

  return (
    <div className="poster-stage-shell">
      <Stage
        ref={stageRef}
        width={width}
        height={height}
        onMouseDown={(event) => {
          if (
            event.target === event.target.getStage() ||
            event.target.name() === "poster-background"
          ) {
            select(null);
          }
        }}
        onTouchStart={(event) => {
          if (
            event.target === event.target.getStage() ||
            event.target.name() === "poster-background"
          ) {
            select(null);
          }
        }}
      >
        <Layer scaleX={zoom} scaleY={zoom}>
          <Rect
            name="poster-background"
            x={0}
            y={0}
            width={document.width}
            height={document.height}
            fill={document.background}
            listening={false}
          />
          {visibleElements.map((element) => {
            const selectable = !readOnly && !element.locked;
            const onSelect = (additive = false) => select(element.id, additive);
            const common = {
              id: element.id,
              x: element.x,
              y: element.y,
              width: element.width,
              height: element.height,
              rotation: element.rotation,
              opacity: element.opacity ?? 1,
              draggable: selectable,
              onClick: (event: Konva.KonvaEventObject<MouseEvent>) => onSelect(event.evt.shiftKey),
              onTap: (event: Konva.KonvaEventObject<TouchEvent>) => onSelect(event.evt.shiftKey),
              onDragStart: handleDragStart,
              onDragMove: (event: Konva.KonvaEventObject<DragEvent>) =>
                handleDragMove(event, element),
              onDragEnd: (event: Konva.KonvaEventObject<DragEvent>) =>
                handleDragEnd(event, element),
              onTransformEnd: (event: Konva.KonvaEventObject<Event>) => {
                const node = event.target;
                updateElement(element.id, {
                  x: node.x(),
                  y: node.y(),
                  width: Math.max(12, node.width() * node.scaleX()),
                  height: Math.max(12, node.height() * node.scaleY()),
                  rotation: node.rotation(),
                  ...(element.type === "text"
                    ? {
                        fontSize: Math.max(
                          8,
                          (element.fontSize || 16) * Math.max(node.scaleX(), node.scaleY())
                        ),
                      }
                    : {}),
                  ...(["line", "arrow"].includes(element.type)
                    ? {
                        points: (element.points || [0, 0, element.width, element.height]).map(
                          (point, index) =>
                            point * (index % 2 === 0 ? node.scaleX() : node.scaleY())
                        ),
                      }
                    : {}),
                });
                node.scaleX(1);
                node.scaleY(1);
              },
            };

            if (element.type === "text") {
              return (
                <Text
                  key={element.id}
                  {...common}
                  text={element.text || ""}
                  fontFamily={element.fontFamily || "Arial"}
                  fontSize={element.fontSize || 16}
                  fontStyle={`${element.fontWeight === "bold" ? "bold" : "normal"}${
                    element.fontStyle === "italic" ? " italic" : ""
                  }`}
                  textDecoration={element.underline ? "underline" : ""}
                  align={element.align || "left"}
                  lineHeight={element.lineHeight || 1.2}
                  fill={element.fill || "#172033"}
                  wrap="word"
                />
              );
            }

            if (element.type === "image") {
              return (
                <CanvasImage
                  key={element.id}
                  element={element}
                  onSelect={onSelect}
                  onChange={(update) => updateElement(element.id, update)}
                  onDragMove={handleDragMove}
                  onDragStart={handleDragStart}
                  onDragEnd={handleDragEnd}
                  readOnly={readOnly}
                />
              );
            }

            if (element.type === "circle") {
              return (
                <Circle
                  key={element.id}
                  {...common}
                  radius={Math.min(element.width, element.height) / 2}
                  fill={element.fill || "#3458e8"}
                  stroke={element.stroke}
                  strokeWidth={element.strokeWidth || 0}
                />
              );
            }

            if (element.type === "line" || element.type === "arrow") {
              const points = element.points || [0, 0, element.width, element.height];
              const Shape = element.type === "arrow" ? Arrow : Line;
              return (
                <Shape
                  key={element.id}
                  {...common}
                  points={points}
                  stroke={element.stroke || element.fill || "#3458e8"}
                  strokeWidth={element.strokeWidth || 4}
                  lineCap="round"
                  lineJoin="round"
                  hitStrokeWidth={16}
                  {...(element.type === "arrow" ? { pointerLength: 14, pointerWidth: 14 } : {})}
                />
              );
            }

            return (
              <Rect
                key={element.id}
                {...common}
                fill={element.fill || "#3458e8"}
                stroke={element.stroke}
                strokeWidth={element.strokeWidth || 0}
                cornerRadius={element.name.includes("background") ? 8 : 2}
              />
            );
          })}
        </Layer>

        <Layer ref={overlayLayerRef} scaleX={zoom} scaleY={zoom}>
          {showGrid &&
            Array.from({ length: Math.floor(document.width / 25) }, (_, index) => (
              <Line
                key={`v-${index}`}
                points={[(index + 1) * 25, 0, (index + 1) * 25, document.height]}
                stroke="#dbe2ef"
                strokeWidth={0.7}
                listening={false}
              />
            ))}
          {showGrid &&
            Array.from({ length: Math.floor(document.height / 25) }, (_, index) => (
              <Line
                key={`h-${index}`}
                points={[0, (index + 1) * 25, document.width, (index + 1) * 25]}
                stroke="#dbe2ef"
                strokeWidth={0.7}
                listening={false}
              />
            ))}
          {guides.x !== undefined && (
            <Line
              points={[guides.x, 0, guides.x, document.height]}
              stroke="#f05280"
              strokeWidth={1}
              dash={[7, 5]}
              listening={false}
            />
          )}
          {guides.y !== undefined && (
            <Line
              points={[0, guides.y, document.width, guides.y]}
              stroke="#f05280"
              strokeWidth={1}
              dash={[7, 5]}
              listening={false}
            />
          )}
          {!readOnly && (
            <Transformer
              ref={transformerRef}
              rotateEnabled={Boolean(
                selectedIds.length &&
                  selectedIds.some(
                    (id) => !document.elements.find((element) => element.id === id)?.locked
                  )
              )}
              keepRatio={Boolean(
                selectedIds.length === 1 &&
                  ["image", "circle"].includes(
                    document.elements.find((element) => element.id === selectedId)?.type || ""
                  )
              )}
              borderStroke="#4563ec"
              anchorStroke="#4563ec"
              anchorFill="#ffffff"
              anchorSize={8}
            />
          )}
        </Layer>
      </Stage>
    </div>
  );
}
