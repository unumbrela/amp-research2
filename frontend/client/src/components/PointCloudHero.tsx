import { Component, useEffect, useMemo, useRef, useState, type ErrorInfo, type ReactNode } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Bloom, EffectComposer, Noise, Vignette } from "@react-three/postprocessing";
import * as THREE from "three";
import Papa from "papaparse";
import { cn } from "@/lib/utils";

interface DataRow {
  x: number | string;
  y: number | string;
  z: number | string;
  part_type?: string;
  source?: string;
  name?: string;
  id?: string | number;
}

interface SelectedPoint {
  id: string | number;
  name: string;
  partType?: string;
  source?: string;
  coordinates: { x: number; y: number; z: number };
}

interface PointCloudHeroProps {
  csvUrl: string;
  className?: string;
  background?: string;
  showLegend?: boolean;
}

const PART_TYPE_COLORS: Record<string, string> = {
  cluster_01: "#d62828",
  cluster_02: "#f77f00",
  cluster_03: "#e09f3e",
  cluster_04: "#2a9d8f",
  cluster_05: "#1d4ed8",
  cluster_06: "#3a86ff",
  cluster_07: "#6a4c93",
  cluster_08: "#8338ec",
  cluster_09: "#ff006e",
  cluster_10: "#b5179e",
  cluster_11: "#2d6a4f",
  cluster_12: "#40916c",
  cluster_13: "#577590",
  cluster_14: "#ef476f",
  apd: "#ff5c5c",
  dramp: "#4ea8de",
  ampainter: "#f4a261",
  diffamp: "#2a9d8f",
  uniprot: "#9b5de5",
  mixed: "#ffd166",
  other: "#adb5bd",
  unknown: "#6c757d",
  cds: "#ff6b6b",
  composite: "#4ecdc4",
  regulatory: "#ffe66d",
  dna: "#95e1d3",
  protein: "#ff8c42",
  rbs: "#c44569",
  intermediate: "#9b59b6",
  reporter: "#3498db",
  promoter: "#2ecc71",
  primer: "#e74c3c",
  rna: "#f39c12",
  generator: "#1abc9c",
  device: "#e67e22",
  tag: "#9b59b6",
  binding: "#34495e",
  protein_domain: "#16a085",
};

function fallbackColorForType(type: string): string {
  let hash = 0;
  for (let i = 0; i < type.length; i++) {
    hash = (hash * 31 + type.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  return `hsl(${hue}, 88%, 45%)`;
}

const vertexShader = /* glsl */ `
  attribute float aSize;
  attribute vec3 aColor;
  attribute float aHighlight;
  varying vec3 vColor;
  varying float vDepth;
  varying float vHighlight;

  void main() {
    vColor = aColor;
    vHighlight = aHighlight;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float dist = -mvPosition.z;
    vDepth = dist;
    float attenuation = clamp(360.0 / dist, 0.0, 10.0);
    gl_PointSize = aSize * attenuation;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = /* glsl */ `
  precision highp float;
  varying vec3 vColor;
  varying float vDepth;
  varying float vHighlight;

  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = dot(uv, uv);
    float alpha = smoothstep(1.0, 0.0, d);

    float depthFade = 1.0 - smoothstep(8.0, 22.0, vDepth) * 0.28;
    alpha *= depthFade;

    float dimFactor = vHighlight > 0.5 ? 1.0 : (vHighlight < -0.5 ? 0.18 : 1.0);
    alpha *= dimFactor;

    float glow = smoothstep(0.22, 0.0, d) * 0.12;
    vec3 color = vColor * 0.90 + vColor * glow;
    gl_FragColor = vec4(color, min(alpha * 1.15, 1.0));
  }
`;

function checkWebGLSupport(): { supported: boolean; error?: string } {
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");

    if (!gl) {
      return { supported: false, error: "WebGL is not supported by your browser" };
    }

    return { supported: true };
  } catch (error) {
    return {
      supported: false,
      error: `WebGL initialization failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

class StarFieldErrorBoundary extends Component<
  { children: ReactNode; onError: (error: Error) => void },
  { hasError: boolean }
> {
  constructor(props: { children: ReactNode; onError: (error: Error) => void }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(_: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("[PointCloudHero] Render error:", error, errorInfo);
    this.props.onError(error);
  }

  render() {
    if (this.state.hasError) {
      return null;
    }

    return this.props.children;
  }
}

function useCSV(csvUrl: string) {
  const [rows, setRows] = useState<DataRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(csvUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to fetch CSV (${response.status})`);
        }
        return response.text();
      })
      .then((csvText) => {
        if (cancelled) {
          return;
        }

        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            if (cancelled) {
              return;
            }

            if (results.errors.length > 0) {
              setError(results.errors[0].message || "CSV parse error");
              setLoading(false);
              return;
            }

            const filtered = (results.data as DataRow[]).filter(
              (row) => row.x != null && row.y != null && row.z != null,
            );

            setRows(filtered);
            setLoading(false);
          },
          error: (parseError) => {
            if (cancelled) {
              return;
            }
            setError(parseError.message || "CSV parse error");
            setLoading(false);
          },
        });
      })
      .catch((fetchError: unknown) => {
        if (cancelled) {
          return;
        }
        setError(fetchError instanceof Error ? fetchError.message : "CSV load error");
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [csvUrl]);

  return { rows, error, loading };
}

function buildGeometryFromRows(rows: DataRow[]) {
  const pointCount = rows.length;
  const positions = new Float32Array(pointCount * 3);
  const colors = new Float32Array(pointCount * 3);
  const sizes = new Float32Array(pointCount);
  const highlights = new Float32Array(pointCount);
  const partTypes = new Array<string>(pointCount).fill("");

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  const parsed = rows.map((row) => ({
    x: typeof row.x === "number" ? row.x : parseFloat(String(row.x)),
    y: typeof row.y === "number" ? row.y : parseFloat(String(row.y)),
    z: typeof row.z === "number" ? row.z : parseFloat(String(row.z)),
  }));

  for (const coord of parsed) {
    if (Number.isFinite(coord.x)) {
      minX = Math.min(minX, coord.x);
      maxX = Math.max(maxX, coord.x);
    }
    if (Number.isFinite(coord.y)) {
      minY = Math.min(minY, coord.y);
      maxY = Math.max(maxY, coord.y);
    }
    if (Number.isFinite(coord.z)) {
      minZ = Math.min(minZ, coord.z);
      maxZ = Math.max(maxZ, coord.z);
    }
  }

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const rangeZ = maxZ - minZ || 1;
  const scale = 240 / Math.max(rangeX, rangeY, rangeZ);

  const fallbackPalette = [
    new THREE.Color("#9ec9ff"),
    new THREE.Color("#ffd6a5"),
    new THREE.Color("#bdb2ff"),
    new THREE.Color("#caffbf"),
    new THREE.Color("#ffadad"),
    new THREE.Color("#fdffb6"),
  ];
  const partMap = new Map<string, number>();
  let nextColor = 0;

  for (let i = 0; i < pointCount; i++) {
    const row = rows[i];
    const coord = parsed[i];

    const x = Number.isFinite(coord.x) ? (coord.x - minX - rangeX / 2) * scale : 0;
    const y = Number.isFinite(coord.y) ? (coord.y - minY - rangeY / 2) * scale : 0;
    const z = Number.isFinite(coord.z) ? (coord.z - minZ - rangeZ / 2) * scale : 0;

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    sizes[i] = 1.15 + Math.random() * 1.6;
    highlights[i] = 0;

    const type = (row.part_type || "").toLowerCase();
    partTypes[i] = type;

    let color: THREE.Color;
    if (type) {
      const known = PART_TYPE_COLORS[type];
      if (known) {
        color = new THREE.Color(known);
      } else {
        if (!partMap.has(type)) {
          partMap.set(type, nextColor++);
        }
        color = fallbackPalette[(partMap.get(type) || 0) % fallbackPalette.length];
      }
    } else {
      const t = THREE.MathUtils.clamp((z + 500) / 1000, 0, 1);
      color = new THREE.Color().setHSL(0.56 * (1 - t) + 0.02, 0.8, 0.5 + 0.2 * (1 - t));
    }

    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute("aHighlight", new THREE.BufferAttribute(highlights, 1));
  geometry.computeBoundingSphere();

  return {
    geometry,
    positions,
    partTypes,
  };
}

function PointCloudPicker({
  rows,
  positions,
  onPointClick,
}: {
  rows: DataRow[];
  positions: Float32Array;
  onPointClick: (point: SelectedPoint) => void;
}) {
  const { camera, gl, size } = useThree();
  const mouseDownRef = useRef<{ x: number; y: number } | null>(null);
  const lastHoverCheckRef = useRef<number>(0);

  useEffect(() => {
    const canvas = gl.domElement;
    canvas.style.cursor = "grab";

    const findClosestIndex = (
      mouseX: number,
      mouseY: number,
      stride: number = 1,
      threshold: number = 7,
    ): number => {
      let closestIndex = -1;
      let closestScreenDist = Infinity;
      let closestDepth = Infinity;

      for (let i = 0; i < rows.length; i += stride) {
        const x = positions[i * 3];
        const y = positions[i * 3 + 1];
        const z = positions[i * 3 + 2];

        const point = new THREE.Vector3(x, y, z);
        const projected = point.clone().project(camera);

        if (projected.z >= 1) {
          continue;
        }

        const screenX = (projected.x * 0.5 + 0.5) * size.width;
        const screenY = (-(projected.y * 0.5) + 0.5) * size.height;
        const dx = screenX - mouseX;
        const dy = screenY - mouseY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const depth = camera.position.distanceTo(point);

        if (distance < threshold) {
          if (distance < closestScreenDist || (distance === closestScreenDist && depth < closestDepth)) {
            closestIndex = i;
            closestScreenDist = distance;
            closestDepth = depth;
          }
        }
      }

      return closestIndex;
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (mouseDownRef.current) {
        return;
      }

      const now = performance.now();
      if (now - lastHoverCheckRef.current < 45) {
        return;
      }
      lastHoverCheckRef.current = now;

      const rect = canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      const closestIndex = findClosestIndex(mouseX, mouseY, 2, 8);

      canvas.style.cursor = closestIndex >= 0 ? "pointer" : "grab";
    };

    const handlePointerDown = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseDownRef.current = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
      canvas.style.cursor = "grabbing";
    };

    const handlePointerUp = (event: PointerEvent) => {
      if (!mouseDownRef.current) {
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const mouseUp = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };

      const dx = mouseUp.x - mouseDownRef.current.x;
      const dy = mouseUp.y - mouseDownRef.current.y;
      const movedDistance = Math.sqrt(dx * dx + dy * dy);
      mouseDownRef.current = null;
      canvas.style.cursor = "grab";

      // Ignore drag gestures; only treat near-stationary up/down as click.
      if (movedDistance > 5) {
        return;
      }

      const closestIndex = findClosestIndex(mouseUp.x, mouseUp.y);
      if (closestIndex < 0) {
        return;
      }

      const row = rows[closestIndex];
      if (row.id == null || row.id === "") {
        return;
      }
      onPointClick({
        id: row.id,
        name: String(row.name || row.id || `Point ${closestIndex}`),
        partType: row.part_type,
        source: row.source,
        coordinates: {
          x: positions[closestIndex * 3],
          y: positions[closestIndex * 3 + 1],
          z: positions[closestIndex * 3 + 2],
        },
      });
    };

    const handlePointerLeave = () => {
      if (!mouseDownRef.current) {
        canvas.style.cursor = "grab";
      }
    };

    canvas.addEventListener("pointermove", handlePointerMove);
    canvas.addEventListener("pointerdown", handlePointerDown);
    canvas.addEventListener("pointerup", handlePointerUp);
    canvas.addEventListener("pointerleave", handlePointerLeave);

    return () => {
      canvas.removeEventListener("pointermove", handlePointerMove);
      canvas.removeEventListener("pointerdown", handlePointerDown);
      canvas.removeEventListener("pointerup", handlePointerUp);
      canvas.removeEventListener("pointerleave", handlePointerLeave);
      canvas.style.cursor = "default";
    };
  }, [camera, gl, onPointClick, positions, rows, size.height, size.width]);

  return null;
}

function StarPoints({
  rows,
  highlightPartType,
  onPointClick,
}: {
  rows: DataRow[];
  highlightPartType: string | null;
  onPointClick?: (point: SelectedPoint) => void;
}) {
  const geometryData = useMemo(() => buildGeometryFromRows(rows), [rows]);

  const uniforms = useMemo(
    () => ({
      uConvergenceProgress: { value: 1.0 },
    }),
    [],
  );

  useEffect(() => {
    const attr = geometryData.geometry.getAttribute("aHighlight") as THREE.BufferAttribute;
    const highlights = attr.array as Float32Array;

    for (let i = 0; i < highlights.length; i++) {
      if (highlightPartType == null) {
        highlights[i] = 0;
      } else if (geometryData.partTypes[i] === highlightPartType) {
        highlights[i] = 1;
      } else {
        highlights[i] = -1;
      }
    }

    attr.needsUpdate = true;
  }, [geometryData, highlightPartType]);

  return (
    <>
      <points frustumCulled>
        <primitive object={geometryData.geometry} attach="geometry" />
        <shaderMaterial
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          blending={THREE.NormalBlending}
          transparent
          depthTest={false}
          depthWrite={false}
          uniforms={uniforms}
        />
      </points>
      {onPointClick && (
        <PointCloudPicker rows={rows} positions={geometryData.positions} onPointClick={onPointClick} />
      )}
    </>
  );
}

export default function PointCloudHero({
  csvUrl,
  className,
  background = "#02040a",
  showLegend = true,
}: PointCloudHeroProps) {
  const { rows, error, loading } = useCSV(csvUrl);
  const [highlightPartType, setHighlightPartType] = useState<string | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<SelectedPoint | null>(null);
  const [infoPanelHeight, setInfoPanelHeight] = useState(0);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [webglSupport, setWebglSupport] = useState<{ supported: boolean; error?: string } | null>(null);

  useEffect(() => {
    setWebglSupport(checkWebGLSupport());
  }, []);

  useEffect(() => {
    const panel = document.getElementById("hero-what-you-are-seeing");
    if (!panel) return;

    const updateHeight = () => {
      setInfoPanelHeight(Math.round(panel.getBoundingClientRect().height));
    };

    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(panel);
    window.addEventListener("resize", updateHeight);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateHeight);
    };
  }, []);

  const canRender = webglSupport?.supported !== false && !renderError;
  const legendItems = useMemo(() => {
    if (!rows || rows.length === 0) {
      return Object.entries(PART_TYPE_COLORS).slice(0, 12) as [string, string][];
    }

    const counts = new Map<string, number>();
    for (const row of rows) {
      const type = (row.part_type || "unknown").toLowerCase();
      counts.set(type, (counts.get(type) || 0) + 1);
    }

    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([type]) => [type, PART_TYPE_COLORS[type] || fallbackColorForType(type)] as [string, string]);
  }, [rows]);

  return (
    <div className={cn("relative w-full h-full overflow-hidden", className)}>
      {canRender && rows && !loading && !error && (
        <div className="absolute inset-0 z-10">
          <StarFieldErrorBoundary
            onError={(err) => setRenderError(err.message || "Failed to render point cloud")}
          >
            <Canvas
              camera={{ fov: 70, near: 0.1, far: 5000, position: [0, 26, 320] }}
              dpr={[1, 2]}
              gl={{ antialias: false, powerPreference: "high-performance", alpha: true }}
              onCreated={({ gl }) => {
                gl.setClearColor(new THREE.Color(background), 0);
              }}
            >
              <StarPoints
                rows={rows}
                highlightPartType={highlightPartType}
                onPointClick={setSelectedPoint}
              />
              <EffectComposer multisampling={0}>
                <Bloom intensity={0.18} luminanceThreshold={0.2} luminanceSmoothing={0.25} mipmapBlur />
                <Noise opacity={0.01} />
                <Vignette eskil={false} offset={0.23} darkness={0.45} />
              </EffectComposer>
              <OrbitControls
                enableDamping
                dampingFactor={0.08}
                enablePan={false}
                minDistance={120}
                maxDistance={900}
                rotateSpeed={0.9}
                zoomSpeed={0.9}
              />
            </Canvas>
          </StarFieldErrorBoundary>
        </div>
      )}

      {canRender && rows && !loading && !error && showLegend && (
        <div className="absolute right-4 top-4 z-30 rounded-lg border border-white/20 bg-black/55 p-3 text-white backdrop-blur-sm">
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className="text-xs font-semibold tracking-wide">Similarity Clusters</span>
            {highlightPartType && (
              <button
                type="button"
                onClick={() => setHighlightPartType(null)}
                className="rounded border border-white/30 px-2 py-0.5 text-[10px] hover:bg-white/10"
              >
                Clear
              </button>
            )}
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            {legendItems.map(([type, color]) => {
              const active = highlightPartType === type;
              const dimmed = highlightPartType != null && !active;
              return (
                <button
                  key={type}
                  type="button"
                  onClick={() => setHighlightPartType(active ? null : type)}
                  className={cn(
                    "flex items-center gap-1.5 rounded px-1 py-0.5 text-left text-[10px] capitalize transition-opacity",
                    dimmed ? "opacity-40" : "opacity-100",
                  )}
                >
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                  <span>{type.replace("_", " ")}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {selectedPoint && (
        <div
          className="absolute right-4 z-40 w-[min(360px,calc(100%-2rem))] rounded-xl border border-black/15 bg-[#f5efdf]/95 p-4 text-[#3c372c] shadow-xl backdrop-blur-sm"
          style={{ bottom: `calc(1rem + ${infoPanelHeight}px + 0.75rem)` }}
        >
          <div className="mb-2 flex items-start justify-between gap-3">
            <div>
              <p className="text-[11px] uppercase tracking-wider text-[#6a634f]">AMP Point</p>
              <h4 className="text-sm font-semibold leading-tight break-all">{selectedPoint.name}</h4>
            </div>
            <button
              type="button"
              onClick={() => setSelectedPoint(null)}
              className="rounded border border-black/20 px-2 py-0.5 text-xs hover:bg-black/5"
              aria-label="Close point panel"
            >
              Close
            </button>
          </div>
          <div className="space-y-1 text-xs leading-relaxed">
            <p><span className="font-medium">ID:</span> {selectedPoint.id}</p>
            <p><span className="font-medium">Cluster:</span> {(selectedPoint.partType || "unknown").replace("_", " ")}</p>
            <p><span className="font-medium">Source:</span> {(selectedPoint.source || "unknown").replace("_", " ")}</p>
            <p>
              <span className="font-medium">Coordinates:</span>{" "}
              ({selectedPoint.coordinates.x.toFixed(1)}, {selectedPoint.coordinates.y.toFixed(1)}, {selectedPoint.coordinates.z.toFixed(1)})
            </p>
            <p className="pt-1 text-[#5b5443]">
              This point represents one AMP sequence. Points in the same cluster usually share similar sequence composition and physicochemical tendencies.
            </p>
          </div>
        </div>
      )}

      {(loading || error || renderError || webglSupport?.supported === false) && (
        <div
          className="absolute inset-0 z-20 flex items-center justify-center p-4 text-center text-sm text-black/70"
          style={{ backgroundColor: background }}
        >
          {loading && <span>Loading 3D point cloud...</span>}
          {!loading && (error || renderError || webglSupport?.error) && (
            <span>{error || renderError || webglSupport?.error}</span>
          )}
        </div>
      )}
    </div>
  );
}
