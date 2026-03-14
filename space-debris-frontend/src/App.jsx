import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Copy,
  Database,
  Download,
  FileJson,
  Loader2,
  Menu,
  Moon,
  Radar,
  Rocket,
  Satellite,
  ShieldAlert,
  Sparkles,
  Sun,
  Trash2,
  Upload,
  X,
  LayoutDashboard,
  Settings2,
  Orbit,
  Bell,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from "recharts";

import "./App.css";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

const EARTH_RADIUS = 6378137.0;
const DEFAULT_ALTITUDE = 400000.0;
const SATELLITE_SPEED = 7500.0;
const SPREAD = 5000.0;
const REL_VEL_SPREAD = 20.0;

const INPUT_MODE_OPTIONS = [
  { value: "real-tle", label: "REAL TLE / NORAD", icon: Satellite },
  { value: "manual", label: "MANUAL DEMO", icon: Sparkles },
];

const RUN_MODE_OPTIONS = [
  { value: "auto", label: "AUTO" },
  { value: "engine1", label: "FAST / ENGINE-1" },
  { value: "engine2", label: "ACCURATE / ENGINE-2" },
  { value: "pipeline", label: "PIPELINE" },
];

const DEBRIS_SOURCE_OPTIONS = [
  { value: "backend", label: "Use backend list", icon: Database },
  { value: "upload", label: "Upload / paste IDs", icon: Upload },
];

const sectionMotion = {
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.12 },
  transition: { duration: 0.45, ease: "easeOut" },
};

function cn(...classes) {
  return classes.filter(Boolean).join(" ");
}

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function truncateId(value, len = 12) {
  const str = String(value ?? "—");
  return str.length > len ? `${str.slice(0, len)}…` : str;
}

function sampleDebrisOrbit(
  baseRadius,
  satSpeed,
  spread,
  relVelSpread,
  allowInclination,
) {
  const theta = rand(0, Math.PI * 2);
  const radial = rand(-spread, spread);
  const zPerturb = allowInclination ? rand(-spread / 10, spread / 10) : 0;
  const r = baseRadius + radial;

  const x = r * Math.cos(theta);
  const y = r * Math.sin(theta);
  const z = zPerturb;

  const vMag = satSpeed + rand(-relVelSpread, relVelSpread);
  const vx = -vMag * Math.sin(theta) + rand(-relVelSpread, relVelSpread);
  const vy = vMag * Math.cos(theta) + rand(-relVelSpread, relVelSpread);
  const vz = allowInclination ? rand(-relVelSpread / 10, relVelSpread / 10) : 0;

  return {
    pos: { x, y, z },
    vel: { x: vx, y: vy, z: vz },
  };
}

function buildManualPayload({
  altitude,
  debrisCount,
  allowInclination,
  mode,
  lookahead,
  dt,
  debrisNames,
}) {
  const r = EARTH_RADIUS + Number(altitude);
  const sat_pos = { x: r, y: 0, z: 0 };
  const sat_vel = { x: 0, y: SATELLITE_SPEED, z: 0 };

  const debris_states = Array.from({ length: Number(debrisCount) }, (_, i) => {
    const sampled = sampleDebrisOrbit(
      r,
      SATELLITE_SPEED,
      SPREAD,
      REL_VEL_SPREAD,
      allowInclination,
    );

    return {
      name: debrisNames[i]?.trim() || `Debris-${i + 1}`,
      pos: sampled.pos,
      vel: sampled.vel,
    };
  });

  return {
    mode,
    lookahead: Number(lookahead),
    dt: Number(dt),
    sat_pos,
    sat_vel,
    debris_states,
    max_candidates: 10,
  };
}

function buildRealTLEPayload({
  mode,
  lookahead,
  dt,
  satelliteNoradId,
  debrisSource,
  debrisFile,
  debrisIdsText,
}) {
  return {
    mode,
    lookahead: Number(lookahead),
    dt: Number(dt),
    satellite_norad_id: Number(satelliteNoradId),
    max_candidates: 10,
    debris_source: debrisSource || "backend",
    debris_file: debrisFile || "debris_ids.txt",
    debris_ids_text: debrisSource === "upload" ? debrisIdsText || "" : null,
  };
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value)))
    return "—";
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function statusFromResponse(data) {
  const summary = data?.summary;
  const results = data?.results || [];
  const risky = results.filter((r) => r?.is_high_risk);

  const escalate = Boolean(summary?.escalate || risky.length > 0);

  if (escalate) {
    return {
      label: "Potentially Dangerous",
      tone: "danger",
      icon: ShieldAlert,
      description:
        "Conjunction risk detected. Review miss distance, shortlist, and confirmation outputs carefully.",
    };
  }

  return {
    label: "Looks Safe",
    tone: "safe",
    icon: CheckCircle2,
    description: "No major high-risk conjunction was flagged in this run.",
  };
}

function downloadJson(data, fileName = "space-debris-result.json") {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(url);
}

function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function SectionTitle({ icon: Icon, title, subtitle }) {
  return (
    <div className="space-section-title">
      <div className="space-section-title-icon">
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <h3>{title}</h3>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
    </div>
  );
}

function SegmentedControl({
  label,
  value,
  onChange,
  options,
  gridClass = "sm:grid-cols-2",
}) {
  return (
    <div className="grid gap-2">
      <Label>{label}</Label>
      <div className={cn("grid grid-cols-1 gap-2", gridClass)}>
        {options.map((option) => {
          const Icon = option.icon;
          const active = value === option.value;

          return (
            <button
              key={option.value}
              type="button"
              className={cn("space-segment-btn", active && "active")}
              onClick={() => onChange(option.value)}
            >
              {Icon ? <Icon className="h-4 w-4" /> : null}
              <span>{option.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function MetricCard({ label, value, hint, icon: Icon }) {
  return (
    <motion.div
      whileHover={{ y: -4, scale: 1.01 }}
      transition={{ type: "spring", stiffness: 260, damping: 18 }}
      className="space-metric-card"
    >
      <div className="space-metric-card-top">
        <span>{label}</span>
        {Icon ? <Icon className="h-4 w-4" /> : null}
      </div>
      <div className="space-metric-value">{value}</div>
      {hint ? <div className="space-metric-hint">{hint}</div> : null}
    </motion.div>
  );
}

function JsonPanel({ title, data, onCopy }) {
  return (
    <div className="space-json-panel">
      <div className="space-json-header">
        <div className="space-json-title">
          <FileJson className="h-4 w-4" />
          <span>{title}</span>
        </div>
        <Button variant="outline" className="rounded-2xl" onClick={onCopy}>
          <Copy className="mr-2 h-4 w-4" />
          Copy
        </Button>
      </div>
      <div className="space-code-block">
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </div>
    </div>
  );
}

function ToastViewport({ toasts }) {
  return (
    <div className="space-toast-viewport">
      <AnimatePresence>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            initial={{ opacity: 0, x: 40, y: -8 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            exit={{ opacity: 0, x: 30, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className={cn("space-toast", `space-toast-${toast.type}`)}
          >
            <div className="space-toast-title">{toast.title}</div>
            {toast.description ? (
              <div className="space-toast-description">{toast.description}</div>
            ) : null}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

function OrbitVisualization({ lastPayload, inputMode, topRows }) {
  const points = useMemo(() => {
    if (inputMode === "manual" && lastPayload?.debris_states?.length) {
      const states = lastPayload.debris_states;
      const maxAbs = Math.max(
        1,
        ...states.flatMap((d) => [
          Math.abs(d?.pos?.x || 0),
          Math.abs(d?.pos?.y || 0),
          Math.abs(d?.pos?.z || 0),
        ]),
      );

      return states.map((d, i) => {
        const x = d?.pos?.x || 0;
        const y = d?.pos?.y || 0;
        return {
          id: d?.name || `Debris-${i + 1}`,
          x: 200 + (x / maxAbs) * 90,
          y: 130 + (y / maxAbs) * 90,
          high: false,
        };
      });
    }

    const rows = topRows?.length
      ? topRows
      : Array.from({ length: 6 }, (_, i) => ({ i }));

    return rows.map((row, i) => {
      const angle = (i / Math.max(rows.length, 1)) * Math.PI * 2 - Math.PI / 2;
      const radius = 86 + (i % 2) * 12;
      return {
        id: row?.debris_id || `Debris-${i + 1}`,
        x: 200 + Math.cos(angle) * radius,
        y: 130 + Math.sin(angle) * radius,
        high: Boolean(row?.is_high_risk),
      };
    });
  }, [inputMode, lastPayload, topRows]);

  return (
    <div className="space-orbit-panel">
      <div className="space-orbit-legend">
        <div>
          <span className="earth-dot" /> Earth
        </div>
        <div>
          <span className="sat-dot" /> Satellite
        </div>
        <div>
          <span className="debris-dot" /> Debris
        </div>
        <div>
          <span className="danger-dot" /> High Risk
        </div>
      </div>

      <svg
        viewBox="0 0 400 260"
        className="space-orbit-svg"
        role="img"
        aria-label="Orbit visualization"
      >
        <defs>
          <radialGradient id="earthGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="1" />
            <stop offset="100%" stopColor="#0f172a" stopOpacity="1" />
          </radialGradient>
          <linearGradient id="orbitRing" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="#06b6d4" />
          </linearGradient>
        </defs>

        <circle cx="200" cy="130" r="102" className="orbit-faint orbit-faint-animate" />
        <circle cx="200" cy="130" r="88" className="orbit-ring orbit-ring-animate" />
        <circle cx="200" cy="130" r="42" fill="url(#earthGlow)" />
        <circle cx="288" cy="130" r="6" className="satellite-node satellite-pulse" />

        {points.map((p, idx) => (
          <g key={`${p.id}-${idx}`}>
            <circle
              cx={p.x}
              cy={p.y}
              r={p.high ? 6 : 4.5}
              className={p.high ? "debris-node-high debris-pulse-high" : "debris-node debris-pulse"}
            />
            <line
              x1="288"
              y1="130"
              x2={p.x}
              y2={p.y}
              className="orbit-connection"
            />
          </g>
        ))}
      </svg>

      <p className="space-orbit-caption">
        Manual mode me real sampled payload positions use ho rahe hain. Real TLE
        mode me visual ring-based preview dikh raha hai.
      </p>
    </div>
  );
}

function SidebarContent({ status, response, onNavigate }) {
  const navItems = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "configure", label: "Configure", icon: Settings2 },
    { id: "results", label: "Results", icon: ShieldAlert },
    { id: "visuals", label: "Visuals", icon: BarChart3 },
    { id: "payload", label: "Payload", icon: FileJson },
  ];

  return (
    <div className="space-sidebar-inner">
      <div className="space-brand">
        <div className="space-brand-icon">
          <Rocket className="h-5 w-5" />
        </div>
        <div>
          <h2>Orbital Ops</h2>
          <p>Collision Dashboard</p>
        </div>
      </div>

      <div className="space-sidebar-group">
        <div className="space-sidebar-label">Navigation</div>
        <div className="space-nav-list">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                type="button"
                className="space-nav-btn"
                onClick={() => onNavigate(item.id)}
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="space-sidebar-group">
        <div className="space-sidebar-label">Run Health</div>
        <div className="space-mini-status-card">
          {status ? (
            <>
              <div className={cn("space-status-chip", status.tone)}>
                <status.icon className="h-4 w-4" />
                <span>{status.label}</span>
              </div>
              <p>{status.description}</p>
            </>
          ) : (
            <>
              <div className="space-status-chip neutral">
                <Bell className="h-4 w-4" />
                <span>Awaiting Run</span>
              </div>
              <p>Simulation start hote hi yaha quick summary show hogi.</p>
            </>
          )}
        </div>
      </div>

      <div className="space-sidebar-group">
        <div className="space-sidebar-label">Snapshot</div>
        <div className="space-mini-metrics">
          <div>
            <span>Min Distance</span>
            <strong>
              {formatNumber(response?.summary?.min_miss_distance)} m
            </strong>
          </div>
          <div>
            <span>High Risk</span>
            <strong>
              {formatNumber(response?.summary?.high_risk_count, 0)}
            </strong>
          </div>
          <div>
            <span>Max Prob.</span>
            <strong>
              {formatNumber(response?.summary?.max_probability, 8)}
            </strong>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function SpaceDebrisFrontend() {
  // const [apiBase, setApiBase] = useState("http://127.0.0.1:8000");
  const [apiBase, setApiBase] = useState(
  import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000"
);
  const [mode, setMode] = useState("auto");
  const [inputMode, setInputMode] = useState("real-tle");

  const [altitude, setAltitude] = useState(DEFAULT_ALTITUDE);
  const [debrisCount, setDebrisCount] = useState(3);
  const [allowInclination, setAllowInclination] = useState(false);
  const [lookahead, setLookahead] = useState(900);
  const [dt, setDt] = useState(2);
  const [debrisNames, setDebrisNames] = useState([
    "Debris-1",
    "Debris-2",
    "Debris-3",
  ]);

  const [satelliteNoradId, setSatelliteNoradId] = useState("25544");
  const [debrisFile, setDebrisFile] = useState("debris_ids.txt");
  const [debrisSource, setDebrisSource] = useState("backend");
  const [debrisIdsText, setDebrisIdsText] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState(null);
  const [lastPayload, setLastPayload] = useState(null);

  const [theme, setTheme] = useState("dark");
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    const stored = localStorage.getItem("space-ui-theme");
    if (stored === "light" || stored === "dark") {
      setTheme(stored);
    } else {
      const prefersDark = window.matchMedia(
        "(prefers-color-scheme: dark)",
      ).matches;
      setTheme(prefersDark ? "dark" : "light");
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("space-ui-theme", theme);
  }, [theme]);

  function pushToast(title, description = "", type = "info") {
    const id = `${Date.now()}-${Math.random()}`;
    setToasts((prev) => [...prev, { id, title, description, type }]);

    window.setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3200);
  }

  const status = useMemo(
    () => (response ? statusFromResponse(response) : null),
    [response],
  );

  const topRows = useMemo(() => {
    const rows = [...(response?.results || [])];
    rows.sort((a, b) => {
      const am = a?.miss_distance ?? a?.distance ?? Number.POSITIVE_INFINITY;
      const bm = b?.miss_distance ?? b?.distance ?? Number.POSITIVE_INFINITY;
      return am - bm;
    });

    const seen = new Set();
    return rows
      .filter((r, index) => {
        const id = r?.debris_id || `unknown-${index}`;
        if (seen.has(id)) return false;
        seen.add(id);
        return true;
      })
      .slice(0, 8);
  }, [response]);

  const chartRows = useMemo(() => {
    return topRows.map((row, idx) => ({
      name: truncateId(row.debris_id || `Debris-${idx + 1}`, 10),
      missDistance: Number(row.miss_distance ?? row.distance ?? 0),
      relativeVelocity: Number(row.relative_velocity ?? 0),
      probabilityScaled: Number(row.probability ?? 0) * 1000000,
      probabilityRaw: Number(row.probability ?? 0),
      risk: row.is_high_risk ? "High" : "Low",
    }));
  }, [topRows]);

  const riskPieData = useMemo(() => {
    const high = topRows.filter((r) => r.is_high_risk).length;
    const low = Math.max(topRows.length - high, 0);

    return [
      { name: "High", value: high },
      { name: "Low", value: low },
    ];
  }, [topRows]);

  function syncDebrisNames(count) {
    const n = Number(count);
    setDebrisNames((prev) =>
      Array.from({ length: n }, (_, i) => prev[i] || `Debris-${i + 1}`),
    );
  }

  function handleNoradFileUpload(file) {
    if (!file) return;

    const isTxt =
      file.type === "text/plain" || file.name.toLowerCase().endsWith(".txt");

    if (!isTxt) {
      setUploadedFileName("");
      setError("Please upload a .txt file containing one NORAD ID per line.");
      pushToast(
        "Invalid file",
        "Only .txt files are allowed for NORAD IDs.",
        "error",
      );
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result || "";
      setDebrisIdsText(String(text));
      setUploadedFileName(file.name);
      setError("");
      pushToast(
        "File loaded",
        `${file.name} imported successfully.`,
        "success",
      );
    };
    reader.onerror = () => {
      setError("Failed to read uploaded file.");
      pushToast("Read failed", "Could not read uploaded file.", "error");
    };
    reader.readAsText(file);
  }

  async function copyJsonBlock(label, data) {
    try {
      await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
      pushToast("Copied", `${label} copied to clipboard.`, "success");
    } catch {
      pushToast("Copy failed", "Clipboard action was not successful.", "error");
    }
  }

  async function handleRun() {
    try {
      setLoading(true);
      setError("");

      if (!apiBase.trim()) {
        throw new Error("Please enter a valid API base URL.");
      }

      if (Number(lookahead) <= 0 || Number.isNaN(Number(lookahead))) {
        throw new Error("Lookahead must be greater than 0.");
      }

      if (Number(dt) <= 0 || Number.isNaN(Number(dt))) {
        throw new Error("Time Step dt must be greater than 0.");
      }

      if (inputMode === "real-tle") {
        if (!satelliteNoradId || Number.isNaN(Number(satelliteNoradId))) {
          throw new Error("Please enter a valid Satellite NORAD ID.");
        }

        if (debrisSource === "upload" && !debrisIdsText.trim()) {
          throw new Error("Please upload or paste NORAD IDs.");
        }
      } else {
        if (Number.isNaN(Number(altitude))) {
          throw new Error("Please enter a valid altitude.");
        }

        if (Number(debrisCount) < 1 || Number.isNaN(Number(debrisCount))) {
          throw new Error("Debris count must be at least 1.");
        }
      }

      const base = apiBase.replace(/\/$/, "");
      let endpoint = "/runs";
      let payload = null;

      if (inputMode === "real-tle") {
        endpoint = "/simulate/real-tle";
        payload = buildRealTLEPayload({
          mode,
          lookahead,
          dt,
          satelliteNoradId,
          debrisSource,
          debrisFile,
          debrisIdsText,
        });
      } else {
        endpoint = "/simulate/";
        payload = buildManualPayload({
          altitude,
          debrisCount,
          allowInclination,
          mode,
          lookahead,
          dt,
          debrisNames,
        });
      }

      setLastPayload(payload);

      const res = await fetch(`${base}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(
          typeof data?.detail === "string"
            ? data.detail
            : JSON.stringify(data?.detail || "Simulation failed"),
        );
      }

      setResponse(data);
      pushToast(
        "Simulation complete",
        "Backend response received successfully.",
        "success",
      );
      scrollToSection("results");
    } catch (err) {
      const message =
        err?.message || "Something went wrong while running simulation.";
      setError(message);
      setResponse(null);
      pushToast("Request failed", message, "error");
    } finally {
      setLoading(false);
    }
  }

  function clearResults() {
    setResponse(null);
    setError("");
    pushToast("Cleared", "Current response and errors were cleared.", "info");
  }

  return (
    <div className={cn("space-dashboard", theme)}>
      <div className="space-bg-orb orb-1" />
      <div className="space-bg-orb orb-2" />
      <div className="space-bg-orb orb-3" />
      <div className="space-grid-overlay" />

      <ToastViewport toasts={toasts} />

      <AnimatePresence>
        {mobileSidebarOpen && (
          <>
            <motion.div
              className="space-mobile-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileSidebarOpen(false)}
            />
            <motion.aside
              className="space-sidebar-mobile"
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{ type: "spring", stiffness: 260, damping: 24 }}
            >
              <div className="space-mobile-sidebar-top">
                <Button
                  variant="outline"
                  className="rounded-2xl"
                  onClick={() => setMobileSidebarOpen(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>

              <SidebarContent
                status={status}
                response={response}
                onNavigate={(id) => {
                  scrollToSection(id);
                  setMobileSidebarOpen(false);
                }}
              />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <div className="space-layout">
        <aside className="space-sidebar desktop-only">
          <SidebarContent
            status={status}
            response={response}
            onNavigate={(id) => scrollToSection(id)}
          />
        </aside>

        <div className="space-main">
          <motion.header
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="space-topbar"
          >
            <div className="space-topbar-left">
              <Button
                variant="outline"
                className="rounded-2xl mobile-only space-header-btn"
                onClick={() => setMobileSidebarOpen(true)}
              >
                <Menu className="h-4 w-4" />
              </Button>

              <div className="space-title-wrap">
                <h1>Space Debris Collision Engine</h1>
                <p>Responsive risk dashboard with charts, orbit preview, and live simulation controls</p>
              </div>
            </div>

            <div className="space-topbar-actions">
              <div className="space-inline-status">
                {status ? (
                  <div className={cn("space-status-chip", status.tone)}>
                    <status.icon className="h-4 w-4" />
                    <span>{status.label}</span>
                  </div>
                ) : (
                  <div className="space-status-chip neutral">
                    <Radar className="h-4 w-4" />
                    <span>Idle</span>
                  </div>
                )}
              </div>

              <Button
                variant="outline"
                className="rounded-2xl space-header-btn"
                onClick={() =>
                  setTheme((prev) => (prev === "dark" ? "light" : "dark"))
                }
              >
                {theme === "dark" ? (
                  <>
                    <Sun className="mr-2 h-4 w-4" />
                    Light
                  </>
                ) : (
                  <>
                    <Moon className="mr-2 h-4 w-4" />
                    Dark
                  </>
                )}
              </Button>

              <Button
                className="space-primary-btn space-run-btn"
                onClick={handleRun}
                disabled={loading}
              >
                {loading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Rocket className="mr-2 h-4 w-4" />
                )}
                {loading ? "Running..." : "Run"}
              </Button>
            </div>
          </motion.header>

          <main className="space-content">
            <motion.section id="overview" className="space-section" {...sectionMotion}>
              <div className="space-hero-card">
                <div className="space-hero-stars">
                  <span />
                  <span />
                  <span />
                  <span />
                </div>

                <div className="space-hero-content">
                  <div className="space-hero-copy">
                    <Badge className="space-badge-gradient">
                      Orbital Risk Intelligence
                    </Badge>
                    <h2>Space Debris Collision Engine Interface</h2>

                    <p>
                      This frontend connects to a locally hosted backend running
                      on my personal machine.
                    </p>

                    <p>
                      Due to the heavy computational workload, live backend
                      access is available daily from
                      <strong> 11:00 AM – 1:00 PM (IST)</strong> via ngrok.
                      Outside this window, the API may be offline.
                    </p>

                    <div className="space-hero-pill-row">
                      <span>Responsive</span>
                      <span>Dark Mode</span>
                      <span>Charts</span>
                      <span>Orbit Preview</span>
                      <span>Toasts</span>
                    </div>
                  </div>

                  <div className="space-hero-stats-grid">
                    <MetricCard
                      label="Input Mode"
                      value={inputMode === "real-tle" ? "Real TLE" : "Manual"}
                      icon={Satellite}
                    />
                    <MetricCard
                      label="Run Mode"
                      value={String(mode).toUpperCase()}
                      icon={Activity}
                    />
                    <MetricCard
                      label="Lookahead"
                      value={`${lookahead}s`}
                      icon={Radar}
                    />
                    <MetricCard
                      label="Time Step"
                      value={`${dt}s`}
                      icon={Orbit}
                    />
                  </div>
                </div>
              </div>
            </motion.section>

            <motion.section id="configure" className="space-section" {...sectionMotion}>
              <Card className="space-surface">
                <CardHeader>
                  <CardTitle>Configure Simulation</CardTitle>
                  <CardDescription>
                    Set backend endpoint, choose mode, and prepare either manual
                    or real TLE input.
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-card-stack">
                  <div className="space-config-grid">
                    <div className="space-config-block">
                      <SectionTitle
                        icon={Rocket}
                        title="Connection & Core Settings"
                        subtitle="Basic request setup before running the backend."
                      />

                      <div className="grid gap-2">
                        <Label>API Base URL</Label>
                        <Input
                          value={apiBase}
                          onChange={(e) => setApiBase(e.target.value)}
                          placeholder="http://127.0.0.1:8000"
                          className="space-input"
                        />
                      </div>

                      <SegmentedControl
                        label="Input Mode"
                        value={inputMode}
                        onChange={setInputMode}
                        options={INPUT_MODE_OPTIONS}
                      />

                      <SegmentedControl
                        label="Run Mode"
                        value={mode}
                        onChange={setMode}
                        options={RUN_MODE_OPTIONS}
                      />

                      <div className="grid gap-4 sm:grid-cols-2">
                        <div className="grid gap-2">
                          <Label>Lookahead (seconds)</Label>
                          <Input
                            type="number"
                            value={lookahead}
                            onChange={(e) => setLookahead(e.target.value)}
                            className="space-input"
                          />
                        </div>

                        <div className="grid gap-2">
                          <Label>Time Step dt</Label>
                          <Input
                            type="number"
                            value={dt}
                            onChange={(e) => setDt(e.target.value)}
                            className="space-input"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="space-config-block">
                      {inputMode === "real-tle" ? (
                        <>
                          <SectionTitle
                            icon={Satellite}
                            title="Real TLE / NORAD Inputs"
                            subtitle="Use live IDs, backend debris file, or upload your own list."
                          />

                          <div className="grid gap-2">
                            <Label>Satellite NORAD ID</Label>
                            <Input
                              type="number"
                              value={satelliteNoradId}
                              onChange={(e) =>
                                setSatelliteNoradId(e.target.value)
                              }
                              placeholder="25544"
                              className="space-input"
                            />
                          </div>

                          <SegmentedControl
                            label="Debris Source"
                            value={debrisSource}
                            onChange={(value) => {
                              setDebrisSource(value);
                              setError("");
                            }}
                            options={DEBRIS_SOURCE_OPTIONS}
                          />

                          {debrisSource === "backend" ? (
                            <div className="grid gap-2">
                              <Label>Backend Debris File</Label>
                              <Input
                                value={debrisFile}
                                onChange={(e) => setDebrisFile(e.target.value)}
                                placeholder="debris_ids.txt"
                                className="space-input"
                              />
                              <p className="space-help-text">
                                Ye file tumhara backend server read karega.
                              </p>
                            </div>
                          ) : (
                            <div className="space-card-stack">
                              <div className="grid gap-2">
                                <Label>Upload NORAD ID TXT File</Label>
                                <Input
                                  type="file"
                                  accept=".txt,text/plain"
                                  className="space-input"
                                  onChange={(e) =>
                                    handleNoradFileUpload(e.target.files?.[0])
                                  }
                                />
                                {uploadedFileName ? (
                                  <div className="space-inline-note success">
                                    Loaded file:{" "}
                                    <strong>{uploadedFileName}</strong>
                                  </div>
                                ) : null}
                                <p className="space-help-text">
                                  One NORAD ID per line. Example: 59588, 24463,
                                  58296...
                                </p>
                              </div>

                              <div className="grid gap-2">
                                <Label>Or Paste NORAD IDs</Label>
                                <textarea
                                  className="space-textarea"
                                  value={debrisIdsText}
                                  onChange={(e) =>
                                    setDebrisIdsText(e.target.value)
                                  }
                                  placeholder={`59588
24463
58296
63746
63748`}
                                />
                                <p className="space-help-text">
                                  Plain text only. One NORAD ID per line.
                                </p>
                              </div>
                            </div>
                          )}
                        </>
                      ) : (
                        <>
                          <SectionTitle
                            icon={Sparkles}
                            title="Manual Orbit Demo"
                            subtitle="Generate synthetic debris states for quick frontend/backed testing."
                          />

                          <div className="grid gap-4 sm:grid-cols-2">
                            <div className="grid gap-2">
                              <Label>Altitude above Earth (m)</Label>
                              <Input
                                type="number"
                                value={altitude}
                                onChange={(e) => setAltitude(e.target.value)}
                                className="space-input"
                              />
                            </div>

                            <div className="grid gap-2">
                              <Label>Number of Debris</Label>
                              <Input
                                type="number"
                                min={1}
                                max={500}
                                value={debrisCount}
                                onChange={(e) => {
                                  setDebrisCount(e.target.value);
                                  syncDebrisNames(e.target.value);
                                }}
                                className="space-input"
                              />
                            </div>
                          </div>

                          <div className="space-switch-card">
                            <div>
                              <Label className="text-sm">
                                Allow random inclinations
                              </Label>
                              <p className="space-help-text">
                                Debris thoda aur realistic 3D spread ke saath
                                generate hoga.
                              </p>
                            </div>
                            <Switch
                              checked={allowInclination}
                              onCheckedChange={setAllowInclination}
                            />
                          </div>

                          <div className="space-card-stack">
                            <div className="flex items-center justify-between gap-3 flex-wrap">
                              <Label>Debris Names</Label>
                              <Badge
                                variant="secondary"
                                className="rounded-full"
                              >
                                Optional
                              </Badge>
                            </div>

                            <div className="grid gap-2">
                              {Array.from(
                                { length: Number(debrisCount) },
                                (_, i) => (
                                  <div key={i} className="flex gap-2 flex-wrap sm:flex-nowrap">
                                    <Input
                                      value={debrisNames[i] ?? ""}
                                      onChange={(e) => {
                                        const next = [...debrisNames];
                                        next[i] = e.target.value;
                                        setDebrisNames(next);
                                      }}
                                      placeholder={`Debris-${i + 1}`}
                                      className="space-input"
                                    />
                                    <Button
                                      type="button"
                                      variant="outline"
                                      className="rounded-2xl space-icon-btn"
                                      onClick={() => {
                                        const next = [...debrisNames];
                                        next[i] = `Debris-${i + 1}`;
                                        setDebrisNames(next);
                                      }}
                                    >
                                      <Sparkles className="h-4 w-4" />
                                    </Button>
                                  </div>
                                ),
                              )}
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  </div>

                  <Separator />

                  <div className="space-action-row">
                    <Button
                      className="space-primary-btn"
                      onClick={handleRun}
                      disabled={loading}
                    >
                      {loading ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Rocket className="mr-2 h-4 w-4" />
                      )}
                      {loading ? "Running Simulation..." : "Run Simulation"}
                    </Button>

                    <Button
                      variant="outline"
                      className="rounded-2xl"
                      onClick={clearResults}
                    >
                      <Trash2 className="mr-2 h-4 w-4" />
                      Clear
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.section>

            <motion.section id="results" className="space-section" {...sectionMotion}>
              <Card className="space-surface">
                <CardHeader>
                  <CardTitle>Run Summary & Results</CardTitle>
                  <CardDescription>
                    Backend verdict, risk metrics, and animated result cards.
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-card-stack">
                  {error ? (
                    <div className="space-error-box">
                      <div className="space-error-title">
                        <AlertTriangle className="h-4 w-4" />
                        <span>Request failed</span>
                      </div>
                      <p>{error}</p>
                    </div>
                  ) : !response ? (
                    <div className="space-empty-box">
                      Run a simulation to view danger verdict, metrics, debris
                      shortlist, and downloadable JSON.
                    </div>
                  ) : (
                    <>
                      {status ? (
                        <motion.div
                          initial={{ opacity: 0, y: 12 }}
                          animate={{ opacity: 1, y: 0 }}
                          className={cn("space-status-banner", status.tone)}
                        >
                          <div className="space-status-banner-icon">
                            <status.icon className="h-5 w-5" />
                          </div>
                          <div>
                            <h4>{status.label}</h4>
                            <p>{status.description}</p>
                          </div>
                        </motion.div>
                      ) : null}

                      <div className="space-metric-grid">
                        <MetricCard
                          label="Min Miss Distance"
                          value={`${formatNumber(response?.summary?.min_miss_distance)} m`}
                          icon={Radar}
                        />
                        <MetricCard
                          label="Max Probability"
                          value={formatNumber(
                            response?.summary?.max_probability,
                            8,
                          )}
                          icon={ShieldAlert}
                        />
                        <MetricCard
                          label="High Risk Count"
                          value={formatNumber(
                            response?.summary?.high_risk_count,
                            0,
                          )}
                          icon={AlertTriangle}
                        />
                        <MetricCard
                          label="Mode"
                          value={String(
                            response?.meta?.mode || "—",
                          ).toUpperCase()}
                          icon={Activity}
                        />
                      </div>

                      {response?.real_tle_meta ? (
                        <div className="space-meta-panel">
                          <SectionTitle
                            icon={Database}
                            title="Real TLE Metadata"
                            subtitle="Additional info returned by backend for TLE-based run."
                          />

                          <div className="space-mini-metadata-grid">
                            <div>
                              <span>Satellite NORAD</span>
                              <strong>
                                {response.real_tle_meta.satellite_norad_id}
                              </strong>
                            </div>
                            <div>
                              <span>Debris Source</span>
                              <strong>
                                {response.real_tle_meta.debris_source || "—"}
                              </strong>
                            </div>
                            <div>
                              <span>Loaded Debris</span>
                              <strong>
                                {formatNumber(
                                  response.real_tle_meta.loaded_debris_count,
                                  0,
                                )}
                              </strong>
                            </div>
                            <div>
                              <span>Debris File</span>
                              <strong>
                                {response.real_tle_meta.debris_file || "—"}
                              </strong>
                            </div>
                          </div>

                          {!!response.real_tle_meta.skipped_debris?.length && (
                            <div className="space-inline-note warn">
                              Skipped debris:{" "}
                              <strong>
                                {response.real_tle_meta.skipped_debris.length}
                              </strong>
                            </div>
                          )}
                        </div>
                      ) : null}

                      <div className="space-action-row">
                        <Button
                          className="space-primary-btn"
                          onClick={() => downloadJson(response)}
                        >
                          <Download className="mr-2 h-4 w-4" />
                          Download Result JSON
                        </Button>

                        {lastPayload ? (
                          <Button
                            variant="outline"
                            className="rounded-2xl"
                            onClick={() =>
                              downloadJson(
                                lastPayload,
                                "space-debris-request.json",
                              )
                            }
                          >
                            <Download className="mr-2 h-4 w-4" />
                            Download Request JSON
                          </Button>
                        ) : null}

                        <Button
                          variant="outline"
                          className="rounded-2xl"
                          onClick={() => copyJsonBlock("Result JSON", response)}
                        >
                          <Copy className="mr-2 h-4 w-4" />
                          Copy Result
                        </Button>
                      </div>

                      <div className="space-result-grid">
                        {topRows.length === 0 ? (
                          <div className="space-empty-box">
                            No debris entries found in backend response.
                          </div>
                        ) : (
                          topRows.map((row, idx) => (
                            <motion.div
                              key={`${row.debris_id || "unknown"}-${idx}`}
                              initial={{ opacity: 0, y: 12 }}
                              animate={{ opacity: 1, y: 0 }}
                              whileHover={{ y: -5, scale: 1.01 }}
                              transition={{
                                type: "spring",
                                stiffness: 250,
                                damping: 18,
                              }}
                              className={cn(
                                "space-result-card",
                                row.is_high_risk && "high-risk",
                              )}
                            >
                              <div className="space-result-card-top">
                                <div>
                                  <p className="space-result-label">
                                    Debris #{idx + 1}
                                  </p>
                                  <h4>{row.debris_id || "—"}</h4>
                                </div>

                                {row.is_high_risk ? (
                                  <Badge className="rounded-full bg-red-600 hover:bg-red-600">
                                    High
                                  </Badge>
                                ) : (
                                  <Badge
                                    variant="secondary"
                                    className="rounded-full"
                                  >
                                    Low
                                  </Badge>
                                )}
                              </div>

                              <div className="space-result-stats">
                                <div>
                                  <span>Miss Distance</span>
                                  <strong>
                                    {formatNumber(
                                      row.miss_distance ?? row.distance,
                                    )}{" "}
                                    m
                                  </strong>
                                </div>
                                <div>
                                  <span>Relative Velocity</span>
                                  <strong>
                                    {formatNumber(row.relative_velocity)} m/s
                                  </strong>
                                </div>
                                <div>
                                  <span>Probability</span>
                                  <strong>
                                    {formatNumber(row.probability, 8)}
                                  </strong>
                                </div>
                              </div>
                            </motion.div>
                          ))
                        )}
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </motion.section>

            <motion.section id="visuals" className="space-section" {...sectionMotion}>
              <Card className="space-surface">
                <CardHeader>
                  <CardTitle>Charts & Orbital Visualization</CardTitle>
                  <CardDescription>
                    Better human-readable overview of miss distance, velocity,
                    risk split, and orbit preview.
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-visual-grid">
                  <motion.div
                    whileHover={{ y: -4 }}
                    transition={{ duration: 0.2 }}
                    className="space-chart-card"
                  >
                    <div className="space-chart-title">Miss Distance</div>
                    {chartRows.length ? (
                      <div className="space-chart-wrap">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartRows}>
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="var(--chart-grid)"
                            />
                            <XAxis dataKey="name" stroke="var(--chart-axis)" />
                            <YAxis stroke="var(--chart-axis)" />
                            <Tooltip
                              contentStyle={{
                                background: "var(--tooltip-bg)",
                                border: "1px solid var(--tooltip-border)",
                                borderRadius: 16,
                                color: "var(--text-primary)",
                              }}
                            />
                            <Bar
                              dataKey="missDistance"
                              radius={[10, 10, 0, 0]}
                              fill="#8b5cf6"
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="space-chart-empty">
                        No chart data yet.
                      </div>
                    )}
                  </motion.div>

                  <motion.div
                    whileHover={{ y: -4 }}
                    transition={{ duration: 0.2 }}
                    className="space-chart-card"
                  >
                    <div className="space-chart-title">Relative Velocity</div>
                    {chartRows.length ? (
                      <div className="space-chart-wrap">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={chartRows}>
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="var(--chart-grid)"
                            />
                            <XAxis dataKey="name" stroke="var(--chart-axis)" />
                            <YAxis stroke="var(--chart-axis)" />
                            <Tooltip
                              contentStyle={{
                                background: "var(--tooltip-bg)",
                                border: "1px solid var(--tooltip-border)",
                                borderRadius: 16,
                                color: "var(--text-primary)",
                              }}
                            />
                            <Area
                              type="monotone"
                              dataKey="relativeVelocity"
                              stroke="#06b6d4"
                              fill="#06b6d4"
                              fillOpacity={0.22}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="space-chart-empty">
                        No chart data yet.
                      </div>
                    )}
                  </motion.div>

                  <motion.div
                    whileHover={{ y: -4 }}
                    transition={{ duration: 0.2 }}
                    className="space-chart-card"
                  >
                    <div className="space-chart-title">Risk Split</div>
                    {topRows.length ? (
                      <div className="space-chart-wrap">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={riskPieData}
                              cx="50%"
                              cy="50%"
                              outerRadius={80}
                              dataKey="value"
                              nameKey="name"
                              label
                            >
                              <Cell fill="#ef4444" />
                              <Cell fill="#10b981" />
                            </Pie>
                            <Tooltip
                              contentStyle={{
                                background: "var(--tooltip-bg)",
                                border: "1px solid var(--tooltip-border)",
                                borderRadius: 16,
                                color: "var(--text-primary)",
                              }}
                            />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="space-chart-empty">
                        No chart data yet.
                      </div>
                    )}
                  </motion.div>

                  <motion.div
                    whileHover={{ y: -4 }}
                    transition={{ duration: 0.2 }}
                    className="space-chart-card orbit-card"
                  >
                    <div className="space-chart-title">
                      Orbital Visualization Panel
                    </div>
                    <OrbitVisualization
                      lastPayload={lastPayload}
                      inputMode={inputMode}
                      topRows={topRows}
                    />
                  </motion.div>
                </CardContent>
              </Card>
            </motion.section>

            <motion.section id="payload" className="space-section" {...sectionMotion}>
              <Card className="space-surface">
                <CardHeader>
                  <CardTitle>Payload & Response Preview</CardTitle>
                  <CardDescription>
                    Exact request payload and raw response for debugging or
                    backend inspection.
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-json-grid">
                  <JsonPanel
                    title="Request Payload"
                    data={lastPayload}
                    onCopy={() => copyJsonBlock("Request JSON", lastPayload)}
                  />

                  <JsonPanel
                    title="Raw Response"
                    data={response}
                    onCopy={() => copyJsonBlock("Response JSON", response)}
                  />
                </CardContent>
              </Card>
            </motion.section>
          </main>
        </div>
      </div>
    </div>
  );
}