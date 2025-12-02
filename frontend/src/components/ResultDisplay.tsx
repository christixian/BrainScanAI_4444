"use client";

import { useState } from "react";
import { clsx } from "clsx";
import { Activity, AlertTriangle, CheckCircle, RefreshCw, Layers, Eye, FileText, Wand2 } from "lucide-react";
import MedicalInfo from "./MedicalInfo";

interface PredictionResult {
    prediction_4class: string;
    prediction_binary: "healthy" | "unhealthy" | "uncertain";
    confidence_scores: Record<string, number>;
    binary_confidence: number;
    is_uncertain?: boolean;
    top_class?: string;
    top_class_confidence?: number;
    uncertain_threshold?: number;
    heatmap_base64?: string;
}

interface ResultDisplayProps {
    result: PredictionResult;
    onReset: () => void;
    initialImageUrl?: string; // URL of the original uploaded image
}

type ViewMode = "original" | "heatmap";

export default function ResultDisplay({ result, onReset, initialImageUrl }: ResultDisplayProps) {
    const [viewMode, setViewMode] = useState<ViewMode>("original");

    const isUncertain = (result.is_uncertain ?? false)
        || result.prediction_4class.toLowerCase() === "uncertain"
        || result.prediction_binary === "uncertain";
    const isHealthy = result.prediction_binary === "healthy" && !isUncertain;

    // Sort confidence scores for display
    const sortedScores = Object.entries(result.confidence_scores).sort(
        ([, a], [, b]) => b - a
    );
    const topScoreEntry = sortedScores[0];
    const topLabel = topScoreEntry?.[0];
    const topScore = topScoreEntry?.[1] ?? 0;
    const highlightedLabel = isUncertain ? topLabel : result.prediction_4class;
    const formatLabel = (label?: string) => {
        if (!label) return "No prediction";
        return label === "notumor" ? "No Tumor" : label;
    };
    const threshold = result.uncertain_threshold ?? 0.5;
    const topClassConfidence = result.top_class_confidence ?? topScore;

    const handleConvertToHeatmap = () => {
        setViewMode("heatmap");
    };

    return (
        <div className="w-full max-w-4xl mx-auto space-y-8 animate-fade-in pb-20">
            {/* Main Result Banner */}
            <div
                className={clsx(
                    "relative overflow-hidden rounded-2xl p-8 text-center border shadow-2xl transition-all",
                    isUncertain
                        ? "bg-amber-950/40 border-amber-500/30 shadow-amber-900/20"
                        : isHealthy
                        ? "bg-emerald-950/30 border-emerald-500/30 shadow-emerald-900/20"
                        : "bg-rose-950/30 border-rose-500/30 shadow-rose-900/20"
                )}
            >
                <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent pointer-events-none" />

                <div className="relative z-10 flex flex-col items-center gap-4">
                    {isUncertain ? (
                        <AlertTriangle className="w-16 h-16 text-amber-300 drop-shadow-[0_0_15px_rgba(251,191,36,0.4)]" />
                    ) : isHealthy ? (
                        <CheckCircle className="w-16 h-16 text-emerald-400 drop-shadow-[0_0_15px_rgba(52,211,153,0.5)]" />
                    ) : (
                        <AlertTriangle className="w-16 h-16 text-rose-400 drop-shadow-[0_0_15px_rgba(251,113,133,0.5)]" />
                    )}

                    <div>
                        <h2 className="text-3xl font-bold tracking-tight text-white mb-1 capitalize">
                            {isUncertain
                                ? "Uncertain Result"
                                : result.prediction_4class === "notumor"
                                    ? "No Tumor"
                                    : result.prediction_4class}
                        </h2>
                        <p className={clsx("text-lg font-medium uppercase tracking-widest opacity-90",
                            isUncertain
                                ? "text-amber-300"
                                : isHealthy
                                    ? "text-emerald-300"
                                    : "text-rose-300"
                        )}>
                            {isUncertain
                                ? `Highest match: ${formatLabel(result.top_class ?? topLabel)} (${(topClassConfidence * 100).toFixed(1)}%)`
                                : (isHealthy ? "No Tumor Detected" : "Tumor Detected")}
                        </p>
                    </div>

                    <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-black/20 backdrop-blur-md border border-white/10">
                        <Activity className="w-4 h-4 text-slate-400" />
                        <span className="text-sm font-mono text-slate-300">
                            {isUncertain
                                ? `Confidence below ${(threshold * 100).toFixed(0)}% (top ${(topClassConfidence * 100).toFixed(1)}%)`
                                : `Tumor Likelihood: ${(isHealthy ? (1 - result.binary_confidence) * 100 : result.binary_confidence * 100).toFixed(1)}%`}
                        </span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left Column: Image & Visualization */}
                <div className="space-y-6">
                    <div className="glass-panel rounded-xl p-6 space-y-4">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
                                Scan Analysis
                            </h3>
                            {viewMode === "heatmap" && (
                                <button
                                    onClick={() => setViewMode("original")}
                                    className="text-xs flex items-center gap-1 text-slate-400 hover:text-white transition-colors"
                                >
                                    <Eye className="w-3 h-3" />
                                    Show Original
                                </button>
                            )}
                        </div>

                        {/* Image Display Area */}
                        <div className="relative aspect-square w-full rounded-lg overflow-hidden border border-slate-700 shadow-2xl bg-black">
                            {viewMode === "original" ? (
                                initialImageUrl ? (
                                    <img
                                        src={initialImageUrl}
                                        alt="Original Scan"
                                        className="w-full h-full object-cover"
                                    />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-slate-500">
                                        No Original Image Available
                                    </div>
                                )
                            ) : (
                                <img
                                    src={result.heatmap_base64}
                                    alt="AI Analysis"
                                    className="w-full h-full object-cover animate-fade-in"
                                />
                            )}
                        </div>

                        {/* Convert Button */}
                        {viewMode === "original" && result.heatmap_base64 && (
                            <button
                                onClick={handleConvertToHeatmap}
                                className="w-full group relative px-6 py-3 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/50 transition-all hover:scale-[1.02] active:scale-[0.98]"
                            >
                                <span className="flex items-center justify-center gap-2">
                                    <Wand2 className="w-5 h-5" />
                                    Convert to Heatmap
                                </span>
                            </button>
                        )}

                        {viewMode === "heatmap" && (
                            <div className="text-xs text-center text-slate-400">
                                Red areas indicate regions most important to the AI's decision.
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Column: Stats & Info */}
                <div className="space-y-6">
                    {/* Detailed Confidence Scores */}
                    <div className="glass-panel rounded-xl p-6 space-y-4">
                        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
                            Class Probabilities
                        </h3>
                        <div className="space-y-3">
                            {sortedScores.map(([label, score]) => (
                                <div key={label} className="group">
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="capitalize text-slate-200 font-medium">
                                            {formatLabel(label)}
                                        </span>
                                        <span className="font-mono text-slate-400">{(score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                        <div
                                            className={clsx(
                                                "h-full rounded-full transition-all duration-1000 ease-out",
                                                label === highlightedLabel
                                                    ? (isUncertain
                                                        ? "bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.5)]"
                                                        : isHealthy
                                                            ? "bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]"
                                                            : "bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]")
                                                    : "bg-slate-600 opacity-50"
                                            )}
                                            style={{ width: `${score * 100}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-4">
                        <button
                            onClick={onReset}
                            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-medium transition-all border border-slate-700 hover:border-slate-500"
                        >
                            <RefreshCw className="w-4 h-4" />
                            New Scan
                        </button>
                    </div>
                </div>
            </div>

            {/* Medical Information Section */}
            <MedicalInfo prediction={result.prediction_4class} />
        </div>
    );
}
