"use client";

import { useState } from "react";
import UploadArea from "@/components/UploadArea";
import ResultDisplay from "@/components/ResultDisplay";
import { Brain, ShieldAlert } from "lucide-react";

interface PredictionResult {
  prediction_4class: string;
  prediction_binary: "healthy" | "unhealthy";
  confidence_scores: Record<string, number>;
  binary_confidence: number;
  is_uncertain?: boolean;
  top_class?: string;
  top_class_confidence?: number;
  uncertain_threshold?: number;
  heatmap_base64?: string;
  image_url?: string;
}

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    setError(null);

    // Create a local preview URL for immediate display
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Failed to analyze image. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setPreviewUrl(null);
  };

  return (
    <main className="min-h-screen p-8 pb-24">
      <div className="max-w-5xl mx-auto space-y-18">
        {/* Header */}
        <div className="text-center space-y-4 animate-fade-in">
          <div className="inline-flex items-center justify-center p-3 rounded-2xl bg-cyan-500/10 mb-4">
            <Brain className="w-12 h-12 text-cyan-400" />
          </div>
          <h1 className="text-5xl font-bold tracking-tight text-white">
            BrainScan <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">AI</span>
          </h1>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Advanced deep learning analysis for MRI scans. Detects Glioma, Meningioma, Pituitary, and no tumors with high precision.
          </p>

          {/* Disclaimer */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-sm font-medium mt-4">
            <ShieldAlert className="w-4 h-4" />
            <span>Research Prototype - Not for Medical Diagnosis</span>
          </div>
        </div>

        {/* Main Content */}
        <div className="transition-all duration-500 ease-in-out">
          {error && (
            <div className="max-w-md mx-auto mb-8 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-center">
              {error}
            </div>
          )}

          {!result ? (
            <UploadArea onFileSelect={handleFileUpload} isAnalyzing={loading} />
          ) : (
            <ResultDisplay
              result={result}
              onReset={handleReset}
              initialImageUrl={previewUrl || result.image_url}
            />
          )}
        </div>

        {/* Team Members Footer */}
        <div className="mt-12 pt-8 border-t border-white/5 text-center">
          <p className="text-sm text-slate-500 mb-2 uppercase tracking-widest font-semibold">Project Team</p>
          <div className="flex flex-wrap justify-center gap-x-6 gap-y-2 text-slate-400 text-sm">
            <span>Christian Cuevas</span>
            <span>Aidan McNamara</span>
            <span>Behrens Richeson</span>
            <span>Caleb Zeringue</span>
            <span>Felix Schafer</span>
            <span>Austin Louque</span>
          </div>
        </div>
      </div>
    </main>
  );
}
