"use client";

import { useState, useCallback } from "react";
import { Upload, FileImage, AlertCircle } from "lucide-react";
import { clsx } from "clsx";

interface UploadAreaProps {
    onFileSelect: (file: File) => void;
    isAnalyzing: boolean;
}

export default function UploadArea({ onFileSelect, isAnalyzing }: UploadAreaProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [preview, setPreview] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setIsDragging(true);
        } else if (e.type === "dragleave") {
            setIsDragging(false);
        }
    }, []);

    const validateAndSetFile = (file: File) => {
        setError(null);
        if (!file.type.startsWith("image/")) {
            setError("Please upload an image file (JPG, PNG, BMP).");
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            setError("File size exceeds 20MB limit.");
            return;
        }

        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);
        onFileSelect(file);
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            validateAndSetFile(e.dataTransfer.files[0]);
        }
    }, [onFileSelect]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            validateAndSetFile(e.target.files[0]);
        }
    };

    return (
        <div className="w-full max-w-xl mx-auto animate-fade-in">
            <div
                className={clsx(
                    "relative border-2 border-dashed rounded-2xl p-10 transition-all duration-300 text-center cursor-pointer glass-panel",
                    isDragging
                        ? "border-cyan-500 bg-cyan-500/10 scale-[1.02]"
                        : "border-slate-600 hover:border-slate-400 hover:bg-slate-800/50",
                    isAnalyzing && "opacity-50 pointer-events-none"
                )}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => document.getElementById("file-upload")?.click()}
            >
                <input
                    id="file-upload"
                    type="file"
                    className="hidden"
                    accept="image/*"
                    onChange={handleChange}
                    disabled={isAnalyzing}
                />

                {preview ? (
                    <div className="relative aspect-square max-h-64 mx-auto overflow-hidden rounded-lg shadow-lg">
                        <img
                            src={preview}
                            alt="Preview"
                            className="object-cover w-full h-full"
                        />
                        {isAnalyzing && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                                <div className="w-12 h-12 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center py-10 space-y-4">
                        <div className="p-4 bg-slate-800 rounded-full shadow-inner">
                            <Upload className="w-10 h-10 text-cyan-400" />
                        </div>
                        <div className="space-y-1">
                            <p className="text-lg font-medium text-slate-200">
                                Click or drag MRI scan here
                            </p>
                            <p className="text-sm text-slate-400">
                                Supports JPG, PNG, BMP (Max 20MB)
                            </p>
                        </div>
                    </div>
                )}
            </div>

            {error && (
                <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-red-400 text-sm">
                    <AlertCircle className="w-4 h-4" />
                    {error}
                </div>
            )}
        </div>
    );
}
