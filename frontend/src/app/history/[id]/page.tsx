"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import ResultDisplay from "@/components/ResultDisplay";
import { ArrowLeft, Loader2 } from "lucide-react";

interface HistoryItem {
    id: number;
    timestamp: string;
    prediction_4class: string;
    prediction_binary: "healthy" | "unhealthy";
    confidence_scores: Record<string, number>;
    binary_confidence: number;
    heatmap_base64?: string;
    image_url?: string;
}

export default function HistoryDetailPage() {
    const params = useParams();
    const router = useRouter();
    const [item, setItem] = useState<HistoryItem | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // In a real app, we might fetch a single item by ID.
        // Here we fetch all and find the one we need, or we could implement /history/{id} in backend.
        // For simplicity, let's fetch all and filter.
        fetch('http://localhost:8000/history')
            .then(res => res.json())
            .then((data: HistoryItem[]) => {
                const found = data.find(i => i.id === Number(params.id));
                setItem(found || null);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setLoading(false);
            });
    }, [params.id]);

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
            </div>
        );
    }

    if (!item) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center gap-4">
                <p className="text-slate-400">Scan not found.</p>
                <button onClick={() => router.back()} className="text-cyan-400 hover:underline">
                    Go Back
                </button>
            </div>
        );
    }

    return (
        <main className="min-h-screen p-8 pb-24">
            <div className="max-w-4xl mx-auto">
                <button
                    onClick={() => router.back()}
                    className="flex items-center gap-2 text-slate-400 hover:text-white mb-8 transition-colors"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back to History
                </button>

                <ResultDisplay
                    result={item}
                    onReset={() => router.push('/')}
                    initialImageUrl={item.image_url}
                />
            </div>
        </main>
    );
}
