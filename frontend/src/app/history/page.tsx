"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, Calendar, Clock, ArrowRight, Image as ImageIcon, Trash2 } from "lucide-react";
import { clsx } from "clsx";

interface HistoryItem {
    id: number;
    timestamp: string;
    prediction_4class: string;
    prediction_binary: "healthy" | "unhealthy";
    confidence_scores: Record<string, number>;
    binary_confidence: number;
    image_url?: string;
}

export default function HistoryPage() {
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [showConfirmModal, setShowConfirmModal] = useState(false);

    useEffect(() => {
        fetch('http://localhost:8000/history')
            .then(res => res.json())
            .then(data => setHistory(data))
            .catch(err => console.error('Failed to load history:', err));
    }, []);

    const formatDate = (isoString: string) => {
        const date = new Date(isoString);
        return {
            date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
            time: date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
        };
    };

    const handleClearClick = () => {
        setShowConfirmModal(true);
    };

    const confirmClearHistory = async () => {
        try {
            const response = await fetch('http://localhost:8000/history', {
                method: 'DELETE',
            });

            if (response.ok) {
                setHistory([]);
                setShowConfirmModal(false);
            } else {
                console.error('Failed to clear history');
            }
        } catch (err) {
            console.error('Error clearing history:', err);
        }
    };

    return (
        <main className="min-h-screen p-8 pb-24">
            <div className="max-w-4xl mx-auto space-y-8 animate-fade-in">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-white mb-2">Scan History</h1>
                        <p className="text-slate-400">Review past analyses and medical insights</p>
                    </div>
                    {history.length > 0 && (
                        <button
                            onClick={handleClearClick}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20 hover:bg-rose-500/20 transition-colors"
                        >
                            <Trash2 className="w-4 h-4" />
                            Clear History
                        </button>
                    )}
                </div>

                <div className="grid gap-4">
                    {history.length === 0 ? (
                        <div className="text-center py-20 rounded-2xl border border-dashed border-slate-700 bg-slate-800/30">
                            <Activity className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                            <p className="text-slate-400">No scan history available yet.</p>
                        </div>
                    ) : (
                        history.map((item) => {
                            const { date, time } = formatDate(item.timestamp);
                            const isHealthy = item.prediction_binary === "healthy";
                            const isUncertain = item.prediction_4class.toLowerCase() === "uncertain"

                            return (
                                <Link
                                    href={`/history/${item.id}`}
                                    key={item.id}
                                    className="group block"
                                >
                                    <div className="relative overflow-hidden rounded-xl bg-slate-800/50 border border-slate-700 hover:border-cyan-500/50 transition-all hover:shadow-lg hover:shadow-cyan-900/20 p-4">
                                        <div className="flex items-center gap-6">
                                            {/* Image Thumbnail */}
                                            <div className="relative w-24 h-24 shrink-0 rounded-lg overflow-hidden bg-black border border-slate-700">
                                                {item.image_url ? (
                                                    <img
                                                        src={item.image_url}
                                                        alt="Scan"
                                                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                                                    />
                                                ) : (
                                                    <div className="w-full h-full flex items-center justify-center text-slate-600">
                                                        <ImageIcon className="w-8 h-8" />
                                                    </div>
                                                )}
                                            </div>

                                            {/* Info */}
                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center gap-3 mb-2">
                                                    <span className={clsx(
                                                        "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider",
                                                        isUncertain
                                                            ? "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"
                                                            :
                                                        isHealthy
                                                            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                                            : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                                                    )}>
                                                        {isHealthy? "No Tumor" : item.prediction_4class}
                                                    </span>
                                                    {!isUncertain ? (
                                                    <span className="text-xs font-mono text-slate-500">
                                                        Conf of {isHealthy? "no tumor" : "tumor"}: {(item.binary_confidence * 100).toFixed(1)}%
                                                    </span>
                                                    ) : null}
                                                </div>

                                                <div className="flex items-center gap-4 text-sm text-slate-400">
                                                    <div className="flex items-center gap-1.5">
                                                        <Calendar className="w-4 h-4 text-slate-500" />
                                                        {date}
                                                    </div>
                                                    <div className="flex items-center gap-1.5">
                                                        <Clock className="w-4 h-4 text-slate-500" />
                                                        {time}
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Arrow */}
                                            <div className="pr-4">
                                                <div className="p-2 rounded-full bg-slate-700/50 group-hover:bg-cyan-500/20 group-hover:text-cyan-400 transition-colors">
                                                    <ArrowRight className="w-5 h-5" />
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </Link>
                            );
                        })
                    )}
                </div>
            </div>

            {/* Custom Confirmation Modal */}
            {showConfirmModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
                    <div className="w-full max-w-md bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl p-6 space-y-6">
                        <div className="space-y-2">
                            <h3 className="text-xl font-bold text-white">Clear Scan History?</h3>
                            <p className="text-slate-400">
                                Are you sure you want to delete all past scan records? This action cannot be undone.
                            </p>
                        </div>

                        <div className="flex items-center justify-end gap-3">
                            <button
                                onClick={() => setShowConfirmModal(false)}
                                className="px-4 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition-colors font-medium"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={confirmClearHistory}
                                className="px-4 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white shadow-lg shadow-rose-500/20 transition-all font-medium"
                            >
                                Yes, Clear History
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </main>
    );
}
