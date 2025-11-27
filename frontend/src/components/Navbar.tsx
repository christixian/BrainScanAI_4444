"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Brain, History, Home } from "lucide-react";
import { clsx } from "clsx";

export default function Navbar() {
    const pathname = usePathname();

    const navItems = [
        { name: "Home", href: "/", icon: Home },
        { name: "History", href: "/history", icon: History },
    ];

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/10 bg-[#0f172a]/80 backdrop-blur-md">
            <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-2 group">
                        <div className="p-2 rounded-lg bg-cyan-500/10 group-hover:bg-cyan-500/20 transition-colors">
                            <Brain className="w-6 h-6 text-cyan-400" />
                        </div>
                        <span className="text-lg font-bold text-white tracking-tight">
                            BrainScan <span className="text-cyan-400">AI</span>
                        </span>
                    </Link>

                    {/* Navigation Links */}
                    <div className="flex items-center gap-1">
                        {navItems.map((item) => {
                            const isActive = pathname === item.href;
                            return (
                                <Link
                                    key={item.name}
                                    href={item.href}
                                    className={clsx(
                                        "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-200",
                                        isActive
                                            ? "bg-white/10 text-white shadow-[0_0_10px_rgba(255,255,255,0.1)]"
                                            : "text-slate-400 hover:text-white hover:bg-white/5"
                                    )}
                                >
                                    <item.icon className="w-4 h-4" />
                                    {item.name}
                                </Link>
                            );
                        })}
                    </div>
                </div>
            </div>
        </nav>
    );
}
