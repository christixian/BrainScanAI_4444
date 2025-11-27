import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "BrainScan AI | MRI Tumor Classification",
  description: "Research-focused deep learning tool for brain tumor classification.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} antialiased min-h-screen bg-[#0f172a] text-slate-50 selection:bg-cyan-500/30`}>
        <div className="fixed inset-0 z-[-1] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-cyan-900/20 via-slate-900 to-slate-950 pointer-events-none" />
        <Navbar />
        <div className="pt-20">
          {children}
        </div>
      </body>
    </html>
  );
}
