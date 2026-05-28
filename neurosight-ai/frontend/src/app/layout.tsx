import type { Metadata, Viewport } from "next";
import { Syne, DM_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/providers";

const syne = Syne({
  subsets: ["latin"],
  variable: "--font-syne",
  display: "swap",
  weight: ["400", "500", "600", "700", "800"],
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
  display: "swap",
  weight: ["300", "400", "500", "600"],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: {
    default: "NeuroSight AI",
    template: "%s — NeuroSight AI",
  },
  description:
    "AI-powered cognitive fatigue and productivity intelligence platform. Real-time fatigue detection, stress analysis, and adaptive recommendations.",
  keywords: [
    "fatigue detection",
    "productivity",
    "AI",
    "cognitive analytics",
    "computer vision",
    "burnout prevention",
  ],
  authors: [{ name: "NeuroSight AI" }],
  robots: "index, follow",
  openGraph: {
    type: "website",
    locale: "en_US",
    title: "NeuroSight AI",
    description: "AI-powered cognitive fatigue & productivity intelligence platform",
    siteName: "NeuroSight AI",
  },
};

export const viewport: Viewport = {
  themeColor: "#050810",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${syne.variable} ${dmSans.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      </head>
      <body className="antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
