import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Clean up after each test
afterEach(() => {
  cleanup();
});

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn() }),
  usePathname: () => "/dashboard",
  useSearchParams: () => new URLSearchParams(),
  redirect: vi.fn(),
}));

// Mock next/font/google
vi.mock("next/font/google", () => ({
  Syne: () => ({ variable: "--font-syne", className: "syne" }),
  DM_Sans: () => ({ variable: "--font-dm-sans", className: "dm-sans" }),
  JetBrains_Mono: () => ({ variable: "--font-jetbrains", className: "jetbrains" }),
}));

// Mock framer-motion to avoid animation issues in tests
vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) =>
      <div {...props}>{children}</div>,
    span: ({ children, ...props }: React.HTMLAttributes<HTMLSpanElement>) =>
      <span {...props}>{children}</span>,
    p: ({ children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) =>
      <p {...props}>{children}</p>,
    button: ({ children, onClick, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) =>
      <button onClick={onClick} {...props}>{children}</button>,
    circle: ({ children, ...props }: React.SVGProps<SVGCircleElement>) =>
      <circle {...props}>{children}</circle>,
    line: (props: React.SVGProps<SVGLineElement>) => <line {...props} />,
    path: (props: React.SVGProps<SVGPathElement>) => <path {...props} />,
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useAnimation: () => ({ start: vi.fn(), stop: vi.fn() }),
}));

// Mock WebSocket
global.WebSocket = vi.fn().mockImplementation(() => ({
  send: vi.fn(),
  close: vi.fn(),
  readyState: 1,
  onopen: null,
  onclose: null,
  onmessage: null,
  onerror: null,
})) as unknown as typeof WebSocket;

// Suppress console errors from intentional test failures
const originalError = console.error;
console.error = (...args: unknown[]) => {
  if (typeof args[0] === "string" && args[0].includes("Warning:")) return;
  originalError(...args);
};
