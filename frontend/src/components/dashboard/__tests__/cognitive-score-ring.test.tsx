import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { CognitiveScoreRing } from "../cognitive-score-ring";

const defaultProps = {
  score: 75,
  fatigueScore: 20,
  stressScore: 30,
  productivityScore: 80,
  isLive: false,
};

describe("CognitiveScoreRing", () => {
  it("renders without crashing", () => {
    const { container } = render(<CognitiveScoreRing {...defaultProps} />);
    expect(container).toBeTruthy();
  });

  it("displays the score as a rounded number", () => {
    render(<CognitiveScoreRing {...defaultProps} score={73.7} />);
    expect(screen.getByText("74")).toBeTruthy();
  });

  it("shows Optimal label at score 85", () => {
    render(<CognitiveScoreRing {...defaultProps} score={85} />);
    expect(screen.getByText("OPTIMAL")).toBeTruthy();
  });

  it("shows Focused label at score 70", () => {
    render(<CognitiveScoreRing {...defaultProps} score={70} />);
    expect(screen.getByText("FOCUSED")).toBeTruthy();
  });

  it("shows Declining label at score 50", () => {
    render(<CognitiveScoreRing {...defaultProps} score={50} />);
    expect(screen.getByText("DECLINING")).toBeTruthy();
  });

  it("shows Fatigued label at score 30", () => {
    render(<CognitiveScoreRing {...defaultProps} score={30} />);
    expect(screen.getByText("FATIGUED")).toBeTruthy();
  });

  it("shows Critical label at score 10", () => {
    render(<CognitiveScoreRing {...defaultProps} score={10} />);
    expect(screen.getByText("CRITICAL")).toBeTruthy();
  });

  it("shows LIVE indicator when isLive is true", () => {
    render(<CognitiveScoreRing {...defaultProps} isLive={true} />);
    expect(screen.getByText("LIVE")).toBeTruthy();
  });

  it("does not show LIVE indicator when isLive is false", () => {
    render(<CognitiveScoreRing {...defaultProps} isLive={false} />);
    expect(screen.queryByText("LIVE")).toBeNull();
  });

  it("renders Cognitive Score label", () => {
    render(<CognitiveScoreRing {...defaultProps} />);
    expect(screen.getByText("Cognitive Score")).toBeTruthy();
  });

  it("renders an SVG element", () => {
    const { container } = render(<CognitiveScoreRing {...defaultProps} />);
    const svg = container.querySelector("svg");
    expect(svg).toBeTruthy();
  });

  it("renders three inner rings for sub-metrics", () => {
    const { container } = render(<CognitiveScoreRing {...defaultProps} />);
    const circles = container.querySelectorAll("circle");
    // Background + 3 colored rings + needle center = multiple circles
    expect(circles.length).toBeGreaterThan(3);
  });
});
