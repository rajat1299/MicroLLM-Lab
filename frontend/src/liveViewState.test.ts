import { describe, expect, it } from "vitest";

import { nextFrameIndexAfterAppend, shouldAcceptEventSequence } from "./liveViewState";

describe("shouldAcceptEventSequence", () => {
  it("accepts strictly increasing sequences", () => {
    expect(shouldAcceptEventSequence(5, 6)).toBe(true);
  });

  it("rejects duplicate and out-of-order sequences", () => {
    expect(shouldAcceptEventSequence(5, 5)).toBe(false);
    expect(shouldAcceptEventSequence(5, 4)).toBe(false);
  });

  it("rejects invalid sequence numbers", () => {
    expect(shouldAcceptEventSequence(5, Number.NaN)).toBe(false);
  });
});

describe("nextFrameIndexAfterAppend", () => {
  it("follows latest frame when live follow is enabled", () => {
    expect(nextFrameIndexAfterAppend(3, true, 0)).toBe(3);
  });

  it("keeps manual selection when live follow is paused", () => {
    expect(nextFrameIndexAfterAppend(3, false, 1)).toBe(1);
  });

  it("clamps manual selection to the latest available frame", () => {
    expect(nextFrameIndexAfterAppend(2, false, 10)).toBe(2);
  });
});
