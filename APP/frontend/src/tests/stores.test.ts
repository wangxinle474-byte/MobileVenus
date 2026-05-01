import { describe, it, expect } from "vitest";
import { useModelStore } from "@/stores/modelStore";
import { useResultStore } from "@/stores/resultStore";
import { useAppStore } from "@/stores/app";
import type { EnhanceResult } from "@/types/api";

// ─── modelStore ───────────────────────────────────────────────────────────────

describe("modelStore", () => {
  it("defaults to distill_v4", () => {
    const store = useModelStore();
    expect(store.selectedVersion).toBe("distill_v4");
  });

  it("select() changes selectedVersion", () => {
    const store = useModelStore();
    store.select("baseline");
    expect(store.selectedVersion).toBe("baseline");
  });

  it("select() accepts all valid versions", () => {
    const store = useModelStore();
    for (const v of ["baseline", "distill_v2", "distill_v4"] as const) {
      store.select(v);
      expect(store.selectedVersion).toBe(v);
    }
  });
});

// ─── resultStore ─────────────────────────────────────────────────────────────

const MOCK_RESULT: EnhanceResult = {
  request_id: "req-001",
  original_image_url: "http://x/orig.jpg",
  enhanced_image_url: "http://x/enh.jpg",
  parameters: {
    ev_compensation: 0.5,
    contrast: 1.1,
    highlights: -0.2,
    shadows: 0.3,
    brightness: 0.1,
    vibrance: -0.1,
    saturation: 1.2,
    white_balance: 5500,
  },
  parameter_confidence: {
    ev_compensation: "high",
    contrast: "medium",
    highlights: "high",
    shadows: "medium",
    brightness: "reference_only",
    vibrance: "reference_only",
    saturation: "high",
    white_balance: "medium",
  },
  meta_info: {
    model_name: "DistillV4",
    backbone: "MobileViT-S",
    pipeline: "SemanticDistill",
    inference_time_ms: 38,
    image_width: 1920,
    image_height: 1080,
    psnr_reference: "27.6 dB",
  },
  aesthetic_status: "done",
  aesthetic_before: { overall: 6.5, dimensions: { composition: 6.0, lighting: 7.0, color: 6.5, clarity: 6.8, subject: 6.2 } },
  aesthetic_after: { overall: 7.8, dimensions: { composition: 7.5, lighting: 8.0, color: 7.8, clarity: 7.6, subject: 7.9 } },
};

describe("resultStore", () => {
  it("starts with null result", () => {
    const store = useResultStore();
    expect(store.result).toBeNull();
    expect(store.requestId).toBeNull();
  });

  it("setResult() stores data correctly", () => {
    const store = useResultStore();
    store.setResult(MOCK_RESULT);
    expect(store.result).toEqual(MOCK_RESULT);
    expect(store.requestId).toBe("req-001");
  });

  it("clear() resets to null", () => {
    const store = useResultStore();
    store.setResult(MOCK_RESULT);
    store.clear();
    expect(store.result).toBeNull();
    expect(store.requestId).toBeNull();
  });
});

// ─── appStore ────────────────────────────────────────────────────────────────

describe("appStore", () => {
  it("starts with idle status", () => {
    const store = useAppStore();
    expect(store.status).toBe("idle");
    expect(store.errorMessage).toBeNull();
  });

  it("setStatus() updates status", () => {
    const store = useAppStore();
    store.setStatus("processing");
    expect(store.status).toBe("processing");
  });

  it("setError() sets status=error and message", () => {
    const store = useAppStore();
    store.setError("something went wrong");
    expect(store.status).toBe("error");
    expect(store.errorMessage).toBe("something went wrong");
  });

  it("reset() restores idle state", () => {
    const store = useAppStore();
    store.setError("oops");
    store.reset();
    expect(store.status).toBe("idle");
    expect(store.errorMessage).toBeNull();
  });

  it("init() sets a clientId", () => {
    const store = useAppStore();
    store.init();
    expect(store.clientId).toMatch(/^mv_/);
  });
});
