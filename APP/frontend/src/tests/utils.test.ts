import { describe, it, expect, vi, beforeEach } from "vitest";
import { getErrorMessage } from "@/utils/error";
import { generateClientId, getClientId } from "@/utils/image";

// ─── error.ts ────────────────────────────────────────────────────────────────

describe("getErrorMessage", () => {
  it("returns mapped message for known code", () => {
    expect(getErrorMessage(1001)).toBe("请选择要上传的图片");
    expect(getErrorMessage(2001)).toBe("模型推理失败，请重试");
    expect(getErrorMessage(3001)).toBe("服务器存储异常，请稍后重试");
  });

  it("returns fallback string for unknown code", () => {
    expect(getErrorMessage(9999, "custom fallback")).toBe("custom fallback");
  });

  it("returns default message when no fallback provided", () => {
    expect(getErrorMessage(9999)).toBe("未知错误，请重试");
  });
});

// ─── image.ts ────────────────────────────────────────────────────────────────

describe("generateClientId", () => {
  it("starts with mv_", () => {
    expect(generateClientId()).toMatch(/^mv_/);
  });

  it("generates unique ids", () => {
    const ids = new Set(Array.from({ length: 20 }, generateClientId));
    expect(ids.size).toBe(20);
  });
});

describe("getClientId", () => {
  beforeEach(() => {
    vi.mocked(uni.getStorageSync).mockReturnValue("");
    vi.mocked(uni.setStorageSync).mockClear();
  });

  it("creates and stores new id when storage is empty", () => {
    const id = getClientId();
    expect(id).toMatch(/^mv_/);
    expect(uni.setStorageSync).toHaveBeenCalledOnce();
  });

  it("returns cached id on second call", () => {
    const newId = generateClientId();
    vi.mocked(uni.getStorageSync).mockReturnValue(newId);
    const id = getClientId();
    expect(id).toBe(newId);
    expect(uni.setStorageSync).not.toHaveBeenCalled();
  });
});
