export type PageStatus = "idle" | "uploading" | "processing" | "success" | "error";

export interface ModelOption {
  version: import("./api").ModelVersion;
  label: string;
  tag: string;
  psnr: string;
  ssim: string;
}

export const MODEL_OPTIONS: ModelOption[] = [
  {
    version: "baseline",
    label: "Baseline",
    tag: "对照基准",
    psnr: "32.05 dB",
    ssim: "0.9269",
  },
  {
    version: "distill_v2",
    label: "Distill v2",
    tag: "精度最优",
    psnr: "33.15 dB",
    ssim: "0.9311",
  },
  {
    version: "distill_v4",
    label: "Distill v4",
    tag: "感知最优",
    psnr: "33.11 dB",
    ssim: "0.9326",
  },
];

export const PARAM_LABELS: Record<string, string> = {
  ev_compensation: "曝光补偿",
  white_balance: "白平衡",
  contrast: "对比度",
  brightness: "亮度",
  shadows: "阴影",
  highlights: "高光",
  saturation: "饱和度",
  vibrance: "自然饱和度",
};

export const CONFIDENCE_LABEL: Record<string, string> = {
  high: "高置信",
  medium: "中置信",
  reference_only: "仅参考",
};
