export interface ApiResponse<T = unknown> {
  code: number;
  message: string;
  data: T | null;
}

export type ModelVersion = "baseline" | "distill_v2" | "distill_v4";

export interface EnhanceRequest {
  file: File;
  model_version?: ModelVersion;
  client_id?: string;
}

export interface ParameterConfidence {
  ev_compensation: string;
  white_balance: string;
  contrast: string;
  brightness: string;
  shadows: string;
  highlights: string;
  saturation: string;
  vibrance: string;
}

export interface InferenceParams {
  ev_compensation: number;
  white_balance: number;
  contrast: number;
  brightness: number;
  shadows: number;
  highlights: number;
  saturation: number;
  vibrance: number;
}

export interface MetaInfo {
  model_name: string;
  backbone: string;
  pipeline: string;
  inference_time_ms: number;
  image_width: number;
  image_height: number;
  psnr_reference: string;
}

export interface AestheticDimensions {
  [key: string]: number;
  composition: number;
  lighting: number;
  color: number;
  clarity: number;
  subject: number;
}

export interface AestheticScore {
  overall: number;
  dimensions: AestheticDimensions;
}

export interface EnhanceResult {
  request_id: string;
  original_image_url: string;
  enhanced_image_url: string;
  parameters: InferenceParams;
  parameter_confidence: ParameterConfidence;
  meta_info: MetaInfo;
  aesthetic_status: "pending" | "running" | "done" | "failed" | "unavailable";
  aesthetic_before: AestheticScore | null;
  aesthetic_after: AestheticScore | null;
}
