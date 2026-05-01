import { uploadFile, get } from "./api";
import type { EnhanceResult, ModelVersion } from "@/types/api";

export async function enhance(
  filePath: string,
  modelVersion: ModelVersion,
  clientId: string
): Promise<EnhanceResult> {
  const resp = await uploadFile<EnhanceResult>("/inference/enhance/", filePath, {
    model_version: modelVersion,
    client_id: clientId,
  });

  if (resp.code !== 0 || !resp.data) {
    throw Object.assign(new Error(resp.message), { code: resp.code });
  }
  return resp.data;
}

export async function fetchHealth(): Promise<{
  models_loaded: string[];
  default_version: string;
}> {
  const resp = await get<{ models_loaded: string[]; default_version: string }>("/health/");
  if (resp.code !== 0 || !resp.data) throw new Error(resp.message);
  return resp.data;
}
