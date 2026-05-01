import type { ApiResponse, AestheticScore } from "@/types/api";

const BASE_URL = "http://127.0.0.1:8002/api/v1";

function request<T>(options: UniApp.RequestOptions): Promise<ApiResponse<T>> {
  return new Promise((resolve, reject) => {
    uni.request({
      ...options,
      url: `${BASE_URL}${options.url}`,
      success: (res) => resolve(res.data as ApiResponse<T>),
      fail: reject,
    });
  });
}

export function uploadFile<T>(
  url: string,
  filePath: string,
  formData: Record<string, string>
): Promise<ApiResponse<T>> {
  return new Promise((resolve, reject) => {
    uni.uploadFile({
      url: `${BASE_URL}${url}`,
      filePath,
      name: "file",
      formData,
      success: (res) => {
        try {
          const data = JSON.parse(res.data) as ApiResponse<T>;
          resolve(data);
        } catch {
          reject(new Error("Invalid response JSON"));
        }
      },
      fail: (err) => { console.error('[uploadFile:fail]', JSON.stringify(err)); reject(err); },
    });
  });
}

export function get<T>(url: string, params?: Record<string, string>) {
  const query = params ? "?" + new URLSearchParams(params).toString() : "";
  return request<T>({ url: `${url}${query}`, method: "GET" });
}

export interface AestheticPollResult {
  request_id: string;
  aesthetic_status: "pending" | "running" | "done" | "failed" | "unavailable";
  aesthetic_before: AestheticScore | null;
  aesthetic_after: AestheticScore | null;
}

export function getAestheticScore(requestId: string) {
  return get<AestheticPollResult>(`/inference/aesthetic/${requestId}/`);
}
