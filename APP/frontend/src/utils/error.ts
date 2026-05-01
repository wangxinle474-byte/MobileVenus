const ERROR_MESSAGES: Record<number, string> = {
  1001: "请选择要上传的图片",
  1002: "仅支持 JPEG / PNG 格式",
  1003: "图片大小不能超过 10MB",
  1004: "图片文件损坏，请重新选择",
  1005: "图片尺寸过小，请选择更清晰的图片",
  2001: "模型推理失败，请重试",
  2002: "图像渲染失败，请重试",
  3001: "服务器存储异常，请稍后重试",
};

export function getErrorMessage(code: number, fallback?: string): string {
  return ERROR_MESSAGES[code] ?? fallback ?? "未知错误，请重试";
}
