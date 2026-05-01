const CLIENT_ID_KEY = "mv_client_id";

export function generateClientId(): string {
  const ts = Date.now().toString(36);
  const rand = Math.random().toString(36).slice(2, 8);
  return `mv_${ts}_${rand}`;
}

export function getClientId(): string {
  let id = uni.getStorageSync(CLIENT_ID_KEY) as string;
  if (!id) {
    id = generateClientId();
    uni.setStorageSync(CLIENT_ID_KEY, id);
  }
  return id;
}

export function chooseImage(): Promise<string> {
  return new Promise((resolve, reject) => {
    uni.chooseImage({
      count: 1,
      sizeType: ["original", "compressed"],
      sourceType: ["album", "camera"],
      success: (res) => resolve(res.tempFilePaths[0]),
      fail: (err) => reject(err),
    });
  });
}

export function downloadToTemp(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    uni.downloadFile({
      url,
      success: (res) => {
        if (res.statusCode === 200) resolve(res.tempFilePath);
        else reject(new Error(`Download failed: ${res.statusCode}`));
      },
      fail: reject,
    });
  });
}
