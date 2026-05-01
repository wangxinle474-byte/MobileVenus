export function requestSaveAlbum(): Promise<void> {
  return new Promise((resolve, reject) => {
    uni.authorize({
      scope: "scope.writePhotosAlbum",
      success: () => resolve(),
      fail: () => {
        uni.showModal({
          title: "需要相册权限",
          content: "请前往设置允许访问相册",
          confirmText: "去设置",
          success: (res) => {
            if (res.confirm) uni.openSetting({});
          },
        });
        reject(new Error("album permission denied"));
      },
    });
  });
}

export function saveImageToAlbum(filePath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    uni.saveImageToPhotosAlbum({
      filePath,
      success: () => resolve(),
      fail: reject,
    });
  });
}
