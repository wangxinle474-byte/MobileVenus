export function showToast(title: string, icon: "success" | "error" | "none" = "none") {
  uni.showToast({ title, icon, duration: 2000 });
}

export function showLoading(title = "处理中…") {
  uni.showLoading({ title, mask: true });
}

export function hideLoading() {
  uni.hideLoading();
}
