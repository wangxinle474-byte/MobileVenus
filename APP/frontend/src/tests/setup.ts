import { createPinia, setActivePinia } from "pinia";
import { beforeEach, vi } from "vitest";

beforeEach(() => {
  setActivePinia(createPinia());
});

const uniMock = {
  getStorageSync: vi.fn(() => ""),
  setStorageSync: vi.fn(),
  showToast: vi.fn(),
  showLoading: vi.fn(),
  hideLoading: vi.fn(),
  showModal: vi.fn(),
  openSetting: vi.fn(),
  authorize: vi.fn(),
  chooseImage: vi.fn(),
  uploadFile: vi.fn(),
  request: vi.fn(),
  downloadFile: vi.fn(),
  saveImageToPhotosAlbum: vi.fn(),
  navigateTo: vi.fn(),
  navigateBack: vi.fn(),
  getSystemInfoSync: vi.fn(() => ({ windowWidth: 375 })),
};

(globalThis as unknown as Record<string, unknown>).uni = uniMock;
