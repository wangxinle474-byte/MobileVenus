<script setup lang="ts">
import { computed, onMounted, onUnmounted } from "vue";
import { useResultStore } from "@/stores/resultStore";
import { downloadToTemp } from "@/utils/image";
import { saveImageToAlbum, requestSaveAlbum } from "@/utils/permission";
import { showToast } from "@/utils/toast";
import CompareSlider from "@/components/CompareSlider/index.vue";
import ParamPanel from "@/components/ParamPanel/index.vue";
import type { AestheticScore } from "@/types/api";
import { getAestheticScore } from "@/services/api";

const POLL_INTERVAL = 6000;
const TERMINAL_STATUSES = new Set(["done", "failed", "unavailable"]);

const resultStore = useResultStore();
const result = computed(() => resultStore.result!);

const aestheticDims = [
  { key: "composition", label: "构图" },
  { key: "lighting",    label: "光线" },
  { key: "color",       label: "色彩" },
  { key: "clarity",     label: "清晰度" },
  { key: "subject",     label: "主体" },
] as const;

function formatScore(a: AestheticScore | null, key: string): string {
  if (!a) return "—";
  const v = (a.dimensions as Record<string, number>)[key];
  return v != null ? v.toFixed(1) : "—";
}

function scoreDelta(
  before: AestheticScore | null,
  after: AestheticScore | null,
  key: string
): number | null {
  if (!before || !after) return null;
  const b = (before.dimensions as Record<string, number>)[key];
  const a = (after.dimensions as Record<string, number>)[key];
  if (b == null || a == null) return null;
  return a - b;
}

let _pollTimer: ReturnType<typeof setInterval> | null = null;

function stopPolling() {
  if (_pollTimer !== null) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
}

async function pollAesthetic() {
  const r = result.value;
  if (!r || TERMINAL_STATUSES.has(r.aesthetic_status)) {
    stopPolling();
    return;
  }
  try {
    const res = await getAestheticScore(r.request_id);
    if (res.code === 0 && res.data) {
      resultStore.patchAesthetic(res.data);
      if (TERMINAL_STATUSES.has(res.data.aesthetic_status)) {
        stopPolling();
      }
    }
  } catch {
    // 网络失败时保持轮询，直到超时
  }
}

onMounted(() => {
  const status = result.value?.aesthetic_status;
  if (status && !TERMINAL_STATUSES.has(status)) {
    _pollTimer = setInterval(pollAesthetic, POLL_INTERVAL);
  }
});

onUnmounted(() => stopPolling());

async function onSave() {
  try {
    await requestSaveAlbum();
    const tempPath = await downloadToTemp(result.value.enhanced_image_url);
    await saveImageToAlbum(tempPath);
    showToast("已保存到相册", "success");
  } catch {
    showToast("保存失败，请重试");
  }
}

function onBack() {
  uni.navigateBack();
}
</script>

<template>
  <scroll-view class="page" scroll-y>
    <template v-if="result">
      <!-- 模型信息条 -->
      <view class="meta-bar">
        <text class="model-name">{{ result.meta_info.model_name }}</text>
        <view class="meta-pills">
          <text class="pill">{{ result.meta_info.psnr_reference }}</text>
          <text class="pill">{{ result.meta_info.inference_time_ms }} ms</text>
        </view>
      </view>

      <!-- 对比滑块 -->
      <view class="section">
        <compare-slider
          :original-url="result.original_image_url"
          :enhanced-url="result.enhanced_image_url"
        />
      </view>

      <!-- 参数面板 -->
      <view class="section">
        <param-panel
          :params="result.parameters"
          :confidence="result.parameter_confidence"
        />
      </view>

      <!-- 美学评分 -->
      <view
        v-if="result.aesthetic_status !== 'pending'"
        class="section aesthetic-card"
      >
        <text class="card-title">Venus 美学评分</text>
        <view v-if="result.aesthetic_status === 'running'" class="aesthetic-loading">
          <text class="loading-dot">● </text>
          <text class="loading-text">Venus 正在评分，请稍候…</text>
        </view>
        <view v-if="result.aesthetic_status === 'failed' || result.aesthetic_status === 'unavailable'" class="aesthetic-loading">
          <text class="loading-text">评分暂不可用（Venus 模型未能运行）</text>
        </view>
        <template v-if="result.aesthetic_status === 'done'">
          <view class="aesthetic-header">
            <text class="col-label" />
            <text class="col-label">原图</text>
            <text class="col-label highlight">增强后</text>
          </view>
          <view
            v-for="dim in aestheticDims"
            :key="dim.key"
            class="aesthetic-row"
          >
            <text class="dim-name">{{ dim.label }}</text>
            <text class="dim-score">{{ formatScore(result.aesthetic_before, dim.key) }}</text>
            <view class="dim-after">
              <text class="dim-score highlight">{{ formatScore(result.aesthetic_after, dim.key) }}</text>
              <text
                v-if="scoreDelta(result.aesthetic_before, result.aesthetic_after, dim.key) !== null"
                :class="['delta', scoreDelta(result.aesthetic_before, result.aesthetic_after, dim.key)! > 0 ? 'pos' : 'neg']"
              >{{ scoreDelta(result.aesthetic_before, result.aesthetic_after, dim.key)! > 0 ? '+' : '' }}{{ scoreDelta(result.aesthetic_before, result.aesthetic_after, dim.key)!.toFixed(1) }}</text>
            </view>
          </view>
          <view class="aesthetic-row overall">
            <text class="dim-name">综合</text>
            <text class="dim-score">{{ result.aesthetic_before ? result.aesthetic_before.overall.toFixed(1) : '—' }}</text>
            <view class="dim-after">
              <text class="dim-score highlight">{{ result.aesthetic_after ? result.aesthetic_after.overall.toFixed(1) : '—' }}</text>
              <text
                v-if="result.aesthetic_before && result.aesthetic_after"
                :class="['delta', result.aesthetic_after.overall - result.aesthetic_before.overall > 0 ? 'pos' : 'neg']"
              >{{ result.aesthetic_after.overall - result.aesthetic_before.overall > 0 ? '+' : '' }}{{ (result.aesthetic_after.overall - result.aesthetic_before.overall).toFixed(1) }}</text>
            </view>
          </view>
        </template>
      </view>

      <!-- 模型元信息 -->
      <view class="section meta-card">
        <view class="meta-row">
          <text class="meta-key">推理骨干</text>
          <text class="meta-val">{{ result.meta_info.backbone }}</text>
        </view>
        <view class="meta-row">
          <text class="meta-key">渲染管线</text>
          <text class="meta-val">{{ result.meta_info.pipeline }}</text>
        </view>
        <view class="meta-row">
          <text class="meta-key">图像分辨率</text>
          <text class="meta-val">
            {{ result.meta_info.image_width }} × {{ result.meta_info.image_height }}
          </text>
        </view>
      </view>

      <!-- 操作按钮 -->
      <view class="actions">
        <view class="btn-save" @tap="onSave">保存增强图</view>
        <view class="btn-back" @tap="onBack">返回重试</view>
      </view>
    </template>
  </scroll-view>
</template>

<style lang="scss" scoped>
.page {
  width: 100%;
  box-sizing: border-box;
  overflow-x: hidden;
  min-height: 100vh;
  background: #f5f6fa;
  padding: 24rpx 32rpx 80rpx;
}

.meta-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24rpx;

  .model-name {
    font-size: 30rpx;
    font-weight: 600;
    color: #1a1a2e;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-right: 16rpx;
  }

  .meta-pills {
    display: flex;
    gap: 12rpx;
    flex-shrink: 0;
  }

  .pill {
    font-size: 22rpx;
    color: #6c63ff;
    background: rgba(108, 99, 255, 0.12);
    padding: 4rpx 14rpx;
    border-radius: 20rpx;
    white-space: nowrap;
  }
}

.section {
  margin-bottom: 28rpx;
}

.meta-card {
  background: #ffffff;
  border-radius: 20rpx;
  padding: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0,0,0,0.06);

  .meta-row {
    display: flex;
    justify-content: space-between;
    padding: 10rpx 0;
    border-bottom: 1rpx solid #e8e8f0;

    &:last-child { border-bottom: none; }
  }

  .meta-key {
    font-size: 26rpx;
    color: #6a6a8a;
  }

  .meta-val {
    font-size: 26rpx;
    color: #1a1a2e;
  }
}

.aesthetic-card {
  background: #ffffff;
  border-radius: 20rpx;
  padding: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0,0,0,0.06);

  .card-title {
    font-size: 26rpx;
    font-weight: 600;
    color: #6c63ff;
    display: block;
    margin-bottom: 16rpx;
  }

  .aesthetic-loading {
    display: flex;
    align-items: center;
    gap: 8rpx;
    padding: 12rpx 0 8rpx;

    .loading-dot {
      color: #a78bfa;
      font-size: 20rpx;
      animation: pulse 1.4s infinite;
    }

    .loading-text {
      font-size: 24rpx;
      color: #9090a8;
    }
  }

  .aesthetic-header {
    display: flex;
    margin-bottom: 8rpx;

    .col-label {
      font-size: 22rpx;
      color: #9090a8;
      flex: 1;
      text-align: center;

      &:first-child { text-align: left; flex: 1.4; }
      &.highlight { color: #a78bfa; }
    }
  }

  .aesthetic-row {
    display: flex;
    align-items: center;
    padding: 10rpx 0;
    border-bottom: 1rpx solid #e8e8f0;

    &:last-child { border-bottom: none; }
    &.overall { margin-top: 6rpx; border-top: 1rpx solid #d0d0e8; border-bottom: none; }

    .dim-name {
      flex: 1.4;
      font-size: 26rpx;
      color: #6a6a8a;
    }

    .dim-score {
      flex: 1;
      text-align: center;
      font-size: 26rpx;
      color: #1a1a2e;
      font-variant-numeric: tabular-nums;

      &.highlight { color: #a78bfa; font-weight: 600; }
    }

    .dim-after {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6rpx;
    }

    .delta {
      font-size: 20rpx;
      padding: 2rpx 8rpx;
      border-radius: 8rpx;

      &.pos { color: #00d4aa; background: rgba(0, 212, 170, 0.12); }
      &.neg { color: #ff6b6b; background: rgba(255, 107, 107, 0.12); }
    }
  }
}

.actions {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
  margin-top: 8rpx;
}

.btn-save {
  background: linear-gradient(135deg, #00d4aa, #00b894);
  color: #fff;
  text-align: center;
  padding: 28rpx;
  border-radius: 56rpx;
  font-size: 32rpx;
  font-weight: 600;
}

.btn-back {
  text-align: center;
  color: #6a6a8a;
  font-size: 28rpx;
  padding: 12rpx;
}
</style>
