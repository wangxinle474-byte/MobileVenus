<script setup lang="ts">
import { ref } from "vue";
import { useAppStore } from "@/stores/app";
import { useModelStore } from "@/stores/modelStore";
import { useResultStore } from "@/stores/resultStore";
import { enhance } from "@/services/inference";
import { getErrorMessage } from "@/utils/error";
import ModelSelector from "@/components/ModelSelector/index.vue";
import UploadEntry from "@/components/UploadEntry/index.vue";
import LoadingState from "@/components/LoadingState/index.vue";
import ErrorState from "@/components/ErrorState/index.vue";

const appStore = useAppStore();
const modelStore = useModelStore();
const resultStore = useResultStore();

const previewPath = ref<string>("");

async function onImagePick(path: string) {
  previewPath.value = path;
}

async function onEnhance() {
  if (!previewPath.value) return;
  appStore.setStatus("uploading");

  try {
    appStore.setStatus("processing");
    const result = await enhance(
      previewPath.value,
      modelStore.selectedVersion,
      appStore.clientId
    );
    resultStore.setResult(result);
    appStore.setStatus("success");
    uni.navigateTo({ url: "/pages/result/index" });
  } catch (err: unknown) {
    console.error('[onEnhance:error]', JSON.stringify(err), err);
    const code = (err as { code?: number }).code ?? 0;
    appStore.setError(getErrorMessage(code, "处理失败，请重试"));
  }
}

function onRetry() {
  previewPath.value = "";
  appStore.reset();
}
</script>

<template>
  <scroll-view class="page" scroll-y>
    <!-- 标题 -->
    <view class="header">
      <text class="title">Intelligence Camera</text>
      <text class="subtitle">语义蒸馏 · AI 图像增强</text>
    </view>

    <!-- 加载中 -->
    <loading-state v-if="appStore.status === 'processing'" message="AI 推理中，请稍候…" />

    <!-- 错误 -->
    <error-state
      v-else-if="appStore.status === 'error'"
      :message="appStore.errorMessage ?? undefined"
      @retry="onRetry"
    />

    <!-- 主内容 -->
    <template v-else>
      <!-- 图片预览 / 上传入口 -->
      <view class="section">
        <image
          v-if="previewPath"
          class="preview-img"
          :src="previewPath"
          mode="aspectFit"
          @tap="onImagePick"
        />
        <upload-entry v-else @pick="onImagePick" />
      </view>

      <!-- 模型选择 -->
      <view class="section">
        <model-selector />
      </view>

      <!-- 操作按钮 -->
      <view class="actions">
        <view
          class="btn-enhance"
          :class="{ disabled: !previewPath }"
          @tap="onEnhance"
        >
          开始增强
        </view>
        <view v-if="previewPath" class="btn-reselect" @tap="onRetry">
          重新选择
        </view>
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
  padding: 40rpx 32rpx 80rpx;
}

.header {
  text-align: center;
  padding: 32rpx 0 40rpx;

  .title {
    display: block;
    font-size: 48rpx;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: 2rpx;
  }

  .subtitle {
    display: block;
    font-size: 26rpx;
    color: #6c63ff;
    margin-top: 8rpx;
  }
}

.section {
  margin-bottom: 32rpx;
}

.preview-img {
  width: 100%;
  height: 480rpx;
  border-radius: 20rpx;
  background: #f0f0f8;
}

.actions {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
  margin-top: 8rpx;
}

.btn-enhance {
  background: linear-gradient(135deg, #6c63ff, #8b85ff);
  color: #fff;
  text-align: center;
  padding: 28rpx;
  border-radius: 56rpx;
  font-size: 32rpx;
  font-weight: 600;

  &.disabled {
    background: #e8e8f0;
    color: #9090a8;
  }
}

.btn-reselect {
  text-align: center;
  color: #6a6a8a;
  font-size: 28rpx;
  padding: 12rpx;
}
</style>
