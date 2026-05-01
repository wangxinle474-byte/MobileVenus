<script setup lang="ts">
import { useModelStore } from "@/stores/modelStore";
import { MODEL_OPTIONS } from "@/types/inference";

const modelStore = useModelStore();
</script>

<template>
  <view class="model-selector">
    <text class="label">选择模型</text>
    <view class="options">
      <view
        v-for="opt in MODEL_OPTIONS"
        :key="opt.version"
        class="option"
        :class="{ active: modelStore.selectedVersion === opt.version }"
        @tap="modelStore.select(opt.version)"
      >
        <view class="option-header">
          <text class="option-name">{{ opt.label }}</text>
          <text class="option-tag">{{ opt.tag }}</text>
        </view>
        <view class="option-metrics">
          <text class="metric">PSNR {{ opt.psnr }}</text>
          <text class="metric">SSIM {{ opt.ssim }}</text>
        </view>
      </view>
    </view>
  </view>
</template>

<style lang="scss" scoped>
.model-selector {
  padding: 24rpx 0;

  .label {
    font-size: 28rpx;
    color: #6a6a8a;
    margin-bottom: 16rpx;
    display: block;
  }

  .options {
    display: flex;
    flex-direction: column;
    gap: 16rpx;
  }

  .option {
    background: #ffffff;
    border: 2rpx solid #e8e8f0;
    border-radius: 16rpx;
    padding: 20rpx 24rpx;
    transition: all 0.2s;

    &.active {
      border-color: #6c63ff;
      background: rgba(108, 99, 255, 0.06);
    }

    .option-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8rpx;
    }

    .option-name {
      font-size: 30rpx;
      font-weight: 600;
      color: #1a1a2e;
    }

    .option-tag {
      font-size: 22rpx;
      color: #6c63ff;
      background: rgba(108, 99, 255, 0.15);
      padding: 4rpx 12rpx;
      border-radius: 20rpx;
    }

    .option-metrics {
      display: flex;
      gap: 24rpx;
    }

    .metric {
      font-size: 24rpx;
      color: #6a6a8a;
    }
  }
}
</style>
