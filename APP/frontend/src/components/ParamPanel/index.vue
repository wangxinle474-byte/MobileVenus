<script setup lang="ts">
import type { InferenceParams, ParameterConfidence } from "@/types/api";
import { PARAM_LABELS, CONFIDENCE_LABEL } from "@/types/inference";

defineProps<{
  params: InferenceParams;
  confidence: ParameterConfidence;
}>();
</script>

<template>
  <view class="param-panel">
    <text class="title">预测参数</text>
    <view class="param-list">
      <view
        v-for="(label, key) in PARAM_LABELS"
        :key="key"
        class="param-row"
      >
        <view class="param-left">
          <text class="param-label">{{ label }}</text>
          <text
            class="conf-badge"
            :class="confidence[key as keyof ParameterConfidence]"
          >
            {{ CONFIDENCE_LABEL[confidence[key as keyof ParameterConfidence]] }}
          </text>
        </view>
        <text class="param-value">
          {{ key === 'white_balance'
            ? `${params[key as keyof InferenceParams]}K`
            : params[key as keyof InferenceParams] }}
        </text>
      </view>
    </view>
  </view>
</template>

<style lang="scss" scoped>
.param-panel {
  background: #ffffff;
  box-shadow: 0 2rpx 12rpx rgba(0,0,0,0.06);
  border-radius: 20rpx;
  padding: 28rpx;

  .title {
    font-size: 30rpx;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 20rpx;
    display: block;
  }

  .param-list {
    display: flex;
    flex-direction: column;
    gap: 18rpx;
  }

  .param-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .param-left {
    display: flex;
    align-items: center;
    gap: 12rpx;
    min-width: 0;
    overflow: hidden;
  }

  .param-label {
    font-size: 26rpx;
    color: #4a4a6a;
  }

  .conf-badge {
    font-size: 20rpx;
    padding: 2rpx 10rpx;
    border-radius: 16rpx;

    &.high { background: rgba(0, 212, 170, 0.15); color: #00d4aa; }
    &.medium { background: rgba(255, 209, 102, 0.15); color: #ffd166; }
    &.reference_only { background: rgba(160, 160, 184, 0.15); color: #a0a0b8; }
  }

  .param-value {
    font-size: 28rpx;
    font-weight: 600;
    color: #6c63ff;
    white-space: nowrap;
    flex-shrink: 0;
    margin-left: 16rpx;
  }
}
</style>
