<script setup lang="ts">
defineProps<{ message?: string }>();
</script>

<template>
  <view class="loading-state">
    <view class="spinner" />
    <text class="message">{{ message ?? "AI 增强中…" }}</text>
  </view>
</template>

<style lang="scss" scoped>
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;

  .spinner {
    width: 72rpx;
    height: 72rpx;
    border: 6rpx solid #e8e8f0;
    border-top-color: #6c63ff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 24rpx;
  }

  .message {
    font-size: 28rpx;
    color: #6a6a8a;
  }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
