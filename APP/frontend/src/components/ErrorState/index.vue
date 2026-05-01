<script setup lang="ts">
defineProps<{ message?: string }>();
const emit = defineEmits<{ (e: "retry"): void }>();
</script>

<template>
  <view class="error-state">
    <text class="icon">⚠</text>
    <text class="message">{{ message ?? "出现错误，请重试" }}</text>
    <view class="retry-btn" @tap="emit('retry')">重试</view>
  </view>
</template>

<style lang="scss" scoped>
.error-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 60rpx 0;

  .icon {
    font-size: 64rpx;
    margin-bottom: 16rpx;
    color: #ff6b6b;
  }

  .message {
    font-size: 28rpx;
    color: #6a6a8a;
    text-align: center;
    padding: 0 40rpx;
    margin-bottom: 32rpx;
  }

  .retry-btn {
    background: #6c63ff;
    color: #fff;
    padding: 18rpx 48rpx;
    border-radius: 48rpx;
    font-size: 28rpx;
  }
}
</style>
