<script setup lang="ts">
const emit = defineEmits<{ (e: "pick", path: string): void }>();

function onTap() {
  uni.chooseImage({
    count: 1,
    sizeType: ["original", "compressed"],
    sourceType: ["album", "camera"],
    success: (res) => emit("pick", res.tempFilePaths[0]),
  });
}
</script>

<template>
  <view class="upload-entry" @tap="onTap">
    <view class="icon-wrap">
      <text class="icon">＋</text>
    </view>
    <text class="hint">点击选择或拍摄图片</text>
    <text class="sub-hint">支持 JPEG / PNG，最大 10MB</text>
  </view>
</template>

<style lang="scss" scoped>
.upload-entry {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 360rpx;
  border: 3rpx dashed #c8c8e8;
  border-radius: 24rpx;
  background: #f8f8ff;

  .icon-wrap {
    width: 96rpx;
    height: 96rpx;
    border-radius: 50%;
    background: rgba(108, 99, 255, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 20rpx;
  }

  .icon {
    font-size: 56rpx;
    color: #6c63ff;
    line-height: 1;
  }

  .hint {
    font-size: 30rpx;
    color: #1a1a2e;
    margin-bottom: 8rpx;
  }

  .sub-hint {
    font-size: 24rpx;
    color: #9090a8;
  }
}
</style>
