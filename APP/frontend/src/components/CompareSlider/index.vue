<script setup lang="ts">
import { ref, computed, onMounted } from "vue";

defineProps<{ originalUrl: string; enhancedUrl: string }>();

const sliderX = ref(50);
const containerW = ref(0);
let startX = 0;

const enhancedOpacity = computed(() => sliderX.value / 100);
const labelText = computed(() => {
  if (sliderX.value < 5) return "原图";
  if (sliderX.value > 95) return "增强后";
  return `增强 ${Math.round(sliderX.value)}%`;
});

onMounted(() => {
  uni.createSelectorQuery()
    .select(".reveal-wrap")
    .boundingClientRect((rect: any) => {
      if (rect) containerW.value = rect.width;
    })
    .exec();
});

function onTouchStart(e: TouchEvent) {
  startX = e.touches[0].clientX;
}

function onTouchMove(e: TouchEvent) {
  const w = containerW.value || uni.getWindowInfo().windowWidth;
  const delta = e.touches[0].clientX - startX;
  startX = e.touches[0].clientX;
  const pct = sliderX.value + (delta / w) * 100;
  sliderX.value = Math.min(100, Math.max(0, pct));
}
</script>

<template>
  <view class="compare-wrap">
    <!-- 图像区：原图底层，增强图透明度叠加 -->
    <view
      class="reveal-wrap"
      @touchstart="onTouchStart"
      @touchmove="onTouchMove"
    >
      <image class="img" :src="originalUrl" mode="aspectFit" />
      <image
        class="img"
        :src="enhancedUrl"
        mode="aspectFit"
        :style="{ opacity: enhancedOpacity }"
      />
      <!-- 当前状态标签 -->
      <view class="state-badge">
        <text class="state-text">{{ labelText }}</text>
      </view>
    </view>

    <!-- 底部拖动轨道 -->
    <view
      class="track"
      @touchstart="onTouchStart"
      @touchmove="onTouchMove"
    >
      <view class="track-bg">
        <view class="track-fill" :style="{ width: `${sliderX}%` }" />
      </view>
      <view class="thumb" :style="{ left: `${sliderX}%` }">
        <text class="thumb-arrow">◀▶</text>
      </view>
    </view>

    <view class="track-labels">
      <text class="track-label">原图</text>
      <text class="track-label">增强后</text>
    </view>
  </view>
</template>

<style lang="scss" scoped>
.compare-wrap {
  width: 100%;
}

.reveal-wrap {
  position: relative;
  width: 100%;
  height: 480rpx;
  overflow: hidden;
  border-radius: 20rpx;
  background: #f0f0f8;

  .img {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }
}

.state-badge {
  position: absolute;
  top: 16rpx;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.55);
  border-radius: 20rpx;
  padding: 4rpx 20rpx;

  .state-text {
    color: #fff;
    font-size: 24rpx;
  }
}

.track {
  position: relative;
  height: 64rpx;
  display: flex;
  align-items: center;
  margin-top: 20rpx;
  padding: 0 28rpx;

  .track-bg {
    flex: 1;
    height: 8rpx;
    background: #e8e8f0;
    border-radius: 4rpx;
    overflow: hidden;

    .track-fill {
      height: 100%;
      background: linear-gradient(90deg, #6c63ff, #a78bfa);
      border-radius: 4rpx;
      transition: width 0.05s linear;
    }
  }

  .thumb {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 56rpx;
    height: 56rpx;
    border-radius: 50%;
    background: #ffffff;
    border: 4rpx solid #6c63ff;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4rpx 12rpx rgba(108, 99, 255, 0.4);

    .thumb-arrow {
      font-size: 18rpx;
      color: #6c63ff;
      line-height: 1;
    }
  }
}

.track-labels {
  display: flex;
  justify-content: space-between;
  padding: 0 28rpx;
  margin-top: 4rpx;

  .track-label {
    font-size: 22rpx;
    color: #606078;
  }
}
</style>
