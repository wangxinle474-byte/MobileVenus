<script setup lang="ts">
import { onLaunch } from "@dcloudio/uni-app";
import { useAppStore } from "@/stores/app";

onLaunch(() => {
  const appStore = useAppStore();
  appStore.init();
  uni.request({
    url: 'http://127.0.0.1:8000/api/v1/health/',
    method: 'GET',
    success: (res) => {
      console.log('[health:ok]', res.statusCode, JSON.stringify(res.data));
    },
    fail: (err) => {
      console.error('[health:fail]', JSON.stringify(err));
    },
  });
});
</script>

<style lang="scss">
page {
  background-color: #0f0f1a;
  color: #ffffff;
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif;
}
</style>
