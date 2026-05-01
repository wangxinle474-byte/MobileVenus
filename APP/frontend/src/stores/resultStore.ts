import { defineStore } from "pinia";
import { ref } from "vue";
import type { EnhanceResult } from "@/types/api";
import type { AestheticPollResult } from "@/services/api";

export const useResultStore = defineStore("result", () => {
  const result = ref<EnhanceResult | null>(null);
  const requestId = ref<string | null>(null);

  function setResult(data: EnhanceResult) {
    result.value = data;
    requestId.value = data.request_id;
  }

  function patchAesthetic(data: AestheticPollResult) {
    if (!result.value) return;
    result.value.aesthetic_status = data.aesthetic_status;
    result.value.aesthetic_before = data.aesthetic_before;
    result.value.aesthetic_after = data.aesthetic_after;
  }

  function clear() {
    result.value = null;
    requestId.value = null;
  }

  return { result, requestId, setResult, patchAesthetic, clear };
});
