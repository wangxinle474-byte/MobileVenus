import { defineStore } from "pinia";
import { ref } from "vue";
import type { ModelVersion } from "@/types/api";

export const useModelStore = defineStore("model", () => {
  const selectedVersion = ref<ModelVersion>("distill_v4");

  function select(version: ModelVersion) {
    selectedVersion.value = version;
  }

  return { selectedVersion, select };
});
