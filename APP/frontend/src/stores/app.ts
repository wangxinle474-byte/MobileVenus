import { defineStore } from "pinia";
import { ref } from "vue";
import type { PageStatus } from "@/types/inference";
import { generateClientId, getClientId } from "@/utils/image";

export const useAppStore = defineStore("app", () => {
  const status = ref<PageStatus>("idle");
  const clientId = ref<string>("");
  const errorMessage = ref<string | null>(null);

  function init() {
    clientId.value = getClientId();
  }

  function setStatus(s: PageStatus) {
    status.value = s;
    if (s !== "error") errorMessage.value = null;
  }

  function setError(msg: string) {
    status.value = "error";
    errorMessage.value = msg;
  }

  function reset() {
    status.value = "idle";
    errorMessage.value = null;
  }

  return { status, clientId, errorMessage, init, setStatus, setError, reset };
});
