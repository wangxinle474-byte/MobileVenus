import { defineConfig, type Plugin } from "vite";
import uni from "@dcloudio/vite-plugin-uni";
import { resolve } from "path";

const VUE_SHIM_ID = "\0vue-uni-shim";

function uniAppVueShim(): Plugin {
  return {
    name: "uni-app-vue-shim",
    enforce: "pre",
    resolveId(id, importer) {
      if (id === "vue" && importer && importer.includes("uni-app.es.js")) {
        return VUE_SHIM_ID;
      }
    },
    load(id) {
      if (id !== VUE_SHIM_ID) return;
      return `
export * from "vue";
export let isInSSRComponentSetup = false;
export function injectHook(type, hook, target) {
  if (target) {
    const hooks = target[type] || (target[type] = []);
    hooks.push(hook);
    return hook;
  }
}
`;
    },
  };
}

export default defineConfig({
  plugins: [uniAppVueShim(), uni()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  css: {
    preprocessorOptions: {
      scss: {
        api: "modern-compiler",
      },
    },
  },
});
