import { defineConfig } from "vitest/config";
import preact from "@preact/preset-vite";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [preact()],
  build: {
    outDir: "dashboard/static/dist",
    emptyOutDir: true,
    manifest: "manifest.json",
    rollupOptions: {
      input: resolve(import.meta.dirname, "frontend/main.tsx"),
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["frontend/test/setup.ts"],
    css: true,
    include: ["frontend/test/**/*.test.tsx"],
  },
});
