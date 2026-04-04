import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  build: {
    outDir: "../../crates/picmash-app/assets/web",
    emptyOutDir: true,
    cssCodeSplit: false,
    rollupOptions: {
      input: "src/main.ts",
      output: {
        entryFileNames: "picmash-client.js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: (assetInfo) =>
          assetInfo.name?.endsWith(".css")
            ? "picmash-client.css"
            : "assets/[name]-[hash][extname]",
      },
    },
  },
});
