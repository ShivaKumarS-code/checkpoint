import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const daemon = "http://127.0.0.1:4040";

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../backend/checkpoint/ui_dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": daemon,
      "/v1": daemon,
      "/ws": { target: daemon.replace("http", "ws"), ws: true },
    },
  },
});
