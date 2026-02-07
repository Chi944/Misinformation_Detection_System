import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/predict": "http://127.0.0.1:5000",
      "/predict/batch": "http://127.0.0.1:5000",
      "/health": "http://127.0.0.1:5000",
      "/models": "http://127.0.0.1:5000",
      "/explain": "http://127.0.0.1:5000",
    },
  },
});
