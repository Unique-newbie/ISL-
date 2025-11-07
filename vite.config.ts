import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "0.0.0.0"
  },
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version ?? "dev")
  }
});
