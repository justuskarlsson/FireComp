import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

const cesium = "node_modules/cesium/Build/Cesium";
const cesiumBase = "cesiumStatic";
const cesiumRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  cesium,
);

const MIME: Record<string, string> = {
  ".css": "text/css",
  ".gif": "image/gif",
  ".glsl": "text/plain",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".js": "text/javascript",
  ".json": "application/json",
  ".ktx": "image/ktx",
  ".ktx2": "image/ktx2",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".wasm": "application/wasm",
  ".webp": "image/webp",
};

/** vite-plugin-static-copy only writes to dist; Vite 8 SPA-falls-back missing /cesiumStatic to index.html. */
function serveCesiumDev(): Plugin {
  return {
    name: "serve-cesium-dev",
    configureServer(server) {
      server.middlewares.use("/cesiumStatic", (req, res, next) => {
        const rel = decodeURIComponent((req.url ?? "").split("?")[0]).replace(/^\/+/, "");
        const file = path.resolve(cesiumRoot, rel);
        if (!file.startsWith(cesiumRoot + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
          next();
          return;
        }
        res.setHeader("Content-Type", MIME[path.extname(file).toLowerCase()] ?? "application/octet-stream");
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  define: {
    CESIUM_BASE_URL: JSON.stringify(`/${cesiumBase}`),
  },
  plugins: [
    serveCesiumDev(),
    viteStaticCopy({
      targets: [
        { src: `${cesium}/ThirdParty`, dest: cesiumBase },
        { src: `${cesium}/Workers`, dest: cesiumBase },
        { src: `${cesium}/Assets`, dest: cesiumBase },
        { src: `${cesium}/Widgets`, dest: cesiumBase },
      ],
    }),
  ],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
