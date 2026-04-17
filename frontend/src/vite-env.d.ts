/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL for the FastAPI backend when using `npm run dev` (e.g. http://127.0.0.1:9000). */
  readonly VITE_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
