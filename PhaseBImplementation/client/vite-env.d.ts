/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PUBLIC_ENGS_API_URL: string;
  // other env variables can be added here if needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
