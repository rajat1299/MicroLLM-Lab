import React from "react";
import ReactDOM from "react-dom/client";
import * as Sentry from "@sentry/browser";

import App from "./App";
import "./styles.css";

const dsn = import.meta.env.VITE_SENTRY_DSN;
if (dsn) {
  Sentry.init({ dsn, tracesSampleRate: 0.0 });
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
