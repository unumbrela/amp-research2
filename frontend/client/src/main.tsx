import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

function mountAnalytics() {
  const endpoint = import.meta.env.VITE_ANALYTICS_ENDPOINT?.trim();
  const websiteId = import.meta.env.VITE_ANALYTICS_WEBSITE_ID?.trim();

  if (!endpoint || !websiteId) return;

  const scriptId = "umami-analytics-script";
  if (document.getElementById(scriptId)) return;

  const script = document.createElement("script");
  script.id = scriptId;
  script.defer = true;
  script.src = `${endpoint.replace(/\/+$/, "")}/umami`;
  script.setAttribute("data-website-id", websiteId);
  document.body.appendChild(script);
}

mountAnalytics();

createRoot(document.getElementById("root")!).render(<App />);
