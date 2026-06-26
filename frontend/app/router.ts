import { useEffect, useState } from "preact/hooks";

export type Route = "overview" | "documents" | "pipeline" | "assistant" | "retrieval";

const routes: Record<string, Route> = {
  "/": "overview",
  "/overview": "overview",
  "/documents": "documents",
  "/pipeline": "pipeline",
  "/assistant": "assistant",
  "/retrieval": "retrieval",
};

export function currentRoute(): Route {
  return routes[window.location.pathname] ?? "overview";
}

export function navigate(path: string): void {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(currentRoute());
  useEffect(() => {
    const update = (): void => setRoute(currentRoute());
    window.addEventListener("popstate", update);
    return () => window.removeEventListener("popstate", update);
  }, []);
  return route;
}
