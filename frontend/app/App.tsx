import type { JSX } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { AssistantView } from "../features/assistant/AssistantView";
import { LLMProvidersView } from "../features/admin/LLMProvidersView";
import { DocumentsView } from "../features/documents/DocumentsView";
import { OverviewView } from "../features/overview/OverviewView";
import { PipelineView } from "../features/pipeline/PipelineView";
import { RetrievalView } from "../features/retrieval/RetrievalView";
import { Button, ErrorState, IconButton } from "../shared/ui";
import { navigate, type Route, useRoute } from "./router";
import styles from "./app.module.css";

const labels: Record<Route, string> = {
  overview: "Workflow overview",
  documents: "Documents",
  pipeline: "Pipeline",
  assistant: "Medical assistant",
  retrieval: "Retrieval inspector",
  "llm-providers": "LLM providers",
};

const icons: Record<Route, string> = {
  overview: "⌂",
  documents: "▤",
  pipeline: "↻",
  assistant: "✦",
  retrieval: "⌕",
  "llm-providers": "$",
};

const primaryRoutes: Route[] = [
  "overview",
  "documents",
  "pipeline",
  "assistant",
  "retrieval",
];

interface AppProps {
  username: string;
  isAdmin: boolean;
}

export function App({ username, isAdmin }: AppProps): JSX.Element {
  const route = useRoute();
  const [menuOpen, setMenuOpen] = useState(false);
  const sidebarRef = useRef<HTMLElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const sidebar = sidebarRef.current;
    const focusable = sidebar?.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    focusable?.[0]?.focus();
    const handleKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        setMenuOpen(false);
        return;
      }
      if (event.key !== "Tab" || !focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("keydown", handleKey);
      menuButtonRef.current?.focus();
    };
  }, [menuOpen]);

  const go = (next: Route): void => {
    navigate(next === "overview" ? "/" : `/${next}`);
    setMenuOpen(false);
  };
  const routes = isAdmin
    ? [...primaryRoutes, "llm-providers" as Route]
    : primaryRoutes;

  return (
    <div class={styles.shell}>
      {menuOpen ? (
        <button
          class={styles.mobileBackdrop}
          aria-label="Close navigation"
          onClick={() => setMenuOpen(false)}
        />
      ) : null}
      <aside
        id="primary-navigation"
        ref={sidebarRef}
        class={`${styles.sidebar} ${menuOpen ? styles.sidebarOpen : ""}`}
      >
        <div class={styles.brand}>
          <span class={styles.brandMark}>M</span>
          <div>
            <strong>Medic RAG</strong>
            <span>Document intelligence</span>
          </div>
        </div>
        <nav class={styles.nav} aria-label="Primary navigation">
          {routes.map((item) => (
            <button
              type="button"
              key={item}
              aria-label={labels[item]}
              class={`${styles.navLink} ${route === item ? styles.navActive : ""}`}
              aria-current={route === item ? "page" : undefined}
              onClick={() => go(item)}
            >
              <span class={styles.navIcon} aria-hidden="true">
                {icons[item]}
              </span>
              <span class={styles.navText}>{labels[item]}</span>
            </button>
          ))}
          {isAdmin ? (
            <a class={styles.navLink} href="/admin" aria-label="Admin">
              <span class={styles.navIcon} aria-hidden="true">⚙</span>
              <span class={styles.navText}>Admin</span>
            </a>
          ) : null}
        </nav>
        <footer class={styles.sidebarFooter}>
          <div class={styles.user}>
            <strong>{username}</strong>
            <span>Signed in</span>
          </div>
          <form class={styles.logout} method="post" action="/logout">
            <input
              type="hidden"
              name="csrf_token"
              value={
                document.querySelector<HTMLMetaElement>('meta[name="csrf-token"]')
                  ?.content ?? ""
              }
            />
            <Button type="submit" variant="secondary">
              Log out
            </Button>
          </form>
        </footer>
      </aside>
      <main class={styles.main}>
        <header class={styles.topbar}>
          <IconButton
            ref={menuButtonRef}
            class={styles.mobileMenu}
            label="Open navigation"
            aria-controls="primary-navigation"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen(true)}
          >
            ☰
          </IconButton>
          <h1 class={styles.title}>{labels[route]}</h1>
          <span class={styles.health}>Live operational workspace</span>
        </header>
        <div class={styles.content}>{view(route, isAdmin)}</div>
      </main>
    </div>
  );
}

function view(route: Route, isAdmin: boolean): JSX.Element {
  switch (route) {
    case "documents":
      return <DocumentsView />;
    case "pipeline":
      return <PipelineView />;
    case "assistant":
      return <AssistantView />;
    case "retrieval":
      return <RetrievalView />;
    case "llm-providers":
      return isAdmin ? (
        <LLMProvidersView />
      ) : (
        <ErrorState
          title="Admin access required"
          message="This page is restricted."
        />
      );
    default:
      return <OverviewView />;
  }
}
