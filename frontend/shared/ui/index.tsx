import type { ComponentChildren, JSX } from "preact";
import { useEffect, useRef } from "preact/hooks";
import styles from "./ui.module.css";

type ButtonVariant = "primary" | "secondary" | "danger" | "ghost";

export function Button(
  props: JSX.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: ButtonVariant;
  },
): JSX.Element {
  const { variant = "primary", class: className, ...rest } = props;
  return (
    <button
      {...rest}
      class={`${styles.button} ${styles[variant]} ${className ?? ""}`}
    />
  );
}

export function IconButton(
  props: JSX.ButtonHTMLAttributes<HTMLButtonElement> & { label: string },
): JSX.Element {
  const { label, type = "button", ...rest } = props;
  return (
    <Button
      {...rest}
      type={type}
      class={styles.iconButton}
      variant="ghost"
      aria-label={label}
      title={label}
    />
  );
}

function statusTone(status: string): string {
  if (["indexed", "succeeded", "ready"].includes(status)) {
    return styles.positive ?? "";
  }
  if (["failed", "stale", "interrupted"].includes(status)) {
    return styles.negative ?? "";
  }
  if (["running", "queued"].includes(status)) return styles.active ?? "";
  return styles.warning ?? "";
}

export function StatusBadge({ status }: { status: string }): JSX.Element {
  return (
    <span class={`${styles.badge} ${statusTone(status)}`}>
      {status.replaceAll("_", " ")}
    </span>
  );
}

export function Alert({
  title,
  children,
  error = false,
}: {
  title: string;
  children: ComponentChildren;
  error?: boolean;
}): JSX.Element {
  return (
    <section
      class={`${styles.alert} ${error ? styles.alertError : ""}`}
      role={error ? "alert" : "status"}
    >
      <strong>{title}</strong>
      <div>{children}</div>
    </section>
  );
}

export function EmptyState({
  title,
  children,
  action,
}: {
  title: string;
  children: ComponentChildren;
  action?: ComponentChildren;
}): JSX.Element {
  return (
    <section class={styles.empty}>
      <h3>{title}</h3>
      <p>{children}</p>
      {action}
    </section>
  );
}

export function LoadingState({ rows = 4 }: { rows?: number }): JSX.Element {
  return (
    <div role="status" aria-label="Loading" aria-busy="true" aria-live="polite">
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton
          style={{ marginBottom: "12px", width: `${90 - index * 7}%` }}
          key={index}
        />
      ))}
    </div>
  );
}

export function Skeleton(
  props: JSX.HTMLAttributes<HTMLDivElement>,
): JSX.Element {
  const { class: className, ...rest } = props;
  return <div {...rest} class={`${styles.skeleton} ${className ?? ""}`} />;
}

export function Toast({
  children,
  error = false,
}: {
  children: ComponentChildren;
  error?: boolean;
}): JSX.Element {
  return (
    <div
      class={`${styles.toast} ${error ? styles.toastError : ""}`}
      role={error ? "alert" : "status"}
    >
      {children}
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  message,
  retry,
}: {
  title?: string;
  message: string;
  retry?: () => void;
}): JSX.Element {
  return (
    <Alert title={title} error>
      <p>{message}</p>
      {retry ? (
        <Button type="button" variant="secondary" onClick={retry}>
          Try again
        </Button>
      ) : null}
    </Alert>
  );
}

function useModalFocus(
  ref: { current: HTMLElement | null },
  close: () => void,
): void {
  const previousFocus = useRef<HTMLElement | null>(null);
  const closeRef = useRef(close);
  useEffect(() => {
    closeRef.current = close;
  }, [close]);
  useEffect(() => {
    previousFocus.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const element = ref.current;
    const firstFocusable = focusableElements(element)[0];
    (firstFocusable ?? element)?.focus();

    const handler = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        closeRef.current();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = focusableElements(ref.current);
      if (!focusable.length) {
        event.preventDefault();
        ref.current?.focus();
        return;
      }
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
    window.addEventListener("keydown", handler);
    return () => {
      window.removeEventListener("keydown", handler);
      previousFocus.current?.focus();
    };
  }, [ref]);
}

function focusableElements(root: HTMLElement | null): HTMLElement[] {
  if (!root) return [];
  return Array.from(
    root.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  );
}

export function Dialog({
  title,
  children,
  close,
  actions,
}: {
  title: string;
  children: ComponentChildren;
  close: () => void;
  actions?: ComponentChildren;
}): JSX.Element {
  const ref = useRef<HTMLDivElement>(null);
  useModalFocus(ref, close);
  return (
    <div class={styles.dialogBackdrop} onMouseDown={close}>
      <div
        class={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        ref={ref}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header class={styles.dialogHeader}>
          <h2>{title}</h2>
          <IconButton label="Close dialog" onClick={close}>
            ×
          </IconButton>
        </header>
        {children}
        {actions ? <footer class={styles.dialogActions}>{actions}</footer> : null}
      </div>
    </div>
  );
}

export function Drawer({
  title,
  children,
  close,
}: {
  title: string;
  children: ComponentChildren;
  close: () => void;
}): JSX.Element {
  const ref = useRef<HTMLElement>(null);
  useModalFocus(ref, close);
  return (
    <div class={styles.drawerBackdrop} onMouseDown={close}>
      <aside
        class={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        ref={ref}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header class={styles.drawerHeader}>
          <h2>{title}</h2>
          <IconButton label="Close drawer" onClick={close}>
            ×
          </IconButton>
        </header>
        {children}
      </aside>
    </div>
  );
}

export function Tabs<T extends string>({
  value,
  tabs,
  onChange,
}: {
  value: T;
  tabs: Array<{ value: T; label: string }>;
  onChange: (value: T) => void;
}): JSX.Element {
  return (
    <div class={styles.tabs} role="tablist" aria-label="Detail sections">
      {tabs.map((tab, index) => (
        <button
          key={tab.value}
          data-tab-index={index}
          type="button"
          role="tab"
          aria-selected={value === tab.value}
          tabIndex={value === tab.value ? 0 : -1}
          class={`${styles.tab} ${value === tab.value ? styles.tabActive : ""}`}
          onClick={() => onChange(tab.value)}
          onKeyDown={(event) => {
            const nextIndex = tabIndexFromKey(event.key, index, tabs.length);
            if (nextIndex === null) return;
            event.preventDefault();
            onChange(tabs[nextIndex]?.value ?? tab.value);
            const buttons =
              event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>(
                '[role="tab"]',
              );
            buttons?.[nextIndex]?.focus();
          }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function tabIndexFromKey(
  key: string,
  current: number,
  length: number,
): number | null {
  if (key === "Home") return 0;
  if (key === "End") return length - 1;
  if (key === "ArrowRight") return (current + 1) % length;
  if (key === "ArrowLeft") return (current - 1 + length) % length;
  return null;
}
