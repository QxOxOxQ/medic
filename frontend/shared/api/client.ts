function csrfToken(): string {
  return (
    document.querySelector<HTMLMetaElement>('meta[name="csrf-token"]')?.content ?? ""
  );
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.method && options.method !== "GET") {
    headers.set("X-CSRF-Token", csrfToken());
  }
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers,
  });
  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? ((await response.json()) as Record<string, unknown>)
    : { detail: await response.text() };
  if (!response.ok) {
    const detail = payload.detail ?? payload.error;
    throw new ApiError(
      detail ? String(detail) : statusMessage(response.status),
      response.status,
    );
  }
  return payload as T;
}

function statusMessage(status: number): string {
  if (status === 401 || status === 403) {
    return "Your session has expired. Please sign in again.";
  }
  if (status === 404) {
    return "We couldn't find what you were looking for.";
  }
  if (status === 429) {
    return "Too many requests right now. Please wait a moment and retry.";
  }
  if (status >= 500) {
    return "The service is temporarily unavailable. Please try again shortly.";
  }
  return "The request couldn't be completed. Please try again.";
}

export function jsonRequest(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export function uploadRequest(files: File[]): RequestInit {
  const data = new FormData();
  for (const file of files) {
    data.append("file", file);
  }
  data.append("csrf_token", csrfToken());
  return { method: "POST", body: data };
}
