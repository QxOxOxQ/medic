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
    throw new ApiError(
      String(payload.detail ?? payload.error ?? `HTTP ${response.status}`),
      response.status,
    );
  }
  return payload as T;
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
