import type { JSX } from "preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { api } from "../../shared/api/client";
import type { LLMProviderStats } from "../../shared/api/types";
import {
  Alert,
  Button,
  ErrorState,
  LoadingState,
  StatusBadge,
} from "../../shared/ui";
import styles from "../features.module.css";

type Provider = LLMProviderStats["providers"][number];
type Credits = NonNullable<Provider["credits"]>;
type Money = Credits["remaining_credits"];
type ActivityTotals = NonNullable<Provider["activity"]>["totals"];

export function LLMProvidersView(): JSX.Element {
  const [data, setData] = useState<LLMProviderStats | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    try {
      setData(await api<LLMProviderStats>("/api/admin/llm-providers"));
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not load provider statistics.",
      );
    }
  }, []);

  useEffect(() => void load(), [load]);

  if (error) return <ErrorState message={error} retry={load} />;
  if (!data) return <LoadingState rows={8} />;

  const provider = data.providers[0];
  if (!provider) {
    return (
      <Alert title="No providers" error>
        <p>No LLM providers are configured.</p>
      </Alert>
    );
  }

  return (
    <div class={styles.stack}>
      <section class={styles.sectionHeader}>
        <div>
          <h2>LLM providers</h2>
          <p>{provider.provider_name}</p>
        </div>
        <div class={styles.actions}>
          <StatusBadge status={provider.status} />
          <Button type="button" variant="secondary" onClick={load}>
            Refresh
          </Button>
        </div>
      </section>

      {provider.issues.length ? (
        <Alert title="Partial provider data" error={provider.status === "unavailable"}>
          <ul class={styles.list}>
            {provider.issues.map((issue) => (
              <li key={issue.section}>
                <strong>{issue.section.replaceAll("_", " ")}</strong>:{" "}
                {issue.message}
              </li>
            ))}
          </ul>
        </Alert>
      ) : null}

      <section class={styles.grid4} aria-label="Provider metrics">
        <Metric
          label="Remaining credits"
          value={money(provider.credits?.remaining_credits)}
          detail={provider.credits ? "OpenRouter balance" : "Unavailable"}
        />
        <Metric
          label="Total usage"
          value={money(provider.credits?.total_usage)}
          detail={provider.credits ? "Purchased credits spent" : "Unavailable"}
        />
        <Metric
          label="Monthly key usage"
          value={money(provider.api_key?.usage_monthly)}
          detail={provider.api_key?.limit_reset ?? "Current UTC month"}
        />
        <Metric
          label="30-day requests"
          value={integer(provider.activity?.totals.requests)}
          detail={provider.activity?.window_label ?? "Unavailable"}
        />
      </section>

      <section class={styles.grid2}>
        <article class={styles.card}>
          <div class={styles.sectionHeader}>
            <div>
              <h3>API key</h3>
              <p>{provider.api_key?.label ?? "Unavailable"}</p>
            </div>
            {provider.api_key ? (
              <StatusBadge
                status={provider.api_key.is_management_key ? "management" : "runtime"}
              />
            ) : null}
          </div>
          {provider.api_key ? <KeyStats provider={provider} /> : unavailable()}
        </article>

        <article class={styles.card}>
          <div class={styles.sectionHeader}>
            <div>
              <h3>Configured models</h3>
              <p>{data.configuration.chat_provider}</p>
            </div>
          </div>
          <dl class={styles.metadata}>
            <Info label="Chat model" value={data.configuration.chat_model} />
            <Info
              label="Embedding model"
              value={`${data.configuration.embedding_provider} / ${data.configuration.embedding_model}`}
            />
          </dl>
          <ul class={styles.list}>
            {data.configuration.agent_models.map((agent) => (
              <li class={styles.listItem} key={agent.agent_name}>
                <div class={styles.itemHeader}>
                  <strong>{agent.agent_name}</strong>
                  <span class={styles.mono}>{agent.model_id}</span>
                </div>
              </li>
            ))}
          </ul>
        </article>
      </section>

      <section class={styles.grid2}>
        <ActivityList
          title="Top models"
          items={provider.activity?.top_models ?? []}
          name={(item) => item.model}
          detail={(item) => item.provider_name}
        />
        <ActivityList
          title="Top providers"
          items={provider.activity?.top_providers ?? []}
          name={(item) => item.provider_name}
          detail={(item) => item.last_activity_date ?? "No recent activity"}
        />
      </section>
    </div>
  );
}

function KeyStats({ provider }: { provider: Provider }): JSX.Element {
  const apiKey = provider.api_key;
  if (!apiKey) return unavailable();
  return (
    <dl class={styles.metadata}>
      <Info label="Daily usage" value={money(apiKey.usage_daily)} />
      <Info label="Weekly usage" value={money(apiKey.usage_weekly)} />
      <Info label="Monthly usage" value={money(apiKey.usage_monthly)} />
      <Info label="Total usage" value={money(apiKey.usage)} />
      <Info label="Limit" value={money(apiKey.limit)} />
      <Info label="Remaining limit" value={money(apiKey.limit_remaining)} />
      <Info label="BYOK monthly" value={money(apiKey.byok_usage_monthly)} />
      <Info label="Expires" value={dateTime(apiKey.expires_at)} />
    </dl>
  );
}

function ActivityList<T extends { totals: ActivityTotals; last_activity_date: string | null }>({
  title,
  items,
  name,
  detail,
}: {
  title: string;
  items: T[];
  name: (item: T) => string;
  detail: (item: T) => string;
}): JSX.Element {
  return (
    <article class={styles.card}>
      <div class={styles.sectionHeader}>
        <div>
          <h3>{title}</h3>
          <p>{items.length ? `${items.length} active` : "No recent activity"}</p>
        </div>
      </div>
      {items.length ? (
        <ul class={styles.list}>
          {items.map((item) => (
            <li class={styles.listItem} key={`${name(item)}-${detail(item)}`}>
              <div class={styles.itemHeader}>
                <div>
                  <strong>{name(item)}</strong>
                  <span class={styles.metricDetail}>{detail(item)}</span>
                </div>
                <strong>{money(item.totals.usage)}</strong>
              </div>
              <span class={styles.metricDetail}>
                {integer(item.totals.requests)} requests,{" "}
                {integer(tokenTotal(item.totals))} tokens
              </span>
            </li>
          ))}
        </ul>
      ) : (
        unavailable()
      )}
    </article>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}): JSX.Element {
  return (
    <article class={`${styles.card} ${styles.metric}`}>
      <span class={styles.metricLabel}>{label}</span>
      <strong class={styles.metricValue}>{value}</strong>
      <span class={styles.metricDetail}>{detail}</span>
    </article>
  );
}

function Info({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function unavailable(): JSX.Element {
  return <p class={styles.muted}>Unavailable</p>;
}

function tokenTotal(totals: ActivityTotals): number {
  return totals.prompt_tokens + totals.completion_tokens + totals.reasoning_tokens;
}

function money(value: Money | null | undefined): string {
  if (!value) return "Unavailable";
  const numeric = Number(value.amount);
  if (!Number.isFinite(numeric)) return `${value.currency} ${value.amount}`;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: value.currency,
    maximumFractionDigits: 4,
  }).format(numeric);
}

function integer(value: number | null | undefined): string {
  if (value === null || value === undefined) return "Unavailable";
  return new Intl.NumberFormat("en-US").format(value);
}

function dateTime(value: string | null | undefined): string {
  if (!value) return "No expiration";
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}
